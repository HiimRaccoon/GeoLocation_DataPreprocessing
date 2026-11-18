"""Augmentation freeze and crash-recoverable publication."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from ..artifacts import (ACCEPTED_POOLS, AUGMENTATION_POLICY, BENCHMARK_ID, _resolve_artifact,
                         atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, canonical_jsonl_bytes,
                         read_benchmark_lock, resolve_cleaned_image, sha256_file, verify_stage)
from ..images import decode_rgb, pixel_sha256
from ..split import GROUP_FIELDS, SPLIT_FIELDS
from .diagnostics import log_dino_diagnostics, membership_sha256
from .policy import next_candidate
from .qc import build_qc_index, check_candidate


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_holdout_collisions(root, rows):
    """Treat exact holdout collisions as incidents even for rejected candidates."""
    root = Path(root).resolve()
    holdout = _csv(root / "splits/val.csv") + _csv(root / "splits/test.csv")
    file_hashes = {row["sha256"] for row in holdout}
    pixel_hashes = {row["decoded_pixel_sha256"] for row in holdout}
    for row in rows:
        path = _resolve_artifact(root, row.get("synthetic_path"))
        if not path.is_file():
            if row.get("accepted") is True:
                raise ValueError("accepted candidate file is missing")
            continue
        if sha256_file(path) in file_hashes:
            raise ValueError("holdout collision detected")
        try:
            decoded = decode_rgb(path)
        except (OSError, ValueError):
            if row.get("accepted") is True:
                raise ValueError("accepted candidate is unreadable")
            continue
        if pixel_sha256(decoded) in pixel_hashes:
            raise ValueError("holdout collision detected")


def validate_dino_changes(before, after):
    allowed = {"dino_source_similarity", "dino_nearest_sibling_similarity", "dino_nearest_sibling_id", "dino_diagnostic_error"}
    projection = lambda row: {key: value for key, value in row.items() if key not in allowed}
    if len(before) != len(after) or any(projection(a) != projection(b) for a, b in zip(before, after)):
        raise ValueError("DINO changed non-diagnostic candidate fields")


def _candidate_records(path):
    rows = _jsonl(path)
    seen = set()
    for index, row in enumerate(rows, 1):
        label = f"candidate record {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be a JSON object")
        if row.get("benchmark_id") != BENCHMARK_ID:
            raise ValueError(f"{label}: incompatible candidate benchmark identity (benchmark_id)")
        status = row.get("filter_status")
        if row.get("augmentation_policy") != AUGMENTATION_POLICY or status == "pending":
            raise ValueError("stale candidate history: archive legacy T039 artifacts and refresh the split lock")
        for field in ("synthetic_id", "source_image_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}: invalid or missing {field}")
        if row.get("augmentation_type") not in ("traditional", "da_fusion"):
            raise ValueError(f"{label}: invalid or missing augmentation_type")
        for field in ("class_id", "attempt_index"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{label}: invalid or missing {field}")
        if status not in ("accepted", "rejected") or row.get("accepted") is not (status == "accepted"):
            raise ValueError(f"{label}: inconsistent or missing filter_status/accepted")
        identifier = row["synthetic_id"]
        if identifier in seen:
            raise ValueError(f"{label}: duplicate candidate IDs")
        seen.add(identifier)
    return rows


def _group_order(group):
    return group["group"] == "tail", int(group["class_id"])


def _recover_interrupted_freeze(root):
    journal_path = root / "augmentation/manifests/.freeze-transaction/journal.json"
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    lock = read_benchmark_lock(root)
    if lock.get("augmentation_policy") != AUGMENTATION_POLICY:
        raise ValueError("stale interrupted freeze: archive legacy T039 artifacts and refresh the split lock")
    if lock.get("stage") == "augmented":
        verify_stage(root, "augmented")
        journal_path.unlink()
        return
    if lock.get("stage") != "membership" or journal.get("membership_sha256") != lock.get("membership_sha256"):
        raise ValueError("invalid interrupted freeze journal")
    fixed = {"augmentation/manifests/candidates.jsonl": "candidates.old",
             "splits/class_groups.csv": "groups.old", "splits/split_metadata.json": "metadata.old"}
    if set(journal.get("files", {})) != set(fixed):
        raise ValueError("invalid interrupted freeze file list")
    for target, backup in fixed.items():
        info = journal["files"][target]
        old = journal_path.parent / backup
        current = root / target
        if (not old.is_file() or sha256_file(old) != info.get("old") or
            info.get("old") != lock.get("artifacts", {}).get(target) or
            not current.is_file() or sha256_file(current) not in (info.get("old"), info.get("new"))):
            raise ValueError("interrupted freeze cannot be safely recovered")
    for target, backup in fixed.items():
        atomic_write_bytes(root / target, (journal_path.parent / backup).read_bytes())
    journal_path.unlink()


def freeze_augmented(root):
    """Run DINO after membership freeze, validate and atomically publish the union."""
    root = Path(root).resolve()
    _recover_interrupted_freeze(root)
    lock = verify_stage(root, "membership")
    if lock["stage"] != "membership":
        return verify_stage(root, "augmented")
    if (root / "augmentation/manifests/holdout_incident.json").exists():
        raise ValueError("recorded holdout collision blocks freeze")
    candidates = _candidate_records(root / "augmentation/manifests/candidates.jsonl")
    expected = lock.get("membership_sha256")
    if membership_sha256(candidates) != expected:
        raise ValueError("frozen membership hash mismatch")
    try:
        audit_holdout_collisions(root, candidates)
    except ValueError as error:
        if "holdout collision" in str(error):
            atomic_write_bytes(root / "augmentation/manifests/holdout_incident.json", canonical_json_bytes({"reason":"holdout_collision"}))
        raise
    diagnosed = log_dino_diagnostics(root)
    if membership_sha256(diagnosed) != expected:
        raise ValueError("DINO changed membership")
    validate_dino_changes(candidates, diagnosed)
    if [row["synthetic_id"] for row in diagnosed] != [row["synthetic_id"] for row in candidates]:
        raise ValueError("DINO changed candidate records")
    try:
        audit_holdout_collisions(root, diagnosed)
    except ValueError as error:
        if "holdout collision" in str(error):
            atomic_write_bytes(root / "augmentation/manifests/holdout_incident.json", canonical_json_bytes({"reason":"holdout_collision"}))
        raise
    train = _csv(root / "splits/train.csv")
    qc_context = build_qc_index(root)
    configs = load_config(root)
    groups = _csv(root / "splits/class_groups.csv")
    class_ids = [int(row["class_id"]) for row in groups]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("duplicate class group IDs")
    by_class = {class_id: [] for class_id in class_ids}
    for row in diagnosed:
        class_id = int(row["class_id"])
        if class_id not in by_class:
            raise ValueError("candidate class not present in groups")
        by_class[class_id].append(row)
    accepted, seen_file, seen_pixel = [], set(), set()
    for group in sorted(groups, key=_group_order):
        class_id = int(group["class_id"])
        sources = [row for row in train if int(row["class_id"]) == class_id]
        attempts = sorted(by_class[class_id], key=lambda row: row["attempt_index"])
        if next_candidate(group, sources, attempts) is not None:
            raise ValueError("class has a quota shortage")
        count = 0
        for row in attempts:
            if row.get("accepted") is not True:
                continue
            source = next((r for r in sources if r["image_id"] == row["source_image_id"]), None)
            if source is None:
                raise ValueError("invalid synthetic source")
            qc = check_candidate(root, row, source, configs[row["augmentation_type"]], {"lock":lock, "context":qc_context, "accepted_file":seen_file, "accepted_pixel":seen_pixel})
            if not qc["passed"] or qc["incident"]:
                raise ValueError(f'accepted candidate failed QC: {qc["reason"]}')
            seen_file.add(row["output_sha256"])
            seen_pixel.add(row["output_pixel_sha256"])
            accepted.append(row)
            count += 1
        if count != int(group["required_accepted"]):
            raise ValueError("class quota mismatch")
        group["accepted_actual"] = count
    immutable = canonical_csv_bytes([{key: row[key] for key in GROUP_FIELDS[:-1]} for row in groups], GROUP_FIELDS[:-1])
    if hashlib.sha256(immutable).hexdigest() != lock.get("group_immutable_sha256"):
        raise ValueError("immutable group assignment changed")
    union = list(train)
    for row in accepted:
        union.append({"image_id":row["synthetic_id"], "filepath":row["synthetic_path"], "class_name":row["class_name"],
                      "class_id":row["class_id"], "split":"train", "sha256":row["output_sha256"],
                      "decoded_pixel_sha256":row["output_pixel_sha256"], "is_synthetic":True})
    if len({row["image_id"] for row in union}) != len(union):
        raise ValueError("duplicate image ID in augmented union")
    union.sort(key=lambda row: (int(row["class_id"]), row["filepath"]))
    reasons = Counter(row.get("rejection_reason") or row.get("filter_reason") or "unspecified" for row in diagnosed if row.get("filter_status") == "rejected")
    summary = {"schema_version":1, "attempts":len(diagnosed), "accepted":len(accepted),
               "rejected":sum(row.get("filter_status") == "rejected" for row in diagnosed), "rejection_reasons":dict(sorted(reasons.items())),
               "methods":{},
               "classes":[{"class_id":int(row["class_id"]), "class_name":row["class_name"], "group":row["group"],
                           "original_train_count":int(row["original_train_count"]), "target":int(row["target"]),
                           "required":int(row["required_accepted"]), "accepted":int(row["accepted_actual"]),
                           "coverage":int(row["accepted_actual"]) / int(row["required_accepted"]) if int(row["required_accepted"]) else 1.0}
                          for row in groups]}
    for method in ("traditional", "da_fusion"):
        rows = [row for row in diagnosed if row.get("augmentation_type") == method]
        method_reasons = Counter(row.get("rejection_reason") or row.get("filter_reason") or "unspecified" for row in rows if row.get("filter_status") == "rejected")
        summary["methods"][method] = {"attempts":len(rows), "accepted":sum(row.get("accepted") is True for row in rows),
                                      "rejected":sum(row.get("filter_status") == "rejected" for row in rows),
                                      "rejection_reasons":dict(sorted(method_reasons.items()))}
    group_bytes = canonical_csv_bytes(groups, GROUP_FIELDS)
    metadata = json.loads((root / "splits/split_metadata.json").read_text(encoding="utf-8"))
    metadata["artifacts"]["splits/class_groups.csv"] = hashlib.sha256(group_bytes).hexdigest()
    outputs = {"augmentation/manifests/candidates.jsonl":canonical_jsonl_bytes(diagnosed),
               "splits/class_groups.csv":group_bytes, "splits/split_metadata.json":canonical_json_bytes(metadata),
               "augmentation/manifests/accepted.jsonl":canonical_jsonl_bytes(accepted),
               "augmentation/manifests/qc_summary.json":canonical_json_bytes(summary),
               "splits/train_augmented.csv":canonical_csv_bytes(union, SPLIT_FIELDS)}
    for method, name in ACCEPTED_POOLS.items():
        outputs[name] = canonical_jsonl_bytes(row for row in accepted if row["augmentation_type"] == method)
    verify_stage(root, "membership")
    transaction = root / "augmentation/manifests/.freeze-transaction"
    transaction.mkdir(parents=True, exist_ok=True)
    fixed = {"augmentation/manifests/candidates.jsonl":"candidates.old", "splits/class_groups.csv":"groups.old", "splits/split_metadata.json":"metadata.old"}
    journal = {"membership_sha256":expected, "files":{}}
    for target, backup in fixed.items():
        old = (root / target).read_bytes()
        old_sha = hashlib.sha256(old).hexdigest()
        if old_sha != lock["artifacts"][target]:
            raise ValueError("locked file changed before freeze")
        atomic_write_bytes(transaction / backup, old)
        journal["files"][target] = {"old":old_sha, "new":hashlib.sha256(outputs[target]).hexdigest()}
    atomic_write_bytes(transaction / "journal.json", canonical_json_bytes(journal))
    new_artifacts = dict(lock["artifacts"])
    for name, payload in outputs.items():
        atomic_write_bytes(root / name, payload)
        new_artifacts[name] = sha256_file(root / name)
    for row in accepted:
        new_artifacts[row["synthetic_path"]] = row["output_sha256"]
    new_lock = dict(lock, stage="augmented", artifacts=new_artifacts)
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(new_lock))
    (transaction / "journal.json").unlink()
    return new_lock


from ..config import load_config
from .diffusion import generate_diffusion, load_pinned_diffusion
from .traditional import generate_traditional


def generate_offline(root, *, tail_method="da_fusion"):
    """Generate ordered candidates until automatic QC fills every quota."""
    if tail_method != "da_fusion":
        raise ValueError(f"{tail_method} is not implemented")
    import io
    from datetime import datetime, timezone
    from ..images import write_output_jpeg
    from .policy import plan_candidate
    from ..preflight import _configure_determinism, verify_runtime

    root = Path(root).resolve()
    lock = verify_stage(root, "preflight")
    if lock["stage"] != "preflight":
        verify_stage(root, "membership")
        return "membership_frozen"
    _configure_determinism()
    verify_runtime(root, lock)
    if (root / "augmentation/manifests/holdout_incident.json").exists():
        raise ValueError("recorded holdout collision blocks generation")
    config = load_config(root)
    train = _csv(root / "splits/train.csv")
    qc_context = build_qc_index(root)
    groups = _csv(root / "splits/class_groups.csv")
    candidates_path = root / "augmentation/manifests/candidates.jsonl"
    records = _candidate_records(candidates_path)
    class_ids = {int(group["class_id"]) for group in groups}
    if len(class_ids) != len(groups) or any(int(row["class_id"]) not in class_ids for row in records):
        raise ValueError("invalid candidate class or duplicate class group")
    seen_file, seen_pixel = set(), set()
    pipeline = None
    def persist():
        records.sort(key=lambda row: (int(row["class_id"]), int(row["attempt_index"])))
        atomic_write_bytes(candidates_path, canonical_jsonl_bytes(records))
    def incident(qc):
        if qc.get("incident"):
            atomic_write_bytes(root / "augmentation/manifests/holdout_incident.json", canonical_json_bytes({"reason":"holdout_collision"}))
            raise ValueError("holdout collision detected")
    for group in sorted(groups, key=_group_order):
        class_id = int(group["class_id"])
        sources = [row for row in train if int(row["class_id"]) == class_id]
        attempts = sorted((row for row in records if int(row["class_id"]) == class_id), key=lambda row: int(row["attempt_index"]))
        budget = int(group["required_accepted"])
        if group["group"] == "many":
            next_candidate(group, sources, attempts)
            continue
        method = "traditional" if group["group"] == "medium" else "da_fusion"
        if group["group"] not in ("medium", "tail"):
            raise ValueError("invalid class group")
        accepted = 0
        for index, row in enumerate(attempts):
            expected = plan_candidate(method, class_id, index, sources)
            if any(row.get(key) != value for key,value in expected.items()):
                raise ValueError("candidate history is inconsistent")
            source = next(item for item in sources if item["image_id"] == row["source_image_id"])
            status = row.get("filter_status")
            if status == "accepted" and row.get("accepted") is True:
                qc = check_candidate(root, row, source, config[method], {"lock":lock,"context":qc_context,"accepted_file":seen_file,"accepted_pixel":seen_pixel})
                incident(qc)
                if not qc["passed"]:
                    raise ValueError("previously accepted candidate no longer passes QC")
                seen_file.add(row["output_sha256"])
                seen_pixel.add(row["output_pixel_sha256"])
                accepted += 1
            elif status == "rejected" and row.get("accepted") is False:
                pass
            else:
                raise ValueError("candidate status is inconsistent")
            if accepted > budget:
                raise ValueError("accepted quota exceeded")
        if accepted >= budget:
            next_candidate(group, sources, attempts)
            continue
        if not sources:
            raise ValueError("class has no original train sources")
        while accepted < budget and len(attempts) < 10 * budget:
            attempt = plan_candidate(method, class_id, len(attempts), sources)
            source = next(item for item in sources if item["image_id"] == attempt["source_image_id"])
            source_path = resolve_cleaned_image(root, source["filepath"])
            if sha256_file(source_path) != source["sha256"]:
                raise ValueError("source file differs from locked train manifest")
            original = decode_rgb(source_path)
            attempt["class_name"] = source["class_name"]
            if method == "traditional":
                generated, parameters = generate_traditional(original, attempt, config[method])
                detail = {"augmentation_parameters":parameters, "method_variant":"traditional_fixed"}
            else:
                if pipeline is None:
                    pipeline = load_pinned_diffusion(root, lock["model_snapshot"])
                generated, detail = generate_diffusion(original, attempt, config[method], {**lock["model_snapshot"], "pipeline":pipeline, "device":"cuda"})
                detail.update({"augmentation_parameters":{}, "method_variant":"sd15_img2img_fixed"})
            relative = f'data/augmented_dataset/{method}/{source["class_name"]}/{attempt["synthetic_id"]}.jpg'
            output = _resolve_artifact(root, relative)
            buffer = io.BytesIO()
            write_output_jpeg(buffer, generated)
            content = buffer.getvalue()
            if output.exists():
                if output.read_bytes() != content:
                    raise ValueError("orphan output differs from deterministic regeneration")
            else:
                atomic_write_bytes(output, content)
            decoded = decode_rgb(output)
            row = {**attempt, **detail, "synthetic_path":relative, "source_sha256":source["sha256"],
                   "augmentation_seed":314159, "generation_timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                   "output_width":decoded.width, "output_height":decoded.height, "output_sha256":sha256_file(output),
                   "output_pixel_sha256":pixel_sha256(decoded), "config_sha256":lock["artifacts"][f'augmentation/{method}/config.yaml'],
                   "environment_lock_sha256":lock["artifacts"]["environment.lock.txt"],
                   "dino_source_similarity":None, "dino_nearest_sibling_similarity":None,
                   "dino_nearest_sibling_id":None, "dino_diagnostic_error":None}
            qc = check_candidate(root, row, source, config[method], {"lock":lock,"context":qc_context, "accepted_file":seen_file, "accepted_pixel":seen_pixel})
            if qc.get("incident"):
                row.update({"qc_passed":False, "qc_scores":qc["scores"], "filter_status":"rejected", "accepted":False,
                            "filter_reason":qc["reason"], "rejection_reason":qc["reason"]})
                records.append(row)
                persist()
            incident(qc)
            status = "accepted" if qc["passed"] else "rejected"
            row.update({"qc_passed":qc["passed"], "qc_scores":qc["scores"], "filter_status":status,
                        "accepted":status == "accepted", "filter_reason":qc["reason"], "rejection_reason":qc["reason"]})
            records.append(row)
            attempts.append(row)
            persist()
            if status == "accepted":
                accepted += 1
                seen_file.add(row["output_sha256"])
                seen_pixel.add(row["output_pixel_sha256"])
        if accepted < budget:
            atomic_write_bytes(root / "augmentation/manifests/shortage.json", canonical_json_bytes({"class_id":class_id, "required":budget, "accepted":accepted, "attempts":len(attempts)}))
            raise ValueError("shortage: candidate attempt cap exhausted")
    persist()
    verify_stage(root, "preflight")
    artifacts = dict(lock["artifacts"])
    artifacts["augmentation/manifests/candidates.jsonl"] = sha256_file(candidates_path)
    for row in records:
        if row["accepted"]:
            artifacts[row["synthetic_path"]] = row["output_sha256"]
    new_lock = dict(lock, stage="membership", artifacts=artifacts,
                    augmentation_policy=AUGMENTATION_POLICY, membership_sha256=membership_sha256(records))
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(new_lock))
    return "membership_frozen"
