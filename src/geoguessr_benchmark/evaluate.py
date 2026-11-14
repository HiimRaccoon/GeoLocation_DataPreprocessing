"""Validation helpers for locked final-test predictions."""
import json
import math


def validate_predictions(manifest, predictions):
    truth = {}
    for row in manifest:
        image_id = row["image_id"]
        class_id = int(row["class_id"])
        if image_id in truth or not 0 <= class_id < 124:
            raise ValueError("invalid or duplicate test manifest entry")
        truth[image_id] = class_id
    if not truth:
        raise ValueError("empty test manifest")
    seen = {}
    for row in predictions:
        image_id = row["image_id"]
        if image_id not in truth or image_id in seen:
            raise ValueError("unknown or duplicate prediction ID")
        try:
            true = int(row["true_class_id"])
            predicted = int(row["predicted_class_id"])
            raw_ids, raw_scores = row["top5_class_ids"], row["top5_scores"]
            ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            scores = json.loads(raw_scores) if isinstance(raw_scores, str) else raw_scores
            if not isinstance(ids, list) or not isinstance(scores, list) or len(ids) != 5 or len(scores) != 5:
                raise ValueError("top5 must contain five IDs and scores")
            if any(type(x) is not int or not 0 <= x < 124 for x in ids) or len(set(ids)) != 5:
                raise ValueError("invalid top5 class IDs")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in scores):
                raise ValueError("invalid top5 scores")
            if true != truth[image_id] or predicted != ids[0] or list(zip(ids, scores)) != sorted(zip(ids, scores), key=lambda item: (-item[1], item[0])):
                raise ValueError("prediction label or ranking mismatch")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prediction: {image_id}") from error
        seen[image_id] = {"image_id": image_id, "true_id": true, "pred_id": predicted, "top5_ids": ids, "top5_scores": scores}
    if set(seen) != set(truth):
        raise ValueError("prediction IDs do not match test manifest")
    return [seen[row["image_id"]] for row in manifest]


def load_test_predictions(root, condition, seed):
    """Load a test prediction file only after the six-run tested lock is verified."""
    import csv
    from pathlib import Path
    from .artifacts import verify_stage

    if condition not in ("original", "augmented") or seed not in (42, 123, 2026):
        raise ValueError("invalid run identity")
    root = Path(root).resolve()
    lock = verify_stage(root, "tested")
    key = f"{condition}/seed_{seed}"
    expected = {"checkpoint_sha256": lock["artifacts"][f"runs/{key}/best_checkpoint.pt"],
                "test_manifest_sha256": lock["artifacts"]["splits/test.csv"],
                "config_sha256": lock["artifacts"]["configs/benchmark.yaml"]}
    if lock.get("test_prediction_inputs", {}).get(key) != expected:
        raise ValueError("prediction input identity does not match frozen run")
    with (root / "splits/test.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    if any(row.get("split") != "test" or row.get("is_synthetic") != "false" for row in manifest):
        raise ValueError("final evaluation requires real test images")
    with (root / f"runs/{key}/predictions_test.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    return validate_predictions(manifest, predictions)


def _run_test_inference(root, condition, seed, lock, manifest):
    import torch
    from .artifacts import _resolve_artifact, sha256_file
    from .classifier import build_model
    from .config import load_config
    from .preflight import _configure_determinism, verify_runtime
    from .training import _ManifestDataset, _init_worker, _training_device
    _configure_determinism()
    device = _training_device()
    verify_runtime(root, lock)
    config = load_config(root)["benchmark"]
    weight_entry = lock["model_snapshot"]["files"]["efficientnet_weights"]
    model = build_model(_resolve_artifact(root, weight_entry["path"]))
    checkpoint = torch.load(root / f"runs/{condition}/seed_{seed}/best_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    for row in manifest:
        path = _resolve_artifact(root, row["filepath"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError("test image does not match its locked manifest")
    loader = torch.utils.data.DataLoader(_ManifestDataset(root,manifest),batch_size=config["batch_size_effective"],shuffle=False,
        generator=torch.Generator().manual_seed(seed),num_workers=config["num_workers"],pin_memory=config["pin_memory"],
        persistent_workers=False,drop_last=False,worker_init_fn=_init_worker)
    predictions = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for images, labels, ids in loader:
            logits = model(images.to(device=device,dtype=torch.float32))
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite test logits")
            ranked = torch.argsort(logits,dim=1,descending=True,stable=True)[:,:5]
            scores = torch.gather(logits,1,ranked).cpu().tolist()
            for image_id, true, classes, values in zip(ids,labels.tolist(),ranked.cpu().tolist(),scores):
                predictions.append({"image_id":image_id,"true_class_id":true,"predicted_class_id":classes[0],
                                    "top5_class_ids":classes,"top5_scores":values})
    return predictions


def _recover_test_transaction(root):
    import json
    from .artifacts import atomic_write_bytes, sha256_file, verify_stage
    transaction = root / "reports/.test-transaction"
    journal_path = transaction / "journal.json"
    if not journal_path.exists():
        return
    lock = json.loads((root / "benchmark_lock.json").read_text(encoding="utf-8"))
    if lock.get("stage") in ("tested", "reported"):
        verify_stage(root,"tested")
        journal_path.unlink()
        return
    if lock.get("stage") != "trained":
        raise ValueError("invalid test evaluation transaction stage")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    fixed = {f"runs/{condition}/seed_{seed}/{name}":f"{condition}_{seed}_{name}.old"
             for condition in ("original","augmented") for seed in (42,123,2026)
             for name in ("metrics.json","group_metrics.json","run_metadata.json")}
    if set(journal.get("files",{})) != set(fixed):
        raise ValueError("invalid test transaction file list")
    for name, backup in fixed.items():
        info = journal["files"][name]
        path, old = root/name, transaction/backup
        if (not path.is_file() or not old.is_file() or sha256_file(old) != info.get("old") or
            lock["artifacts"].get(name) != info.get("old") or sha256_file(path) not in (info.get("old"),info.get("new"))):
            raise ValueError("interrupted test publication cannot be recovered safely")
    for name, backup in fixed.items():
        atomic_write_bytes(root/name, (transaction/backup).read_bytes())
    journal_path.unlink()


def evaluate_test_once(root):
    """Infer six frozen checkpoints once and publish test metrics transactionally."""
    import csv
    import hashlib
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from .artifacts import atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, canonical_jsonl_bytes, sha256_file, verify_stage
    from .metrics import evaluate_predictions
    root = Path(root).resolve()
    _recover_test_transaction(root)
    lock = verify_stage(root,"trained")
    if lock["stage"] != "trained":
        return verify_stage(root,"tested")
    from .preflight import _configure_determinism, verify_runtime
    _configure_determinism()
    verify_runtime(root, lock)
    with (root/"splits/test.csv").open(encoding="utf-8",newline="") as stream:
        manifest=list(csv.DictReader(stream))
    if not manifest or any(row.get("split")!="test" or row.get("is_synthetic")!="false" for row in manifest):
        raise ValueError("test manifest must contain real images")
    meta=json.loads((root/"splits/split_metadata.json").read_text(encoding="utf-8"))
    with (root/"splits/class_groups.csv").open(encoding="utf-8",newline="") as stream:
        groups={name:[] for name in ("many","medium","tail")}
        for row in csv.DictReader(stream): groups[row["group"]].append(int(row["class_id"]))
    fields=["image_id","true_class_id","predicted_class_id","top5_class_ids","top5_scores"]
    outputs={}
    input_identities={}
    failure_paths=[]
    for seed in (42,123,2026):
        for condition in ("original","augmented"):
            verify_stage(root,"trained")
            key=f"{condition}/seed_{seed}"
            base=root/"runs"/condition/f"seed_{seed}"
            path=base/"predictions_test.csv"
            sidecar=base/"predictions_test.lock.json"
            staged=base/"predictions_test.csv.staged"
            pending=base/"predictions_test.pending.json"
            failures_path=base/"test_failures.jsonl"
            identity={"checkpoint_sha256":lock["artifacts"][f"runs/{key}/best_checkpoint.pt"],
                      "test_manifest_sha256":lock["artifacts"]["splits/test.csv"],
                      "config_sha256":lock["artifacts"]["configs/benchmark.yaml"]}
            input_identities[key]=identity
            try:
                if sidecar.exists():
                    saved=json.loads(sidecar.read_text(encoding="utf-8"))
                    if saved.get("identity") != identity:
                        raise ValueError("locked test predictions do not match run identity")
                    if not path.is_file() and staged.is_file() and sha256_file(staged) == saved.get("prediction_sha256"):
                        atomic_write_bytes(path,staged.read_bytes())
                    if not path.is_file() or saved.get("prediction_sha256") != sha256_file(path):
                        raise ValueError("locked test predictions do not match run identity")
                    with path.open(encoding="utf-8",newline="") as stream:
                        raw=list(csv.DictReader(stream))
                    normalized=validate_predictions(manifest,raw)
                else:
                    had_pending = pending.exists()
                    if had_pending:
                        if json.loads(pending.read_text(encoding="utf-8")).get("identity") != identity:
                            raise ValueError("pending test inference has a different identity")
                    else:
                        if path.exists() or staged.exists():
                            raise ValueError("unowned test prediction files exist")
                        atomic_write_bytes(pending,canonical_json_bytes({"identity":identity}))
                    if staged.is_file():
                        with staged.open(encoding="utf-8",newline="") as stream:
                            normalized=validate_predictions(manifest,list(csv.DictReader(stream)))
                    else:
                        if had_pending:
                            previous=[]
                            if failures_path.exists():
                                previous=[json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                            previous.append({"timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                                             "error":"interrupted before predictions were staged; retrying the same frozen checkpoint"})
                            atomic_write_bytes(failures_path,canonical_jsonl_bytes(previous))
                        predictions=_run_test_inference(root,condition,seed,lock,manifest)
                        normalized=validate_predictions(manifest,predictions)
                        rows=[{"image_id":r["image_id"],"true_class_id":r["true_id"],"predicted_class_id":r["pred_id"],
                               "top5_class_ids":json.dumps(r["top5_ids"],separators=(",",":")),
                               "top5_scores":json.dumps(r["top5_scores"],separators=(",",":"),allow_nan=False)} for r in normalized]
                        atomic_write_bytes(staged,canonical_csv_bytes(rows,fields))
                    atomic_write_bytes(sidecar,canonical_json_bytes({"identity":identity,"prediction_sha256":sha256_file(staged)}))
                    atomic_write_bytes(path,staged.read_bytes())
                result=evaluate_predictions(normalized,meta["evaluable_ids"]["test"],groups)
            except Exception as error:
                previous=[]
                if failures_path.exists():
                    previous=[json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                previous.append({"timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"error":str(error)})
                atomic_write_bytes(failures_path,canonical_jsonl_bytes(previous))
                raise
            prefix=f"runs/{key}"
            metrics=json.loads((base/"metrics.json").read_text(encoding="utf-8"))
            group_metrics=json.loads((base/"group_metrics.json").read_text(encoding="utf-8"))
            run_meta=json.loads((base/"run_metadata.json").read_text(encoding="utf-8"))
            metrics["final_test"]=result["metrics"]
            group_metrics["final_test"]=result["groups"]
            run_meta["test_prediction_inputs"]=identity
            run_meta["test_failure_history"]=[json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()] if failures_path.exists() else []
            outputs[f"{prefix}/metrics.json"]=canonical_json_bytes(metrics)
            outputs[f"{prefix}/group_metrics.json"]=canonical_json_bytes(group_metrics)
            outputs[f"{prefix}/run_metadata.json"]=canonical_json_bytes(run_meta)
            per_fields=["class_id","support","tp","fp","fn","precision","recall","f1","evaluable","status"]
            outputs[f"{prefix}/per_class_metrics_test.csv"]=canonical_csv_bytes(result["per_class"],per_fields)
            matrix_fields=["true_class_id"]+[str(i) for i in range(124)]
            matrix=[{"true_class_id":i,**{str(j):v for j,v in enumerate(row)}} for i,row in enumerate(result["confusion"])]
            outputs[f"{prefix}/confusion_matrix_test.csv"]=canonical_csv_bytes(matrix,matrix_fields)
            if failures_path.exists(): failure_paths.append(failures_path.relative_to(root).as_posix())
    transaction=root/"reports/.test-transaction"
    transaction.mkdir(parents=True,exist_ok=True)
    fixed={f"runs/{condition}/seed_{seed}/{name}":f"{condition}_{seed}_{name}.old"
           for condition in ("original","augmented") for seed in (42,123,2026)
           for name in ("metrics.json","group_metrics.json","run_metadata.json")}
    journal={"files":{}}
    for name, backup in fixed.items():
        old=(root/name).read_bytes()
        old_sha=hashlib.sha256(old).hexdigest()
        if lock["artifacts"].get(name)!=old_sha:
            raise ValueError("locked run metadata changed during test")
        atomic_write_bytes(transaction/backup,old)
        journal["files"][name]={"old":old_sha,"new":hashlib.sha256(outputs[name]).hexdigest()}
    atomic_write_bytes(transaction/"journal.json",canonical_json_bytes(journal))
    artifacts=dict(lock["artifacts"])
    for name,payload in outputs.items():
        atomic_write_bytes(root/name,payload)
        artifacts[name]=sha256_file(root/name)
    for condition in ("original","augmented"):
        for seed in (42,123,2026):
            for name in ("predictions_test.csv","predictions_test.lock.json"):
                relative=f"runs/{condition}/seed_{seed}/{name}"
                artifacts[relative]=sha256_file(root/relative)
    for name in failure_paths: artifacts[name]=sha256_file(root/name)
    result=dict(lock,stage="tested",artifacts=artifacts,test_prediction_inputs=input_identities)
    atomic_write_bytes(root/"benchmark_lock.json",canonical_json_bytes(result))
    (transaction/"journal.json").unlink()
    return result
