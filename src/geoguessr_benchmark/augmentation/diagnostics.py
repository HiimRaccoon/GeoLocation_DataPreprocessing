"""DINO diagnostics: similarity metadata never changes acceptance."""
import copy
import hashlib
import math

import numpy as np

from ..artifacts import canonical_json_bytes


def membership_sha256(rows):
    fields = ("synthetic_id", "source_image_id", "output_sha256", "output_pixel_sha256", "accepted", "filter_status")
    accepted = [{key: row.get(key) for key in fields} for row in rows if row.get("accepted") is True]
    accepted.sort(key=lambda row: row["synthetic_id"])
    return hashlib.sha256(canonical_json_bytes(accepted)).hexdigest()


def diagnose_embeddings(rows, embed):
    """Compute diagnostics for records using an embedding callback; preserve membership."""
    output = copy.deepcopy(list(rows))
    ids = [row["synthetic_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate synthetic IDs")
    original_hash = membership_sha256(output)
    vectors = {}
    errors = {}
    def normalized(vector):
        value = np.asarray(vector, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(value))
        if not value.size or not np.isfinite(value).all() or not math.isfinite(norm) or norm == 0:
            raise ValueError("invalid or zero embedding")
        return value / norm
    for row in output:
        sid = row["synthetic_id"]
        try:
            vectors[sid] = normalized(embed("output", row))
        except Exception as error:
            errors.setdefault(sid, []).append(f"output embedding: {error}")
    for row in output:
        sid = row["synthetic_id"]
        row["dino_source_similarity"] = None
        row["dino_nearest_sibling_similarity"] = None
        row["dino_nearest_sibling_id"] = None
        try:
            source = normalized(embed("source", row))
            if sid in vectors:
                row["dino_source_similarity"] = float(np.dot(source, vectors[sid]))
        except Exception as error:
            errors.setdefault(sid, []).append(f"source embedding: {error}")
        siblings = []
        if sid in vectors:
            for candidate in output:
                other = candidate["synthetic_id"]
                if (other != sid and candidate.get("accepted") is True and candidate.get("filter_status") == "accepted"
                    and candidate["source_image_id"] == row["source_image_id"] and other in vectors):
                    try:
                        siblings.append((float(np.dot(vectors[sid], vectors[other])), other))
                    except ValueError as error:
                        errors.setdefault(sid, []).append(f"sibling embedding: {error}")
        if siblings:
            similarity, other = sorted(siblings, key=lambda item: (-item[0], item[1]))[0]
            row["dino_nearest_sibling_similarity"] = similarity
            row["dino_nearest_sibling_id"] = other
        elif sid in vectors:
            errors.setdefault(sid, []).append("no readable accepted sibling")
        row["dino_diagnostic_error"] = "; ".join(errors.get(sid, [])) or None
    if membership_sha256(output) != original_hash:
        raise ValueError("DINO diagnostics changed accepted membership")
    return output


def dino_transform(image):
    """Letterbox RGB content to 224x224 using zero-normalized mean padding."""
    import torch
    import torch.nn.functional as functional
    from ..images import letterbox_geometry
    if image.mode != "RGB":
        raise ValueError("DINO input must be RGB")
    geometry = letterbox_geometry(image.width, image.height, 224, 224)
    data = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)
    resized = functional.interpolate(tensor, size=(geometry["height"], geometry["width"]), mode="bilinear", align_corners=False, antialias=True)[0]
    mean = torch.tensor([.485, .456, .406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([.229, .224, .225], dtype=torch.float32)[:, None, None]
    output = torch.zeros((3,224,224), dtype=torch.float32)
    left, top, right, bottom = geometry["source_bbox"]
    output[:, top:bottom, left:right] = (resized - mean) / std
    return output


def _load_dino_model(root, snapshot):
    import subprocess
    import torch
    from ..artifacts import _resolve_artifact, sha256_file
    from ..preflight import validate_model_snapshot
    validate_model_snapshot(root, snapshot)
    directory = _resolve_artifact(root, snapshot["dino_code_directory"])
    entries = snapshot["files"].get("dino_code", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("DINO code snapshot is not locked")
    declared = {_resolve_artifact(root, entry["path"]): entry["sha256"] for entry in entries}
    actual = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("DINO code snapshot contains a symlink")
        if path.is_file():
            actual.add(path.resolve())
    if not actual or actual != set(declared) or any(sha256_file(path) != declared[path] for path in actual):
        raise ValueError("DINO code snapshot checksum mismatch")
    try:
        revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify DINO code revision") from error
    if revision != snapshot["dino_revision"]:
        raise ValueError("DINO code revision mismatch")
    weights = _resolve_artifact(root, snapshot["files"]["dino_weights"]["path"])
    if not torch.cuda.is_available():
        raise ValueError("DINO diagnostics require the locked CUDA environment")
    model = torch.hub.load(str(directory), "dino_vits16", source="local", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.to("cuda").eval()


def _embed_candidate(model, root, row, kind, train_index):
    import torch
    from PIL import Image
    from ..artifacts import _resolve_artifact, sha256_file
    from ..images import decode_rgb
    source = train_index.get(row["source_image_id"])
    if not source or source["filepath"] != row["source_image_path"] or source["sha256"] != row["source_sha256"]:
        raise ValueError("DINO source is not the locked original train image")
    output = _resolve_artifact(root, row["synthetic_path"])
    source_path = _resolve_artifact(root, row["source_image_path"])
    if sha256_file(output) != row["output_sha256"] or sha256_file(source_path) != row["source_sha256"]:
        raise ValueError("DINO input hash mismatch")
    generated = decode_rgb(output)
    image = generated if kind == "output" else decode_rgb(source_path).resize(generated.size, Image.Resampling.BILINEAR)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        vector = model(dino_transform(image).unsqueeze(0).to("cuda"))
    return vector.detach().float().cpu().numpy().reshape(-1)


def log_dino_diagnostics(root):
    """Stage diagnostics after membership freeze; never alter locked candidate bytes."""
    import json
    import csv
    from pathlib import Path
    from ..artifacts import atomic_write_bytes, canonical_jsonl_bytes, verify_stage
    from ..preflight import _configure_determinism, verify_runtime
    root = Path(root).resolve()
    lock = verify_stage(root, "membership")
    if lock["stage"] != "membership":
        raise ValueError("DINO diagnostics require membership stage")
    _configure_determinism()
    verify_runtime(root, lock)
    path = root / "augmentation/manifests/candidates.jsonl"
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    before = membership_sha256(records)
    if before != lock.get("membership_sha256"):
        raise ValueError("frozen membership hash mismatch")
    model = _load_dino_model(root, lock["model_snapshot"])
    with (root / "splits/train.csv").open(encoding="utf-8", newline="") as stream:
        train_index = {row["image_id"]:row for row in csv.DictReader(stream)}
    diffusion = [r for r in records if r.get("augmentation_type") == "da_fusion"]
    diagnosed = diagnose_embeddings(diffusion, lambda kind, row: _embed_candidate(model, root, row, kind, train_index))
    by_id = {row["synthetic_id"]: row for row in diagnosed}
    updated = [by_id.get(row["synthetic_id"], row) for row in records]
    if membership_sha256(updated) != before:
        raise ValueError("DINO changed accepted membership")
    atomic_write_bytes(root / "augmentation/manifests/candidates.dino.staged.jsonl", canonical_jsonl_bytes(updated))
    return updated
