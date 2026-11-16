"""Canonical serialization and fail-closed verification of published artifacts."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

STAGES = ("splits", "preflight", "membership", "augmented", "initialized", "trained", "tested", "reported")
BENCHMARK_ID = "geoguessr50k_country_augmentation_v1_2"
CLEANED_DATASET = "data/cleaned_dataset"
CLEANING_MANIFEST = "reports/cleaning_manifest.json"
_REQUIRED = {
    "splits": {CLEANING_MANIFEST, "docs/specs.md", "reports/dataset_inventory.csv", "reports/split_audit.json", "splits/class_to_idx.json", "splits/train.csv", "splits/val.csv", "splits/test.csv", "splits/class_groups.csv", "splits/split_metadata.json"},
    "preflight": {"configs/benchmark.yaml", "augmentation/traditional/config.yaml", "augmentation/da_fusion/config.yaml", "environment.lock.txt"},
    "membership": {"augmentation/manifests/candidates.jsonl"},
    "augmented": {"augmentation/manifests/accepted.jsonl", "augmentation/manifests/qc_summary.json", "splits/train_augmented.csv"},
    "initialized": {f"initializations/seed_{seed}.pt" for seed in (42, 123, 2026)},
    "trained": {f"runs/{condition}/seed_{seed}/best_checkpoint.pt" for condition in ("original", "augmented") for seed in (42, 123, 2026)},
    "tested": {f"runs/{condition}/seed_{seed}/predictions_test.csv" for condition in ("original", "augmented") for seed in (42, 123, 2026)},
    "reported": {"reports/per_seed_metrics.csv", "reports/aggregate_metrics.json", "reports/group_comparison.csv", "reports/final_report.md"},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value, exclude_fields=()):
    excluded = set(exclude_fields)
    def project(item):
        if isinstance(item, dict):
            return {k: project(v) for k, v in item.items() if k not in excluded}
        if isinstance(item, (list, tuple)):
            return [project(v) for v in item]
        return item
    return (json.dumps(project(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows):
    return b"".join(canonical_json_bytes(row) for row in rows)


def canonical_csv_bytes(rows, fieldnames):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        rendered = {key: ("" if value is None else str(value).lower() if isinstance(value, bool) else value) for key, value in row.items()}
        writer.writerow(rendered)
    return output.getvalue().encode("utf-8")


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            name = file.name
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def _resolve_artifact(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("artifact path must be POSIX relative")
    path = PurePosixPath(name)
    if path.is_absolute() or not name or any(part in (".", "..") for part in name.split("/")) or ":" in name:
        raise ValueError("artifact path must be POSIX relative")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact escapes root")
    return resolved


def read_benchmark_lock(root):
    try:
        lock = json.loads((root / "benchmark_lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("missing or invalid benchmark lock") from error
    if (not isinstance(lock, dict) or lock.get("schema_version") != 1
            or lock.get("benchmark_id") != BENCHMARK_ID or lock.get("stage") not in STAGES):
        raise ValueError("invalid benchmark lock identity or stage")
    return lock


def resolve_cleaned_image(root, name):
    path = _resolve_artifact(root, name)
    if not name.startswith(CLEANED_DATASET + "/") or not path.is_relative_to(root / CLEANED_DATASET):
        raise ValueError(f"real benchmark image is outside the cleaned dataset: {name}")
    return path


def resolve_manifest_image(root, row):
    if row["is_synthetic"] == "false":
        return resolve_cleaned_image(root, row["filepath"])
    name = row["filepath"]
    path = _resolve_artifact(root, name)
    if (row["is_synthetic"] != "true" or row["split"] != "train"
            or not name.startswith("data/augmented_dataset/") or not path.is_relative_to(root / "data/augmented_dataset")):
        raise ValueError(f"invalid synthetic manifest path or split: {name}")
    return path


def verify_stage(root, stage):
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    root = Path(root).resolve()
    lock = read_benchmark_lock(root)
    current = STAGES.index(lock["stage"])
    if current < STAGES.index(stage):
        raise ValueError("required stage not published")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("lock artifacts missing")
    required = set().union(*(_REQUIRED[name] for name in STAGES[:current + 1]))
    if not required.issubset(artifacts):
        raise ValueError(f"lock missing required artifacts: {sorted(required - set(artifacts))}")
    for name, expected in artifacts.items():
        path = _resolve_artifact(root, name)
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")
    return lock
