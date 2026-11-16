import hashlib
import json

import pytest

from geoguessr_benchmark.artifacts import BENCHMARK_ID, CLEANING_MANIFEST, atomic_write_bytes, canonical_json_bytes, sha256_file, verify_stage


def test_canonical_projection_ignores_only_excluded_timestamps():
    first = {"time": "one", "row": {"value": 2}, "schema_version": 1}
    second = {"time": "two", "row": {"value": 2}, "schema_version": 1}
    assert canonical_json_bytes(first, exclude_fields={"time"}) == canonical_json_bytes(second, exclude_fields={"time"})
    second["row"]["value"] = 3
    assert canonical_json_bytes(first, exclude_fields={"time"}) != canonical_json_bytes(second, exclude_fields={"time"})
    with pytest.raises(ValueError):
        canonical_json_bytes({"number": float("nan")})


def test_atomic_write_replaces_file_and_hash_checks_bytes(tmp_path):
    path = tmp_path / "a" / "file"
    atomic_write_bytes(path, b"first")
    atomic_write_bytes(path, b"second")
    assert path.read_bytes() == b"second"
    assert sha256_file(path) == hashlib.sha256(b"second").hexdigest()


def test_interrupted_atomic_write_keeps_published_bytes(tmp_path, monkeypatch):
    from geoguessr_benchmark import artifacts

    path = tmp_path / "manifest.json"
    path.write_bytes(b"published")

    def interrupt(*_):
        raise OSError("interrupted before replacement")

    monkeypatch.setattr(artifacts.os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_bytes(path, b"replacement")
    assert path.read_bytes() == b"published"
    assert list(tmp_path.iterdir()) == [path]


def test_verify_rejects_incomplete_stage_and_tampered_file(tmp_path):
    paths = [CLEANING_MANIFEST, "docs/specs.md", "reports/dataset_inventory.csv", "reports/split_audit.json", "splits/class_to_idx.json", "splits/train.csv",
             "splits/val.csv", "splits/test.csv", "splits/class_groups.csv", "splits/split_metadata.json"]
    artifacts = {}
    for name in paths:
        path = tmp_path / name
        atomic_write_bytes(path, name.encode())
        artifacts[name] = sha256_file(path)
    lock = {"schema_version": 1, "benchmark_id": "geoguessr50k_country_augmentation_v1_1", "stage": "splits", "artifacts": artifacts}
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    with pytest.raises(ValueError, match="identity"):
        verify_stage(tmp_path, "splits")
    lock["benchmark_id"] = BENCHMARK_ID
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    assert verify_stage(tmp_path, "splits")
    with pytest.raises(ValueError):
        verify_stage(tmp_path, "preflight")
    (tmp_path / paths[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_stage(tmp_path, "splits")
    lock["artifacts"].pop(paths[0])
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    with pytest.raises(ValueError):
        verify_stage(tmp_path, "splits")
