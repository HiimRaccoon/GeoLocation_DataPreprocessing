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


def test_refresh_split_lock_checks_original_hashes_and_preserves_splits(tmp_path):
    from geoguessr_benchmark.__main__ import main
    from test_qc import setup_fixture

    setup_fixture(tmp_path)
    args = ["--root", str(tmp_path), "refresh-split-lock", "--previous-spec", "old-spec.md"]
    spec = tmp_path / "docs/specs.md"
    (tmp_path / "old-spec.md").write_bytes(spec.read_bytes())
    spec.write_bytes(b"automatic QC spec")
    lock_path = tmp_path / "benchmark_lock.json"
    old_lock = json.loads(lock_path.read_bytes())
    old_lock.update(stage="membership", model_snapshot={"old": True})
    atomic_write_bytes(lock_path, canonical_json_bytes(old_lock))
    with pytest.raises(ValueError, match="stale augmentation lock"):
        verify_stage(tmp_path, "membership")
    previous = lock_path.read_bytes()
    val = tmp_path / "splits/val.csv"
    original = val.read_bytes()
    val.write_bytes(b"changed")
    assert main(args) == 1 and lock_path.read_bytes() == previous
    val.write_bytes(original)
    orphan = tmp_path / "data/augmented_dataset/old.jpg"
    atomic_write_bytes(orphan, b"old")
    assert main(args) == 1 and orphan.read_bytes() == b"old"
    orphan.rename(tmp_path / "archived.jpg")
    orphan.parent.rmdir()
    assert main(args) == 0
    result = verify_stage(tmp_path, "splits")
    assert result["stage"] == "splits"
    assert "model_snapshot" not in result
    for name, digest in old_lock["artifacts"].items():
        if name != "docs/specs.md":
            assert result["artifacts"][name] == digest
