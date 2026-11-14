import copy

import numpy as np

from geoguessr_benchmark.augmentation.diagnostics import diagnose_embeddings, membership_sha256


def test_dino_diagnostics_never_changes_membership_and_excludes_self():
    rows = [
        {"synthetic_id":"a", "source_image_id":"s", "output_sha256":"1", "output_pixel_sha256":"2", "accepted":True, "filter_status":"accepted"},
        {"synthetic_id":"b", "source_image_id":"s", "output_sha256":"3", "output_pixel_sha256":"4", "accepted":True, "filter_status":"accepted"},
        {"synthetic_id":"c", "source_image_id":"q", "output_sha256":"5", "output_pixel_sha256":"6", "accepted":False, "filter_status":"rejected"},
    ]
    vectors = {"source:s": [1.,0.], "source:q": [0.,1.], "a":[1.,0.], "b":[0.,1.]}
    def embed(kind, row):
        key = "source:"+row["source_image_id"] if kind == "source" else row["synthetic_id"]
        if key not in vectors:
            raise ValueError("bad candidate")
        return np.array(vectors[key])
    before = membership_sha256(rows)
    diagnostics = diagnose_embeddings(rows, embed)
    assert membership_sha256(diagnostics) == before
    assert diagnostics[0]["dino_source_similarity"] == 1.0
    assert diagnostics[0]["dino_nearest_sibling_id"] == "b"
    assert diagnostics[1]["dino_nearest_sibling_id"] == "a"
    assert diagnostics[2]["dino_source_similarity"] is None
    assert diagnostics[2]["dino_diagnostic_error"]
    assert rows[0].get("dino_source_similarity") is None


def test_membership_hash_detects_changed_acceptance_but_ignores_dino():
    rows = [{"synthetic_id":"x", "source_image_id":"s", "output_sha256":"1", "output_pixel_sha256":"2", "accepted":True, "filter_status":"accepted"}]
    modified = copy.deepcopy(rows)
    modified[0]["dino_source_similarity"] = .4
    assert membership_sha256(rows) == membership_sha256(modified)
    modified[0]["accepted"] = False
    assert membership_sha256(rows) != membership_sha256(modified)


def test_diagnostics_accepts_one_pass_iterables_and_rejects_duplicate_ids():
    import pytest
    row = {"synthetic_id":"a", "source_image_id":"s", "output_sha256":"1", "output_pixel_sha256":"2", "accepted":True, "filter_status":"accepted"}
    output = diagnose_embeddings(iter([row]), lambda kind, entry: [1., 0.])
    assert len(output) == 1
    with pytest.raises(ValueError):
        diagnose_embeddings([row, row], lambda kind, entry: [1.,0.])


def test_dino_preprocessing_is_rgb_letterboxed_with_zero_normalized_padding():
    import torch
    from PIL import Image
    from geoguessr_benchmark.augmentation.diagnostics import dino_transform
    image = Image.new("RGB", (371,160), (40,90,140))
    tensor = dino_transform(image)
    assert tensor.shape == (3,224,224)
    assert tensor.dtype == torch.float32
    assert torch.count_nonzero(tensor[:, :20, :]) == 0
    assert torch.equal(tensor, dino_transform(image))


def test_log_dino_stages_metadata_without_unlocking_membership(tmp_path, monkeypatch):
    import json
    from geoguessr_benchmark.artifacts import _REQUIRED, STAGES, atomic_write_bytes, canonical_json_bytes, canonical_jsonl_bytes, sha256_file
    from geoguessr_benchmark.augmentation import diagnostics
    import geoguessr_benchmark.preflight as preflight
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    from test_qc import setup_fixture
    source, candidate = setup_fixture(tmp_path)
    candidate.update({"synthetic_id":"a"*64, "augmentation_type":"da_fusion", "accepted":True, "filter_status":"accepted"})
    path = tmp_path / "augmentation/manifests/candidates.jsonl"
    atomic_write_bytes(path, canonical_jsonl_bytes([candidate]))
    lock = json.loads((tmp_path / "benchmark_lock.json").read_text(encoding="utf-8"))
    for stage in ("preflight", "membership"):
        for name in _REQUIRED[stage]:
            if not (tmp_path / name).exists():
                atomic_write_bytes(tmp_path / name, b"fixture")
            lock["artifacts"][name] = sha256_file(tmp_path / name)
    lock.update({"stage":"membership", "membership_sha256": membership_sha256([candidate]), "model_snapshot":{}})
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    monkeypatch.setattr(diagnostics, "_load_dino_model", lambda root, snapshot: object())
    def embedding(model, root, row, kind, train_index):
        assert row["source_image_id"] in train_index
        if kind == "output":
            raise ValueError("embedding failure")
        return [1.0, 0.0]
    monkeypatch.setattr(diagnostics, "_embed_candidate", embedding)
    result = diagnostics.log_dino_diagnostics(tmp_path)
    assert result[0]["dino_source_similarity"] is None
    assert "embedding failure" in result[0]["dino_diagnostic_error"]
    assert membership_sha256(result) == lock["membership_sha256"]
    assert (tmp_path / "augmentation/manifests/candidates.dino.staged.jsonl").exists()
    assert path.read_bytes() == canonical_jsonl_bytes([candidate])
