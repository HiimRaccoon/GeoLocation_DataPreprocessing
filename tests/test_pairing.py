from pathlib import Path
import json

import pytest
import torch

from geoguessr_benchmark.artifacts import AUGMENTATION_POLICY, _REQUIRED, STAGES, BENCHMARK_ID, atomic_write_bytes, canonical_json_bytes, sha256_file
from geoguessr_benchmark.experiment import initialize_pairs


def test_initializations_are_seeded_paired_and_locked(tmp_path, monkeypatch):
    import geoguessr_benchmark.classifier as classifier
    import geoguessr_benchmark.config as config
    import geoguessr_benchmark.preflight as preflight
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    for stage in STAGES[:STAGES.index("augmented") + 1]:
        for name in _REQUIRED[stage]:
            atomic_write_bytes(tmp_path / name, b"fixture")
    weights = tmp_path / "models/efficientnet.pt"
    atomic_write_bytes(weights, b"weights")
    artifacts = {name: sha256_file(tmp_path / name) for stage in STAGES[:STAGES.index("augmented") + 1] for name in _REQUIRED[stage]}
    artifacts["models/efficientnet.pt"] = sha256_file(weights)
    lock = {"schema_version":1,"benchmark_id":BENCHMARK_ID,"stage":"augmented","augmentation_policy":AUGMENTATION_POLICY, "artifacts":artifacts,
            "model_snapshot":{"files":{"efficientnet_weights":{"path":"models/efficientnet.pt","sha256":sha256_file(weights)}}}}
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    monkeypatch.setattr(config, "load_config", lambda root: {"benchmark":{"batch_size_effective": 16}})
    monkeypatch.setattr(classifier, "build_model", lambda path: torch.nn.Linear(4, 3))
    result = initialize_pairs(tmp_path)
    assert result["stage"] == "initialized"
    states = [torch.load(tmp_path / f"initializations/seed_{seed}.pt", weights_only=True) for seed in (42,123,2026)]
    assert not torch.equal(states[0]["weight"], states[1]["weight"])
    for seed in (42,123,2026):
        path = f"initializations/seed_{seed}.pt"
        assert result["artifacts"][path] == sha256_file(tmp_path / path)
    assert initialize_pairs(tmp_path)["stage"] == "initialized"
    from geoguessr_benchmark.__main__ import main
    assert main(["--root", str(tmp_path), "init"]) == 0
