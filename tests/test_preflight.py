import pytest
import torch

from geoguessr_benchmark.preflight import memory_ladder


def test_ladder_falls_back_only_for_oom_and_resets_between_attempts():
    attempts = []
    resets = []
    def probe(size):
        attempts.append(size)
        if size > 16:
            raise torch.OutOfMemoryError("out of memory")
    assert memory_ladder(probe, lambda: resets.append(1)) == 16
    assert attempts == [128, 64, 32, 16]
    assert len(resets) == len(attempts)


def test_ladder_stops_on_non_oom_and_exhaustion():
    attempts = []
    def bad(size):
        attempts.append(size)
        raise RuntimeError("deterministic operation unsupported")
    with pytest.raises(RuntimeError, match="deterministic"):
        memory_ladder(bad, lambda: None)
    assert attempts == [128]
    with pytest.raises(torch.OutOfMemoryError):
        memory_ladder(lambda size: (_ for _ in ()).throw(torch.OutOfMemoryError("OOM")), lambda: None)


def test_preflight_publishes_only_after_both_probes(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    import geoguessr_benchmark.preflight as preflight
    from geoguessr_benchmark.artifacts import sha256_file, verify_stage
    from test_qc import setup_fixture
    setup_fixture(tmp_path)
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "configs", tmp_path / "configs")
    for method in ("traditional", "da_fusion"):
        shutil.copytree(project / "augmentation" / method, tmp_path / "augmentation" / method)
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    snapshot = {"files":{"efficientnet_weights":{"path":"model.bin","sha256":sha256_file(model)}}}
    monkeypatch.setattr(preflight,"resolve_all_model_snapshots",lambda root:snapshot)
    monkeypatch.setattr(preflight,"validate_model_snapshot",lambda root,snapshot:None)
    monkeypatch.setattr(preflight,"_require_cuda",lambda:None)
    monkeypatch.setattr(preflight,"_configure_determinism",lambda:None)
    monkeypatch.setattr(preflight,"_build_environment",lambda root,snapshot:{"runtime":"fixture"})
    monkeypatch.setattr(preflight,"_classifier_probe_batch",lambda root,snapshot,config,size:None)
    monkeypatch.setattr(preflight,"_reset_probe_state",lambda:None)
    def broken(root,snapshot,config):
        raise RuntimeError("diffusion probe failed")
    monkeypatch.setattr(preflight,"_diffusion_probe",broken)
    with pytest.raises(RuntimeError,match="diffusion"):
        preflight.run_preflight(tmp_path)
    assert verify_stage(tmp_path,"splits")["stage"]=="splits"
    assert not (tmp_path/"environment.lock.txt").exists()
    monkeypatch.setattr(preflight,"_diffusion_probe",lambda root,snapshot,config:{"actual_timesteps":[99]})
    from geoguessr_benchmark.__main__ import main
    assert main(["--root", str(tmp_path), "preflight"]) == 0
    result=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    assert result["stage"]=="preflight"
    assert verify_stage(tmp_path,"preflight")
    assert result["batch_size_effective"]==128
