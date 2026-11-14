import os
import subprocess
import sys
from pathlib import Path


def test_help_is_available_without_gpu_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", "geoguessr_benchmark", "--help"],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout


def test_internal_training_command_requires_launcher_seed(monkeypatch, tmp_path):
    from geoguessr_benchmark.__main__ import main
    import geoguessr_benchmark.training as training
    called=[]
    monkeypatch.setattr(training,"train_run",lambda *args:called.append(args))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert main(["--root",str(tmp_path),"_train-run","--condition","original","--seed","42"]) == 1
    assert not called
