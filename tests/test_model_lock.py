import hashlib

import pytest

from geoguessr_benchmark.preflight import validate_model_snapshot


def test_snapshot_requires_immutable_revisions_and_verifies_each_file(tmp_path):
    files = {}
    for name in ("sd_weights", "tokenizer", "vae", "scheduler", "dino_weights", "efficientnet_weights"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = {"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
    snapshot = {"sd_revision": "a" * 40, "dino_revision": "b" * 40, "files": files}
    validate_model_snapshot(tmp_path, snapshot)
    invalid = dict(snapshot, sd_revision="main")
    with pytest.raises(ValueError, match="revision"):
        validate_model_snapshot(tmp_path, invalid)
    (tmp_path / "vae").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_model_snapshot(tmp_path, snapshot)


def test_snapshot_verification_covers_every_sd_file(tmp_path):
    import hashlib
    from geoguessr_benchmark.preflight import validate_sd_directory
    directory = tmp_path / "sd"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    snapshot = {"sd_directory":"sd", "files":{"scheduler":{"path":"sd/config.json", "sha256":hashlib.sha256(b"{}").hexdigest()}}}
    validate_sd_directory(tmp_path, snapshot)
    (directory / "untracked.bin").write_bytes(b"unverified")
    with pytest.raises(ValueError, match="snapshot"):
        validate_sd_directory(tmp_path, snapshot)


def test_resolve_sd_revision_uses_immutable_commit_before_download(tmp_path, monkeypatch):
    import huggingface_hub
    import geoguessr_benchmark.preflight as preflight
    from geoguessr_benchmark.preflight import resolve_sd_snapshot
    calls = {}
    class API:
        def model_info(self, repo_id, revision):
            calls["query"] = (repo_id, revision)
            names = ("unet/model.safetensors", "tokenizer/vocab.json", "vae/model.safetensors", "scheduler/scheduler_config.json", "model_index.json")
            return type("Info", (), {"sha": "a"*40, "siblings":[type("File",(),{"rfilename":name})() for name in names]})()
    def download(repo_id, revision, local_dir, **kwargs):
        calls["download"] = (repo_id, revision)
        base = tmp_path / "sd"
        for name in ("unet/model.safetensors", "tokenizer/vocab.json", "vae/model.safetensors", "scheduler/scheduler_config.json", "model_index.json"):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        return str(base)
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(preflight, "_select_sd_files", lambda files: sorted(files))
    result = resolve_sd_snapshot(tmp_path, "sd")
    assert calls["download"][1] == "a"*40
    assert result["sd_revision"] == "a"*40
    assert all(result["files"][name] for name in ("sd_weights", "tokenizer", "vae", "scheduler"))


def test_runtime_verification_rejects_changed_code_or_gpu(tmp_path,monkeypatch):
    import json
    import geoguessr_benchmark.preflight as preflight
    frozen={"code_sha256":"code-a","gpu":"gpu-a","packages":{"torch":"x"}}
    (tmp_path/"environment.lock.txt").write_text(json.dumps(frozen),encoding="utf-8")
    lock={"model_snapshot":{}}
    monkeypatch.setattr(preflight,"_build_environment",lambda root,snapshot:dict(frozen))
    preflight.verify_runtime(tmp_path,lock)
    monkeypatch.setattr(preflight,"_build_environment",lambda root,snapshot:{**frozen,"code_sha256":"code-b"})
    with pytest.raises(ValueError,match="code_sha256"):
        preflight.verify_runtime(tmp_path,lock)
    monkeypatch.setattr(preflight,"_build_environment",lambda root,snapshot:{**frozen,"gpu":"gpu-b"})
    with pytest.raises(ValueError,match="gpu"):
        preflight.verify_runtime(tmp_path,lock)


def test_interrupted_sd_download_resumes_original_immutable_revision(tmp_path,monkeypatch):
    import huggingface_hub
    import geoguessr_benchmark.preflight as preflight
    queried=[]
    downloads=[]
    class API:
        def model_info(self, repo_id, revision):
            queried.append(revision)
            names=("partial", "unet/w", "tokenizer/t", "vae/v", "scheduler/s")
            return type("Info",(),{"sha":"a"*40 if len(queried)==1 else "b"*40,"siblings":[type("File",(),{"rfilename":name})() for name in names]})()
    def download(repo_id,revision,local_dir,**kwargs):
        downloads.append(revision)
        base=Path(local_dir)
        base.mkdir(parents=True,exist_ok=True)
        (base/"partial").write_text("partial",encoding="utf-8")
        if len(downloads)==1:
            raise OSError("connection lost")
        for name in ("unet/w", "tokenizer/t", "vae/v", "scheduler/s"):
            path=base/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(name,encoding="utf-8")
        return str(base)
    from pathlib import Path
    monkeypatch.setattr(huggingface_hub,"HfApi",API)
    monkeypatch.setattr(huggingface_hub,"snapshot_download",download)
    monkeypatch.setattr(preflight,"_select_sd_files",lambda files:sorted(files))
    with pytest.raises(OSError,match="connection"):
        preflight.resolve_sd_snapshot(tmp_path,"sd")
    result=preflight.resolve_sd_snapshot(tmp_path,"sd")
    assert result["sd_revision"]=="a"*40
    assert downloads==["a"*40,"a"*40] and len(queried)==1


def test_weight_download_resumes_without_overwriting_unowned_files(tmp_path,monkeypatch):
    import io
    import urllib.request
    from geoguessr_benchmark.preflight import _download_snapshot_file
    calls=[]
    def urlopen(url):
        calls.append(url)
        if len(calls)==1:
            class Broken(io.BytesIO):
                def read(self, size=-1):
                    raise OSError("network interruption")
            return Broken()
        return io.BytesIO(b"weights")
    monkeypatch.setattr(urllib.request,"urlopen",urlopen)
    with pytest.raises(OSError): _download_snapshot_file(tmp_path,"https://example.org/weight","models/weight.bin")
    result=_download_snapshot_file(tmp_path,"https://example.org/weight","models/weight.bin")
    assert result["sha256"] == __import__("hashlib").sha256(b"weights").hexdigest()
    assert _download_snapshot_file(tmp_path,"https://example.org/weight","models/weight.bin") == result
    assert len(calls)==2
    (tmp_path/"models/foreign.bin").write_bytes(b"user")
    with pytest.raises(ValueError,match="owned"):
        _download_snapshot_file(tmp_path,"https://example.org/other","models/foreign.bin")


def test_interrupted_dino_clone_reuses_revision_and_preserves_incomplete_dir(tmp_path,monkeypatch):
    import subprocess
    from pathlib import Path
    import geoguessr_benchmark.preflight as preflight
    calls=[]
    def run(args, **kwargs):
        calls.append(args)
        if args[1]=="ls-remote": return type("R",(),{"stdout":"a"*40+"\tHEAD\n"})()
        if args[1]=="clone":
            path=Path(args[-1]); path.mkdir(parents=True)
            if len([x for x in calls if x[1]=="clone"])==1:
                (path/"partial").write_text("keep",encoding="utf-8")
                raise subprocess.CalledProcessError(1,args)
            (path/"ready").write_text("yes",encoding="utf-8")
            return type("R",(),{"stdout":""})()
        if "rev-parse" in args:
            if not (Path(args[2])/"ready").exists(): raise subprocess.CalledProcessError(1,args)
            return type("R",(),{"stdout":"a"*40+"\n"})()
        return type("R",(),{"stdout":""})()
    monkeypatch.setattr(subprocess,"run",run)
    with pytest.raises(ValueError): preflight._resolve_dino_source(tmp_path)
    code,revision=preflight._resolve_dino_source(tmp_path)
    assert revision=="a"*40 and (code/"ready").exists()
    assert len([x for x in calls if x[1]=="ls-remote"])==1
    assert list((tmp_path/"models").glob("dino_code.incomplete.*"))


def test_sd_snapshot_selects_only_fp32_diffusers_files():
    from geoguessr_benchmark.preflight import _select_sd_files
    required=["model_index.json","feature_extractor/preprocessor_config.json","safety_checker/config.json","safety_checker/model.safetensors",
              "scheduler/scheduler_config.json","text_encoder/config.json","text_encoder/model.safetensors","tokenizer/merges.txt",
              "tokenizer/special_tokens_map.json","tokenizer/tokenizer_config.json","tokenizer/vocab.json","unet/config.json",
              "unet/diffusion_pytorch_model.safetensors","vae/config.json","vae/diffusion_pytorch_model.safetensors"]
    selected=_select_sd_files(required+["v1-5-pruned.ckpt","unet/diffusion_pytorch_model.fp16.safetensors","unet/diffusion_pytorch_model.non_ema.safetensors"])
    assert set(selected)==set(required)
    with pytest.raises(ValueError): _select_sd_files(required[:-1])


def test_report_source_check_rejects_code_change_without_requiring_gpu(tmp_path):
    import json
    from geoguessr_benchmark.preflight import source_code_sha256, verify_source
    for name,text in (("src/demo.py","x=1"),("pyproject.toml","[project]"),("docs/specs.md","frozen")):
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="utf-8")
    digest=source_code_sha256(tmp_path)
    (tmp_path/"environment.lock.txt").write_text(json.dumps({"code_sha256":digest}),encoding="utf-8")
    verify_source(tmp_path)
    (tmp_path/"src/demo.py").write_text("x=2",encoding="utf-8")
    with pytest.raises(ValueError,match="code"):
        verify_source(tmp_path)
