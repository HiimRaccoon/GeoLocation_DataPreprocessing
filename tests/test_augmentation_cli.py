import pytest

from geoguessr_benchmark.__main__ import main


def test_geo_disc_ds_is_recognized_but_not_runnable(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "augment", "--tail-method", "geo_disc_ds"]) == 1
    assert "geo_disc_ds is not implemented" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("command", [
    ["preflight"], ["augment"], ["augment-freeze"],
    ["review-export", "--output", "augmentation/manifests/review_queue.jsonl"],
    ["review-import", "--input", "reviews.jsonl"],
])
def test_augmentation_commands_fail_closed_without_split_lock(tmp_path, command):
    assert main(["--root", str(tmp_path), *command]) == 1
    assert not (tmp_path / "benchmark_lock.json").exists()


def test_augmentation_cli_records_shortage_and_does_not_freeze(tmp_path,monkeypatch):
    import json
    from PIL import Image
    import geoguessr_benchmark.preflight as preflight
    import geoguessr_benchmark.augmentation.workflow as workflow
    from geoguessr_benchmark.artifacts import _REQUIRED, atomic_write_bytes, canonical_json_bytes, sha256_file
    from test_qc import setup_fixture
    setup_fixture(tmp_path)
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for name in _REQUIRED["preflight"]:
        if not (tmp_path/name).exists(): atomic_write_bytes(tmp_path/name,b"fixture")
        lock["artifacts"][name]=sha256_file(tmp_path/name)
    lock.update({"stage":"preflight","model_snapshot":{}})
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    monkeypatch.setattr(workflow,"load_config",lambda root:{"benchmark":{},"traditional":{},"da_fusion":{}})
    monkeypatch.setattr(workflow,"load_pinned_diffusion",lambda root,snapshot:object())
    monkeypatch.setattr(workflow,"generate_diffusion",lambda *args:(Image.new("RGB",(4,4),"black"),{"safety_flag":False}))
    monkeypatch.setattr(workflow,"check_candidate",lambda *args:{"passed":False,"reason":"blank","scores":{},"incident":None})
    assert main(["--root",str(tmp_path),"augment"]) == 1
    shortage=json.loads((tmp_path/"augmentation/manifests/shortage.json").read_text(encoding="utf-8"))
    assert shortage["required"]==8 and shortage["accepted"]==0 and shortage["attempts"]==80
    assert json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))["stage"]=="preflight"
    assert len((tmp_path/"augmentation/manifests/candidates.jsonl").read_text(encoding="utf-8").splitlines())==80
