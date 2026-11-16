import csv
import json
from pathlib import Path

import pytest
import torch

from geoguessr_benchmark.artifacts import atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, sha256_file


def setup_run(root, monkeypatch):
    import geoguessr_benchmark.training as training
    import geoguessr_benchmark.config as config_module
    import geoguessr_benchmark.classifier as classifier
    import geoguessr_benchmark.preflight as preflight
    monkeypatch.setenv("PYTHONHASHSEED","42")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    root.mkdir()
    atomic_write_bytes(root/"models/weights.pt",b"weights")
    torch.manual_seed(42)
    initial=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3,124))
    (root/"initializations").mkdir()
    torch.save(initial.state_dict(),root/"initializations/seed_42.pt")
    fields=["image_id","filepath","class_name","class_id","split","sha256","decoded_pixel_sha256","is_synthetic"]
    train=[]
    val=[]
    for name, split, label in [("a","train",0),("b","train",1),("c","val",0),("d","val",1)]:
        path=root/f"data/cleaned_dataset/{label}/{name}.jpg"
        atomic_write_bytes(path,name.encode())
        row={"image_id":name,"filepath":path.relative_to(root).as_posix(),"class_name":str(label),"class_id":label,"split":split,"sha256":sha256_file(path),"decoded_pixel_sha256":name,"is_synthetic":False}
        (train if split=="train" else val).append(row)
    for name, rows in (("train",train),("val",val)):
        atomic_write_bytes(root/f"splits/{name}.csv",canonical_csv_bytes(rows,fields))
    atomic_write_bytes(root/"splits/split_metadata.json",canonical_json_bytes({"evaluable_ids":{"val":[0,1],"test":[0,1]},"counts":{"train":2,"val":2,"test":2}}))
    atomic_write_bytes(root/"splits/class_groups.csv",b"class_id,group\n0,tail\n1,tail\n")
    atomic_write_bytes(root/"configs/benchmark.yaml",b"config")
    atomic_write_bytes(root/"environment.lock.txt",canonical_json_bytes({"gpu":"fixture","driver":"fixture","packages":{}}))
    artifacts={name:sha256_file(root/name) for name in ("models/weights.pt","initializations/seed_42.pt","splits/train.csv","splits/val.csv","splits/split_metadata.json","splits/class_groups.csv","configs/benchmark.yaml","environment.lock.txt")}
    lock={"stage":"initialized","artifacts":artifacts,"model_snapshot":{"files":{"efficientnet_weights":{"path":"models/weights.pt","sha256":artifacts["models/weights.pt"]}}}}
    monkeypatch.setattr(training,"verify_stage",lambda root,stage:lock)
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    monkeypatch.setattr(training,"_training_device",lambda:torch.device("cpu"))
    monkeypatch.setattr(config_module,"load_config",lambda root:{"benchmark":{"batch_size_effective":2,"learning_rate":.0003,"weight_decay":.0001,"betas":[.9,.999],"eps":1e-8,"epochs":3,"minimum_learning_rate":1e-6,"num_workers":0,"pin_memory":False,"loss_reduction":"mean","label_smoothing":0.0,"class_weights":None}})
    monkeypatch.setattr(classifier,"build_model",lambda path:torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3,124)))
    monkeypatch.setattr(classifier,"classifier_transform",lambda path:torch.tensor([[[float(Path(path).stem in ('a','c'))]],[[float(Path(path).stem in ('b','d'))]],[[.2]]]))
    return training


def test_resume_epoch_boundary_matches_uninterrupted_training(tmp_path, monkeypatch):
    first=tmp_path/"first"
    training=setup_run(first,monkeypatch)
    training.train_run(first,"original",42)
    expected=torch.load(first/"runs/original/seed_42/last_checkpoint.pt",map_location="cpu",weights_only=True)
    second=tmp_path/"second"
    training=setup_run(second,monkeypatch)
    import geoguessr_benchmark.artifacts as artifacts
    real_write=artifacts.atomic_write_bytes
    def interrupted(path,payload):
        if Path(path).name == "best_checkpoint.pt":
            raise OSError("simulated interruption")
        return real_write(path,payload)
    monkeypatch.setattr(artifacts,"atomic_write_bytes",interrupted)
    with pytest.raises(OSError,match="interruption"):
        training.train_run(second,"original",42)
    monkeypatch.setattr(artifacts,"atomic_write_bytes",real_write)
    training.train_run(second,"original",42)
    actual=torch.load(second/"runs/original/seed_42/last_checkpoint.pt",map_location="cpu",weights_only=True)
    assert actual["epoch"] == 3 and actual["global_step"] == expected["global_step"]
    assert all(torch.equal(actual["model"][key],expected["model"][key]) for key in actual["model"])
    assert [r["sample_ids"] for r in actual["history"]] == [r["sample_ids"] for r in expected["history"]]
    assert len(actual["history"]) == 3
    metadata=json.loads((second/"runs/original/seed_42/run_metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["resume_history"]) == 1
    assert metadata["test_count"] == 2 and metadata["val_count"] == 2
    assert metadata["runtime_identity"]["gpu"] == "fixture"


def test_checkpoint_snapshot_copies_all_tensors_to_cpu():
    from geoguessr_benchmark.training import _cpu_snapshot
    source = {"model":{"weight":torch.tensor([1.0])},"optimizer":{"state":{0:{"exp_avg":torch.tensor([2.0])}}}}
    saved = _cpu_snapshot(source)
    source["model"]["weight"].add_(3)
    source["optimizer"]["state"][0]["exp_avg"].add_(4)
    assert saved["model"]["weight"].device.type == "cpu"
    assert saved["optimizer"]["state"][0]["exp_avg"].device.type == "cpu"
    assert saved["model"]["weight"].item() == 1
    assert saved["optimizer"]["state"][0]["exp_avg"].item() == 2


def test_train_run_refuses_to_start_without_launcher_environment(tmp_path,monkeypatch):
    from geoguessr_benchmark.training import train_run
    monkeypatch.delenv("PYTHONHASHSEED",raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
    with pytest.raises(ValueError,match="launcher"):
        train_run(tmp_path,"original",42)


def test_training_rejects_nonfinite_logits_before_checkpoint(tmp_path,monkeypatch):
    import geoguessr_benchmark.classifier as classifier
    training=setup_run(tmp_path/"nan",monkeypatch)
    monkeypatch.setattr(classifier,"classifier_transform",lambda path:torch.full((3,1,1),float("nan")))
    with pytest.raises(ValueError,match="non-finite"):
        training.train_run(tmp_path/"nan","original",42)
    assert not (tmp_path/"nan/runs/original/seed_42/last_checkpoint.pt").exists()


def test_dataloader_rejects_raw_real_path_and_synthetic_holdouts(tmp_path):
    from geoguessr_benchmark.training import _ManifestDataset

    for row in ({"filepath": "data/compressed_dataset/A/a.jpg", "split": "train", "is_synthetic": "false"},
                {"filepath": "data/augmented_dataset/da_fusion/A/a.jpg", "split": "val", "is_synthetic": "true"}):
        with pytest.raises(ValueError, match="cleaned|synthetic"):
            _ManifestDataset(tmp_path, [row])[0]


def test_training_rejects_legacy_checkpoint_identity(tmp_path, monkeypatch):
    root = tmp_path / "legacy"
    training = setup_run(root, monkeypatch)
    state = root / "runs/original/seed_42/last_checkpoint.pt"
    state.parent.mkdir(parents=True)
    torch.save({"identity": {"config_sha256": "legacy-config"}}, state)
    before = state.read_bytes()
    with pytest.raises(ValueError, match="resume checkpoint identity mismatch"):
        training.train_run(root, "original", 42)
    assert state.read_bytes() == before
