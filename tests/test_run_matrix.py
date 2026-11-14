import json

import pytest
import torch

from geoguessr_benchmark.artifacts import _REQUIRED, STAGES, BENCHMARK_ID, atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, sha256_file, verify_stage
from geoguessr_benchmark.experiment import freeze_checkpoints


def test_freeze_requires_six_complete_runs_and_locked_best(tmp_path):
    paths = set().union(*(_REQUIRED[s] for s in STAGES[:STAGES.index("initialized")+1]))
    for path in paths:
        atomic_write_bytes(tmp_path/path,b"fixture")
    atomic_write_bytes(tmp_path/"configs/benchmark.yaml",b"batch_size_effective: 2\nepochs: 20\n")
    for name in ("train.csv", "train_augmented.csv"):
        atomic_write_bytes(tmp_path/f"splits/{name}",b"image_id,is_synthetic\na,false\nb,false\n")
    atomic_write_bytes(tmp_path/"splits/val.csv",b"image_id,class_id\nv1,0\nv2,0\nv3,0\n")
    atomic_write_bytes(tmp_path/"splits/split_metadata.json",canonical_json_bytes({"evaluable_ids":{"val":[0],"test":[0]}}))
    group_csv="class_id,group\n0,tail\n"+"".join(f"{i},many\n" for i in range(1,124))
    atomic_write_bytes(tmp_path/"splits/class_groups.csv",group_csv.encode())
    for seed in (42,123,2026):
        torch.save({"weight":torch.tensor([1.0])},tmp_path/f"initializations/seed_{seed}.pt")
    artifacts={name:sha256_file(tmp_path/name) for name in paths}
    lock={"schema_version":1,"benchmark_id":BENCHMARK_ID,"stage":"initialized","artifacts":artifacts}
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    for condition in ("original","augmented"):
        for seed in (42,123,2026):
            base=tmp_path/f"runs/{condition}/seed_{seed}"
            base.mkdir(parents=True)
            train = "splits/train.csv" if condition=="original" else "splits/train_augmented.csv"
            identity={"condition":condition,"seed":seed,"train_sha256":artifacts[train],"val_sha256":artifacts["splits/val.csv"],
                      "init_sha256":artifacts[f"initializations/seed_{seed}.pt"],"config_sha256":artifacts["configs/benchmark.yaml"],
                      "environment_sha256":artifacts["environment.lock.txt"],"group_sha256":artifacts["splits/class_groups.csv"],
                      "split_metadata_sha256":artifacts["splits/split_metadata.json"]}
            torch.save({"identity":identity,"epoch":1,"val_macro_f1":.5,"model":{"weight":torch.tensor([1.0])}},base/"best_checkpoint.pt")
            torch.save({"identity":identity,"epoch":20,"global_step":20,"model":{"weight":torch.tensor([1.0])}},base/"last_checkpoint.pt")
            atomic_write_bytes(base/"history.csv",("epoch,val_macro_f1\n"+"".join(f"{i},0.5\n" for i in range(1,21))).encode())
            atomic_write_bytes(base/"lr_steps.csv",("global_step\n"+"".join(f"{i}\n" for i in range(20))).encode())
            atomic_write_bytes(base/"config.yaml",b"batch_size_effective: 2\nepochs: 20\n")
            atomic_write_bytes(base/"metrics.json",canonical_json_bytes({"validation_at_best":{"macro_f1":.5,"macro_precision":1.0,"macro_recall":1/3,"weighted_f1":.5,"top1_accuracy":1/3,"top5_accuracy":1.0}}))
            atomic_write_bytes(base/"group_metrics.json",canonical_json_bytes({"validation_at_best":{
                "tail":{"macro_f1":.5,"macro_recall":1/3,"evaluable_classes":1,"support":3},
                "many":{"macro_f1":None,"macro_recall":None,"evaluable_classes":0,"support":0},
                "medium":{"macro_f1":None,"macro_recall":None,"evaluable_classes":0,"support":0}}}))
            matrix = "true_class_id," + ",".join(str(i) for i in range(124)) + "\n" + "".join(str(i)+","+",".join(str(1 if i==0 and j==0 else 2 if i==0 and j==1 else 0) for j in range(124))+"\n" for i in range(124))
            atomic_write_bytes(base/"confusion_matrix_val.csv",matrix.encode())
            per_fields=["class_id","support","tp","fp","fn","precision","recall","f1","evaluable","status"]
            per_rows=[{"class_id":i,"support":3 if i==0 else 0,"tp":1 if i==0 else 0,"fp":2 if i==1 else 0,"fn":2 if i==0 else 0,
                       "precision":1.0 if i==0 else None,"recall":1/3 if i==0 else None,"f1":.5 if i==0 else None,
                       "evaluable":i==0,"status":"evaluable" if i==0 else "not evaluable on this split"} for i in range(124)]
            atomic_write_bytes(base/"per_class_metrics_val.csv",canonical_csv_bytes(per_rows,per_fields))
            info={"condition":condition,"seed":seed,"identity":identity,"status":"completed","epochs_completed":20,"optimizer_steps":20,
                  "selected_epoch":1,"best_checkpoint_sha256":sha256_file(base/"best_checkpoint.pt"),"batch_size":2,"train_count":2,
                  "original_count":2,"synthetic_count":0,"duration_seconds":1,"failure_history":[]}
            atomic_write_bytes(base/"run_metadata.json",canonical_json_bytes(info))
    missing=tmp_path/"runs/augmented/seed_2026/history.csv"
    data=missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValueError):
        freeze_checkpoints(tmp_path)
    assert json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))["stage"]=="initialized"
    atomic_write_bytes(missing,data)
    matrix_path=tmp_path/"runs/original/seed_42/confusion_matrix_val.csv"
    correct_matrix=matrix_path.read_bytes()
    matrix_path.write_bytes(correct_matrix.replace(b"0,1,2,0",b"0,0,3,0",1))
    with pytest.raises(ValueError):
        freeze_checkpoints(tmp_path)
    atomic_write_bytes(matrix_path,correct_matrix)
    checkpoint=tmp_path/"runs/original/seed_42/best_checkpoint.pt"
    checkpoint_bytes=checkpoint.read_bytes()
    metadata_path=checkpoint.parent/"run_metadata.json"
    metadata_bytes=metadata_path.read_bytes()
    bad=torch.load(checkpoint,weights_only=True)
    bad["model"]={}
    torch.save(bad,checkpoint)
    metadata=json.loads(metadata_bytes)
    metadata["best_checkpoint_sha256"]=sha256_file(checkpoint)
    atomic_write_bytes(metadata_path,canonical_json_bytes(metadata))
    with pytest.raises(ValueError):
        freeze_checkpoints(tmp_path)
    atomic_write_bytes(checkpoint,checkpoint_bytes)
    atomic_write_bytes(metadata_path,metadata_bytes)
    from geoguessr_benchmark.__main__ import main
    assert main(["--root",str(tmp_path),"freeze-checkpoints"]) == 0
    result=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    assert result["stage"]=="trained"
    assert verify_stage(tmp_path,"trained")
    (tmp_path/"runs/original/seed_42/best_checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify_stage(tmp_path,"trained")


