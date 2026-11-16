"""Tiny end-to-end benchmark smoke; all data and models are fixtures."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from geoguessr_benchmark.__main__ import main
from geoguessr_benchmark.artifacts import atomic_write_bytes, canonical_jsonl_bytes, sha256_file, verify_stage
from geoguessr_benchmark.augmentation.diffusion import native_generation_geometry
from geoguessr_benchmark.cleaning import HUD_BOXES


def test_complete_fixture_workflow_with_review_resume_and_test_gate(tmp_path, monkeypatch):
    import subprocess
    import geoguessr_benchmark.inventory as inventory
    import geoguessr_benchmark.split as split
    import geoguessr_benchmark.preflight as preflight
    import geoguessr_benchmark.augmentation.workflow as workflow
    import geoguessr_benchmark.classifier as classifier
    import geoguessr_benchmark.training as training

    root = tmp_path
    project = Path(__file__).resolve().parents[1]
    (root/"docs").mkdir()
    (root/"docs/specs.md").write_text("fixture spec",encoding="utf-8")
    shutil.copytree(project/"src",root/"src")
    shutil.copyfile(project/"pyproject.toml",root/"pyproject.toml")
    shutil.copytree(project/"configs",root/"configs")
    for method in ("traditional","da_fusion"):
        shutil.copytree(project/"augmentation"/method,root/"augmentation"/method)
    rng=np.random.default_rng(2026)
    for class_id in range(124):
        directory=root/"data/compressed_dataset"/f"C{class_id:03d}"
        directory.mkdir(parents=True)
        for index in range(4 if class_id==0 else 1):
            pixels=rng.integers(0,256,(40,60,3),dtype=np.uint8)
            Image.fromarray(pixels).save(directory/f"{index}.jpg",quality=95)
    monkeypatch.setattr(inventory,"DEFAULT_BASELINE",{"images":127,"classes":124})
    # Fixture-only budgets keep this smoke test small; production group boundaries
    # are tested independently and the standard policy remains unchanged.
    monkeypatch.setattr(split,"group_for_count",lambda n:("tail",4,2) if n==2 else ("tail",1,0))
    monkeypatch.setitem(HUD_BOXES, (60, 40), ((41, 1, 57, 5), (42, 25, 57, 36)))
    assert main(["--root",str(root),"clean"]) == 0
    assert main(["--root",str(root),"inventory"]) == 0
    assert main(["--root",str(root),"split"]) == 0
    split_audit = json.loads((root/"reports/split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["split_counts"] == {"train": 125, "val": 1, "test": 1}
    assert split_audit["split_support"] == {"train": 124, "val": 1, "test": 1}
    assert split_audit["cross_split_duplicates"] == []
    assert main(["--root",str(root),"test-all"]) == 1
    original=(root/"splits/test.csv").read_bytes()
    (root/"splits/test.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError): verify_stage(root,"splits")
    (root/"splits/test.csv").write_bytes(original)

    model=root/"model.bin"
    model.write_bytes(b"fixture weights")
    snapshot={"sd_revision":"a"*40,"files":{"efficientnet_weights":{"path":"model.bin","sha256":sha256_file(model)}}}
    monkeypatch.setattr(preflight,"resolve_all_model_snapshots",lambda root:snapshot)
    monkeypatch.setattr(preflight,"validate_model_snapshot",lambda root,snapshot:None)
    monkeypatch.setattr(preflight,"_require_cuda",lambda:None)
    monkeypatch.setattr(preflight,"_configure_determinism",lambda:None)
    monkeypatch.setattr(preflight,"_build_environment",lambda root,snapshot:{"runtime":"fixture","code_sha256":preflight.source_code_sha256(root)})
    monkeypatch.setattr(preflight,"_classifier_probe_batch",lambda root,snapshot,config,size:None)
    monkeypatch.setattr(preflight,"_reset_probe_state",lambda:None)
    monkeypatch.setattr(preflight,"_diffusion_probe",lambda root,snapshot,config:{"actual_timesteps":[1],"weights_sha256":"b"*64,"scheduler_config":{"x":1}})
    assert main(["--root",str(root),"preflight"]) == 0
    monkeypatch.setattr(workflow,"load_pinned_diffusion",lambda root,snapshot:object())
    def generate(source, attempt, config, snapshot):
        assert attempt["source_image_path"].startswith("data/cleaned_dataset/")
        g=native_generation_geometry(*source.size)
        data=np.random.default_rng(attempt["generation_seed"]).integers(0,256,(source.height,source.width,3),dtype=np.uint8)
        return Image.fromarray(data), {**g, "safety_flag":False,
            "diffusion_model":config["diffusion_model"],"model_revision":"a"*40,"weights_sha256":"b"*64,"scheduler_config":{"x":1},
            "prompt":config["prompt"].format(class_name=attempt["class_name"]),"negative_prompt":config["negative_prompt"],
            "strength":config["strength"],"guidance_scale":config["guidance_scale"],"num_inference_steps":config["num_inference_steps"],
            "eta":config["eta"],"precision":"float32","actual_timesteps":[1]}
    monkeypatch.setattr(workflow,"generate_diffusion",generate)
    assert main(["--root",str(root),"augment"]) == 0
    assert main(["--root",str(root),"review-export","--output","augmentation/manifests/review_queue.jsonl"]) == 0
    queue=[json.loads(line) for line in (root/"augmentation/manifests/review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(queue)==2
    for row in queue:
        source = root / row["source_image_path"]
        raw = root / "data/compressed_dataset" / source.relative_to(root / "data/cleaned_dataset")
        copied = root / "augmentation/review/sources" / f'{row["synthetic_id"]}.jpg'
        assert copied.read_bytes() == source.read_bytes() != raw.read_bytes()
    decisions=[{"synthetic_id":row["synthetic_id"],"source_sha256":row["source_sha256"],"output_sha256":row["output_sha256"],
                "reviewer_id":"fixture","review_timestamp":"2026-11-14T00:00:00Z","rubric_version":row["rubric_version"],"decision":"accept","reason":None} for row in queue]
    atomic_write_bytes(root/"reviews.jsonl",canonical_jsonl_bytes(decisions))
    assert main(["--root",str(root),"review-import","--input","reviews.jsonl"]) == 0
    assert main(["--root",str(root),"augment"]) == 0
    assert verify_stage(root,"membership")["stage"]=="membership"
    def diagnostics(root):
        assert verify_stage(root, "membership")["stage"] == "membership"
        for relative in ("augmentation/manifests/accepted.jsonl", "augmentation/manifests/qc_summary.json",
                         "splits/train_augmented.csv"):
            assert not (root / relative).exists()
        rows=[json.loads(line) for line in (root/"augmentation/manifests/candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        train = {row["filepath"] for row in csv.DictReader((root/"splits/train.csv").read_text(encoding="utf-8").splitlines())}
        assert all(row["source_image_path"] in train and row["source_image_path"].startswith("data/cleaned_dataset/") for row in rows)
        rows[0]["dino_source_similarity"]=None
        rows[0]["dino_diagnostic_error"]="fixture diagnostic failure"
        return rows
    monkeypatch.setattr(workflow,"log_dino_diagnostics",diagnostics)
    assert main(["--root",str(root),"augment-freeze"]) == 0
    assert verify_stage(root,"augmented")["stage"]=="augmented"

    monkeypatch.setattr(classifier,"build_model",lambda path:torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3,124)))
    def classifier_input(path):
        assert path.relative_to(root).parts[1] in {"cleaned_dataset", "augmented_dataset"}
        return torch.tensor([[[.2]],[[.3]],[[.4]]])
    monkeypatch.setattr(classifier,"classifier_transform",classifier_input)
    monkeypatch.setattr(training,"_training_device",lambda:torch.device("cpu"))
    real_loader=torch.utils.data.DataLoader
    def loader(*args,**kwargs):
        kwargs["num_workers"]=0
        kwargs["pin_memory"]=False
        return real_loader(*args,**kwargs)
    monkeypatch.setattr(torch.utils.data,"DataLoader",loader)
    assert main(["--root",str(root),"init"]) == 0
    real_publish=training._publish_training_state
    interrupted={"pending":True}
    def publish(path,state):
        real_publish(path,state)
        if interrupted["pending"] and state["epoch"]==1:
            interrupted["pending"]=False
            raise OSError("fixture interruption")
    monkeypatch.setattr(training,"_publish_training_state",publish)
    def run(command,env,cwd):
        condition=command[command.index("--condition")+1]
        seed=int(command[command.index("--seed")+1])
        monkeypatch.setenv("PYTHONHASHSEED",str(seed))
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        try:
            training.train_run(root,condition,seed)
            code=0
        except OSError:
            code=1
        return type("Result",(),{"returncode":code})()
    monkeypatch.setattr(subprocess,"run",run)
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert main(["--root",str(root),"train-all"]) == 1
        assert main(["--root",str(root),"test-all"]) == 1
        assert main(["--root",str(root),"train-all"]) == 0
        assert main(["--root",str(root),"freeze-checkpoints"]) == 0
        assert main(["--root",str(root),"test-all"]) == 0
        assert main(["--root",str(root),"report"]) == 0
    finally:
        torch.set_num_threads(old_threads)
    assert verify_stage(root,"reported")["stage"]=="reported"
    report=json.loads((root/"reports/aggregate_metrics.json").read_text(encoding="utf-8"))
    assert len(report["per_class"])==124
    assert report["groups"]["many"]["macro_f1"]["original"]["mean"] is None
