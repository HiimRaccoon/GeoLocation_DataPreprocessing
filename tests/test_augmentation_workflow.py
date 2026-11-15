import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from geoguessr_benchmark.artifacts import _REQUIRED, atomic_write_bytes, canonical_json_bytes, canonical_jsonl_bytes, sha256_file, verify_stage
from geoguessr_benchmark.images import decode_rgb, pixel_sha256, letterbox_geometry
from geoguessr_benchmark.augmentation.policy import plan_candidate
from geoguessr_benchmark.augmentation.diagnostics import membership_sha256
from geoguessr_benchmark.augmentation.workflow import freeze_augmented
from test_qc import setup_fixture

DIFF_CONFIG = {"diffusion_model":"fixture-model", "prompt":"road {class_name}", "negative_prompt":"negative", "strength":.35, "guidance_scale":5.0, "num_inference_steps":40, "eta":0.0}

def fixture_provenance(class_name):
    return {"diffusion_model":"fixture-model", "model_revision":"a"*40, "weights_sha256":"b"*64, "scheduler_config":{"x":1},
            "prompt":f"road {class_name}", "negative_prompt":"negative", "strength":.35, "guidance_scale":5.0,
            "num_inference_steps":40, "eta":0.0, "precision":"float32", "actual_timesteps":[9,8]}


def test_freeze_checks_dino_membership_then_publishes_union(tmp_path, monkeypatch):
    source, _ = setup_fixture(tmp_path)
    with (tmp_path / "splits/train.csv").open(encoding="utf-8",newline="") as file:
        train = list(csv.DictReader(file))
    with (tmp_path / "splits/class_groups.csv").open(encoding="utf-8",newline="") as file:
        group = list(csv.DictReader(file))[0]
    rng = np.random.default_rng(11)
    candidates, decisions = [], []
    for i in range(int(group["required_accepted"])):
        attempt = plan_candidate("da_fusion", 0, i, train)
        src = next(r for r in train if r["image_id"] == attempt["source_image_id"])
        original = decode_rgb(tmp_path / src["filepath"])
        geometry = letterbox_geometry(original.width,original.height,384,160)
        name = f'data/augmented_dataset/da_fusion/A/{attempt["synthetic_id"]}.jpg'
        output = tmp_path / name
        output.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rng.integers(0,256,(geometry["height"], geometry["width"],3),dtype=np.uint8)).save(output,quality=95,subsampling=0)
        candidate = dict(attempt, synthetic_path=name, source_sha256=src["sha256"], class_name="A", method_variant="sd15_img2img_fixed", accepted=True, filter_status="accepted", source_bbox=geometry["source_bbox"], generation_canvas_width=384, generation_canvas_height=160, resized_source_width=geometry["width"], resized_source_height=geometry["height"], output_content_only=True, safety_flag=False, output_sha256=sha256_file(output), output_pixel_sha256=pixel_sha256(decode_rgb(output)))
        candidate.update({"reviewer_id":"reviewer","review_timestamp":"2026-11-14T00:00:00Z","review_rubric_version":"geographic-cues-v1","review_reason":None})
        candidates.append(candidate)
        decisions.append({"synthetic_id":candidate["synthetic_id"], "source_sha256":candidate["source_sha256"], "output_sha256":candidate["output_sha256"], "reviewer_id":"reviewer", "review_timestamp":"2026-11-14T00:00:00Z", "rubric_version":"geographic-cues-v1", "decision":"accept", "reason":None})
    candidates_path=tmp_path/"augmentation/manifests/candidates.jsonl"
    atomic_write_bytes(candidates_path,canonical_jsonl_bytes(candidates))
    atomic_write_bytes(tmp_path/"augmentation/manifests/reviews.jsonl",canonical_jsonl_bytes(decisions))
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for stage in ("preflight","membership"):
        for path in _REQUIRED[stage]:
            if not (tmp_path/path).exists(): atomic_write_bytes(tmp_path/path,b"fixture")
            lock["artifacts"][path]=sha256_file(tmp_path/path)
    for candidate in candidates:
        candidate.update(fixture_provenance(candidate["class_name"]))
        candidate.update({"augmentation_seed":314159, "config_sha256":lock["artifacts"]["augmentation/da_fusion/config.yaml"],
                          "environment_lock_sha256":lock["artifacts"]["environment.lock.txt"]})
    atomic_write_bytes(candidates_path,canonical_jsonl_bytes(candidates))
    lock["artifacts"]["augmentation/manifests/candidates.jsonl"]=sha256_file(candidates_path)
    lock["model_snapshot"]={"sd_revision":"a"*40}
    lock["diffusion_probe"]={"weights_sha256":"b"*64,"scheduler_config":{"x":1},"actual_timesteps":[9,8]}
    lock["artifacts"]["augmentation/manifests/reviews.jsonl"] = sha256_file(tmp_path/"augmentation/manifests/reviews.jsonl")
    lock.update({"stage":"membership","membership_sha256":membership_sha256(candidates)})
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    import geoguessr_benchmark.augmentation.workflow as workflow
    monkeypatch.setattr(workflow,"load_config",lambda root:{"traditional":{},"da_fusion":DIFF_CONFIG})
    def changed(root):
        result = [dict(row) for row in candidates]
        result[0]["accepted"] = False
        return result
    monkeypatch.setattr(workflow,"log_dino_diagnostics",changed)
    with pytest.raises(ValueError,match="membership"):
        freeze_augmented(tmp_path)
    assert not (tmp_path/"augmentation/manifests/accepted.jsonl").exists()
    assert not (tmp_path/"splits/train_augmented.csv").exists()
    monkeypatch.setattr(workflow,"log_dino_diagnostics",lambda root: candidates)
    incident = tmp_path / "augmentation/manifests/holdout_incident.json"
    atomic_write_bytes(incident, canonical_json_bytes({"reason":"holdout_collision"}))
    with pytest.raises(ValueError, match="holdout"):
        freeze_augmented(tmp_path)
    incident.unlink()
    real_write = workflow.atomic_write_bytes
    def fail_during_publish(path, payload):
        if Path(path).name == "split_metadata.json" and (tmp_path / "augmentation/manifests/.freeze-transaction/journal.json").exists():
            raise OSError("interrupted publish")
        return real_write(path, payload)
    monkeypatch.setattr(workflow, "atomic_write_bytes", fail_during_publish)
    with pytest.raises(OSError, match="interrupted"):
        freeze_augmented(tmp_path)
    monkeypatch.setattr(workflow, "atomic_write_bytes", real_write)
    from geoguessr_benchmark.__main__ import main
    assert main(["--root", str(tmp_path), "augment-freeze"]) == 0
    result = json.loads((tmp_path / "benchmark_lock.json").read_text(encoding="utf-8"))
    assert result["stage"] == "augmented"
    assert verify_stage(tmp_path,"augmented")
    with (tmp_path/"splits/train_augmented.csv").open(encoding="utf-8",newline="") as file:
        union=list(csv.DictReader(file))
    assert len(union) == len(train)+len(candidates)
    assert len({r["image_id"] for r in union}) == len(union)
    summary=json.loads((tmp_path/"augmentation/manifests/qc_summary.json").read_text(encoding="utf-8"))
    assert summary["classes"][0]["target"] == int(group["target"])
    assert summary["classes"][0]["coverage"] == 1.0
    assert summary["methods"]["da_fusion"]["accepted"] == len(candidates)



def test_rejected_output_collision_with_holdout_blocks_freeze(tmp_path):
    from geoguessr_benchmark.augmentation.workflow import audit_holdout_collisions
    setup_fixture(tmp_path)
    with (tmp_path / "splits/test.csv").open(encoding="utf-8",newline="") as file:
        holdout = next(csv.DictReader(file))
    out = tmp_path / "rejected.jpg"
    out.write_bytes((tmp_path / holdout["filepath"]).read_bytes())
    with pytest.raises(ValueError, match="holdout"):
        audit_holdout_collisions(tmp_path, [{"synthetic_path":"rejected.jpg", "filter_status":"rejected", "accepted":False}])


def test_dino_may_only_add_diagnostic_fields(tmp_path):
    from geoguessr_benchmark.augmentation.workflow import validate_dino_changes
    before = [{"synthetic_id":"x","accepted":True,"filter_status":"accepted","source_image_id":"s","output_sha256":"a","output_pixel_sha256":"b","class_id":7}]
    after = [dict(before[0], dino_source_similarity=.5)]
    validate_dino_changes(before, after)
    after[0]["class_id"] = 8
    with pytest.raises(ValueError, match="DINO"):
        validate_dino_changes(before, after)


def test_augment_resume_from_offline_review_to_membership(tmp_path, monkeypatch):
    import geoguessr_benchmark.augmentation.workflow as workflow
    import geoguessr_benchmark.preflight as preflight
    import geoguessr_benchmark.config as config_module
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    from geoguessr_benchmark.augmentation.review import export_review_queue, import_review_decisions
    from geoguessr_benchmark.__main__ import main
    from test_qc import setup_fixture
    setup_fixture(tmp_path)
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for path in _REQUIRED["preflight"]:
        if not (tmp_path/path).exists(): atomic_write_bytes(tmp_path/path,b"fixture")
        lock["artifacts"][path]=sha256_file(tmp_path/path)
    lock.update({"stage":"preflight","model_snapshot":{"sd_revision":"a"*40},"diffusion_probe":{"weights_sha256":"b"*64,"scheduler_config":{"x":1},"actual_timesteps":[9,8]}})
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    monkeypatch.setattr(workflow,"load_config",lambda root:{"benchmark":{},"traditional":{},"da_fusion":DIFF_CONFIG})
    monkeypatch.setattr(config_module,"load_config",lambda root:{"benchmark":{},"traditional":{},"da_fusion":DIFF_CONFIG})
    monkeypatch.setattr(workflow,"load_pinned_diffusion",lambda root,snapshot:object())
    def fake_generate(source, attempt, config, snapshot):
        g=letterbox_geometry(source.width,source.height,384,160)
        rng=np.random.default_rng(attempt["generation_seed"])
        img=Image.fromarray(rng.integers(0,256,(g["height"],g["width"],3),dtype=np.uint8))
        return img, {**fixture_provenance(attempt["class_name"]), "generation_canvas_width":384,"generation_canvas_height":160,"source_bbox":g["source_bbox"],"resized_source_width":g["width"],"resized_source_height":g["height"],"output_content_only":True,"safety_flag":False}
    monkeypatch.setattr(workflow,"generate_diffusion",fake_generate)
    assert main(["--root",str(tmp_path),"augment"]) == 0
    candidates_path=tmp_path/"augmentation/manifests/candidates.jsonl"
    pending=[json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    assert len(pending)==8 and all(row["filter_status"]=="pending" for row in pending)
    assert main(["--root",str(tmp_path),"review-export","--output","augmentation/manifests/review_queue.jsonl"]) == 0
    queue=[json.loads(line) for line in (tmp_path/"augmentation/manifests/review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions=[{"synthetic_id":r["synthetic_id"],"source_sha256":r["source_sha256"],"output_sha256":r["output_sha256"],"reviewer_id":"reviewer","review_timestamp":"2026-11-14T00:00:00Z","rubric_version":r["rubric_version"],"decision":"accept","reason":None} for r in queue]
    review_input=tmp_path/"decisions.jsonl"
    atomic_write_bytes(review_input,canonical_jsonl_bytes([decisions[-1]]))
    assert main(["--root",str(tmp_path),"review-import","--input",str(review_input)]) == 0
    assert main(["--root",str(tmp_path),"augment"]) == 0
    still_pending=[json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["filter_status"] == "pending" for row in still_pending)
    atomic_write_bytes(review_input,canonical_jsonl_bytes(decisions))
    assert main(["--root",str(tmp_path),"review-import","--input",str(review_input)]) == 0
    assert main(["--root",str(tmp_path),"augment"]) == 0
    lock=verify_stage(tmp_path,"membership")
    assert lock["stage"] == "membership"
    final=[json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    assert all(r["accepted"] for r in final) and len(final)==8
    assert membership_sha256(final) == lock["membership_sha256"]


def test_augment_records_holdout_incident_and_attempt(tmp_path, monkeypatch):
    import geoguessr_benchmark.augmentation.workflow as workflow
    import geoguessr_benchmark.preflight as preflight
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    from geoguessr_benchmark.__main__ import main
    setup_fixture(tmp_path)
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for path in _REQUIRED["preflight"]:
        if not (tmp_path/path).exists(): atomic_write_bytes(tmp_path/path,b"fixture")
        lock["artifacts"][path]=sha256_file(tmp_path/path)
    lock.update({"stage":"preflight","model_snapshot":{}})
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    monkeypatch.setattr(workflow,"load_config",lambda root:{"benchmark":{},"traditional":{},"da_fusion":{}})
    monkeypatch.setattr(workflow,"load_pinned_diffusion",lambda root,snapshot:object())
    monkeypatch.setattr(workflow,"generate_diffusion",lambda source,attempt,config,snapshot:(Image.new("RGB",(240,160),"black"),{"safety_flag":False}))
    monkeypatch.setattr(workflow,"check_candidate",lambda *args:{"passed":False,"reason":"exact_duplicate","scores":{},"incident":"holdout_collision"})
    assert main(["--root",str(tmp_path),"augment"]) == 1
    assert (tmp_path/"augmentation/manifests/holdout_incident.json").exists()
    rows=[json.loads(line) for line in (tmp_path/"augmentation/manifests/candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==1 and rows[0]["rejection_reason"]=="exact_duplicate"


def test_review_binding_requires_matching_metadata_and_utc_timestamp():
    from geoguessr_benchmark.augmentation.workflow import validate_review_binding
    row={"source_sha256":"s","output_sha256":"o","reviewer_id":"r","review_timestamp":"2026-11-14T00:00:00Z","review_rubric_version":"geographic-cues-v1","review_reason":None}
    review={"source_sha256":"s","output_sha256":"o","reviewer_id":"r","review_timestamp":"2026-11-14T00:00:00Z","rubric_version":"geographic-cues-v1","decision":"accept","reason":None}
    assert validate_review_binding(row,review)
    assert not validate_review_binding(dict(row,reviewer_id="other"),review)
    assert not validate_review_binding(dict(row,review_timestamp="2026-11-14T00:00:00"),dict(review,review_timestamp="2026-11-14T00:00:00"))


def test_holdout_collision_stops_before_dino_diagnostics(tmp_path,monkeypatch):
    import geoguessr_benchmark.augmentation.workflow as workflow
    source,candidate=setup_fixture(tmp_path)
    with (tmp_path/"splits/test.csv").open(encoding="utf-8",newline="") as file:
        holdout=next(csv.DictReader(file))
    out=tmp_path/"rejected.jpg"
    out.write_bytes((tmp_path/holdout["filepath"]).read_bytes())
    candidate.update({"synthetic_path":"rejected.jpg","augmentation_type":"da_fusion","filter_status":"rejected","accepted":False})
    atomic_write_bytes(tmp_path/"augmentation/manifests/candidates.jsonl",canonical_jsonl_bytes([candidate]))
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for stage in ("preflight","membership"):
        for name in _REQUIRED[stage]:
            if not (tmp_path/name).exists(): atomic_write_bytes(tmp_path/name,b"fixture")
            lock["artifacts"][name]=sha256_file(tmp_path/name)
    lock.update({"stage":"membership","membership_sha256":membership_sha256([candidate])})
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    called=[]
    monkeypatch.setattr(workflow,"log_dino_diagnostics",lambda root:called.append(True))
    with pytest.raises(ValueError,match="holdout"):
        freeze_augmented(tmp_path)
    assert not called
    assert (tmp_path/"augmentation/manifests/holdout_incident.json").exists()
