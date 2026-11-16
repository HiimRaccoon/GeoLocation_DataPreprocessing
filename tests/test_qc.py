import csv

import numpy as np
from PIL import Image
import pytest

from geoguessr_benchmark.artifacts import sha256_file
from geoguessr_benchmark.images import decode_rgb, pixel_sha256
from geoguessr_benchmark.inventory import build_inventory
from geoguessr_benchmark.split import build_split
from geoguessr_benchmark.augmentation.qc import check_candidate
from geoguessr_benchmark.augmentation.policy import plan_candidate
from geoguessr_benchmark.cleaning import HUD_BOXES
from cleaned_fixture import seal_cleaned_fixture


def setup_fixture(root):
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs/specs.md").write_text("fixture spec", encoding="utf-8")
    directory = root / "data/cleaned_dataset/A"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(4):
        Image.fromarray(rng.integers(0,256,(40,60,3), dtype=np.uint8)).save(directory / f"{i}.jpg")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(HUD_BOXES, (60, 40), ((0, 0, 1, 1),))
        seal_cleaned_fixture(root)
        build_inventory(root, baseline={"images":4,"classes":1})
        build_split(root)
    with (root / "splits/train.csv").open(encoding="utf-8", newline="") as file:
        train = list(csv.DictReader(file))
        source = train[0]
    output = root / "candidate.jpg"
    Image.fromarray(rng.integers(10,240,(40,60,3), dtype=np.uint8)).save(output, quality=95, subsampling=0)
    candidate = {"synthetic_path":"candidate.jpg", "source_image_id":source["image_id"],
                 "source_image_path":source["filepath"], "source_sha256":source["sha256"], "source_split":"train",
                 "class_name":"A", "class_id":0, "augmentation_type":"traditional", "method_variant":"traditional_fixed",
                 "output_sha256":sha256_file(output), "output_pixel_sha256":pixel_sha256(decode_rgb(output))}
    candidate.update(plan_candidate("traditional", 0, 0, train))
    return source, candidate


def test_traditional_qc_has_no_review_or_blur_gate(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    result = check_candidate(tmp_path, candidate, source, {}, {})
    assert result["passed"] is True
    assert result["incident"] is None
    assert "blur" not in result["scores"]


def test_native_diffusion_qc_rejects_stale_geometry_and_wrong_jpeg_dimensions(tmp_path):
    from geoguessr_benchmark.augmentation.diffusion import native_generation_geometry

    _, candidate = setup_fixture(tmp_path)
    with (tmp_path / "splits/train.csv").open(encoding="utf-8", newline="") as file:
        train = list(csv.DictReader(file))
    attempt = plan_candidate("da_fusion", 0, 0, train)
    source = next(row for row in train if row["image_id"] == attempt["source_image_id"])
    candidate.update(attempt)
    candidate.update(native_generation_geometry(60, 40))
    candidate.update(source_sha256=source["sha256"], method_variant="sd15_img2img_fixed")
    assert check_candidate(tmp_path, candidate, source, {}, {})["passed"]
    for change in ({"generation_canvas_width":384, "generation_canvas_height":160},
                   {"source_bbox":[0, 0, 60, 40]}, {"padding":{}}, {"output_width":61}):
        assert check_candidate(tmp_path, dict(candidate, **change), source, {}, {})["reason"] == "invalid_dimensions"
    path = tmp_path / candidate["synthetic_path"]
    decode_rgb(path).crop((0, 0, 59, 40)).save(path)
    candidate.update(output_sha256=sha256_file(path), output_pixel_sha256=pixel_sha256(decode_rgb(path)))
    assert check_candidate(tmp_path, candidate, source, {}, {})["reason"] == "invalid_dimensions"


def test_holdout_collision_is_incident_and_prevents_acceptance(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    index = {"holdout_pixel": {candidate["output_pixel_sha256"]}}
    result = check_candidate(tmp_path, candidate, source, {}, index)
    assert not result["passed"]
    assert result["reason"] == "exact_duplicate"
    assert result["incident"] == "holdout_collision"


def test_qc_rejects_exact_train_and_accepted_matches_and_corrupt_output(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    result = check_candidate(tmp_path, candidate, source, {}, {"accepted_file": {candidate["output_sha256"]}})
    assert result["reason"] == "exact_duplicate" and result["incident"] is None
    path = tmp_path / candidate["synthetic_path"]
    path.write_bytes((tmp_path / source["filepath"]).read_bytes())
    candidate.update(output_sha256=source["sha256"], output_pixel_sha256=source["decoded_pixel_sha256"])
    assert check_candidate(tmp_path, candidate, source, {}, {})["reason"] == "exact_duplicate"
    path.write_bytes(b"unreadable JPEG")
    assert check_candidate(tmp_path, candidate, source, {}, {})["reason"] == "unreadable"


def test_invalid_train_source_and_blank_output_are_rejected(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    bad_source = dict(source, split="test")
    assert check_candidate(tmp_path, candidate, bad_source, {}, {})["reason"] == "invalid_source"
    Image.new("RGB", (60,40), "white").save(tmp_path / "candidate.jpg")
    candidate["output_sha256"] = sha256_file(tmp_path / "candidate.jpg")
    candidate["output_pixel_sha256"] = pixel_sha256(decode_rgb(tmp_path / "candidate.jpg"))
    assert check_candidate(tmp_path, candidate, source, {}, {})["reason"] == "blank"


def test_holdout_collision_remains_incident_even_if_output_is_blank(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    Image.new("RGB", (60,40), "white").save(tmp_path / "candidate.jpg")
    candidate["output_sha256"] = sha256_file(tmp_path / "candidate.jpg")
    candidate["output_pixel_sha256"] = pixel_sha256(decode_rgb(tmp_path / "candidate.jpg"))
    result = check_candidate(tmp_path, candidate, source, {}, {"holdout_pixel": {candidate["output_pixel_sha256"]}})
    assert result["incident"] == "holdout_collision"


def test_qc_rejects_synthetic_attempt_with_wrong_seed(tmp_path):
    from geoguessr_benchmark.augmentation.policy import plan_candidate
    source, candidate = setup_fixture(tmp_path)
    with (tmp_path / "splits/train.csv").open(encoding="utf-8", newline="") as file:
        all_sources = list(csv.DictReader(file))
    attempt = plan_candidate("traditional", 0, 0, all_sources)
    matching = next(r for r in all_sources if r["image_id"] == attempt["source_image_id"])
    candidate.update(attempt)
    candidate.update({"source_sha256":matching["sha256"], "class_name":"A", "generation_seed":attempt["generation_seed"] + 1})
    assert check_candidate(tmp_path, candidate, matching, {}, {})["reason"] == "invalid_provenance"


def test_reuse_verified_qc_index_does_not_rehash_lock_per_candidate(tmp_path,monkeypatch):
    import geoguessr_benchmark.augmentation.qc as qc_module
    source,candidate=setup_fixture(tmp_path)
    context=qc_module.build_qc_index(tmp_path)
    monkeypatch.setattr(qc_module,"verify_stage",lambda *args: (_ for _ in ()).throw(AssertionError("rehashed lock")))
    assert qc_module.check_candidate(tmp_path,candidate,source,{}, {"context":context})["passed"]


def test_provenance_rejects_changed_diffusion_parameters_or_unpinned_weights():
    from copy import deepcopy
    from geoguessr_benchmark.augmentation.qc import validate_candidate_provenance
    config={"diffusion_model":"fixed-model","prompt":"road {class_name}","negative_prompt":"negative","strength":.35,"guidance_scale":5.0,"num_inference_steps":40,"eta":0.0}
    scheduler = {"prediction_type": "epsilon", "nested": {
        "_use_default_values": ["thresholding", "sample_max_value"], "trained_betas": [.01, .02]}}
    lock={"artifacts":{"augmentation/da_fusion/config.yaml":"config-sha","environment.lock.txt":"env-sha"},
          "model_snapshot":{"sd_revision":"a"*40},"diffusion_probe":{"weights_sha256":"b"*64,
          "scheduler_config":scheduler,"actual_timesteps":[9,8]}}
    source={"class_name":"A"}
    candidate={"augmentation_type":"da_fusion","augmentation_seed":314159,"config_sha256":"config-sha","environment_lock_sha256":"env-sha",
               "diffusion_model":"fixed-model","model_revision":"a"*40,"weights_sha256":"b"*64,"scheduler_config":deepcopy(scheduler),
               "prompt":"road A","negative_prompt":"negative","strength":.35,"guidance_scale":5.0,"num_inference_steps":40,
               "eta":0.0,"precision":"float32","actual_timesteps":[9,8],"safety_flag":False}
    candidate["scheduler_config"]["nested"]["_use_default_values"].reverse()
    before = deepcopy((candidate, lock))
    assert validate_candidate_provenance(candidate,source,config,lock)
    assert (candidate, lock) == before
    for field,value in (
        ("model_revision", "f" * 40), ("weights_sha256", "wrong"), ("strength", .4),
        ("prompt", "other"), ("negative_prompt", "other"), ("guidance_scale", 6.0),
        ("num_inference_steps", 39), ("eta", .1), ("precision", "float16"),
        ("safety_flag", 0), ("config_sha256", "wrong"), ("environment_lock_sha256", "wrong"),
        ("actual_timesteps", []), ("actual_timesteps", [8, 9]), ("actual_timesteps", [9, 7]),
    ):
        bad=dict(candidate,**{field:value})
        assert not validate_candidate_provenance(bad,source,config,lock), field
    for field, value in (("trained_betas", [.02, .01]), ("_use_default_values", ["thresholding"]),
                         ("_use_default_values", [7, "thresholding"])):
        bad = deepcopy(candidate)
        bad["scheduler_config"]["nested"][field] = value
        assert not validate_candidate_provenance(bad, source, config, lock), field
    bad = deepcopy(candidate)
    bad["scheduler_config"]["prediction_type"] = "v_prediction"
    assert not validate_candidate_provenance(bad, source, config, lock)


def test_provenance_validates_traditional_transform_order_and_ranges():
    from geoguessr_benchmark.augmentation.qc import validate_candidate_provenance
    lock={"artifacts":{"augmentation/traditional/config.yaml":"c","environment.lock.txt":"e"}}
    candidate={"augmentation_type":"traditional","augmentation_seed":314159,"config_sha256":"c","environment_lock_sha256":"e",
               "augmentation_parameters":[{"name":"brightness","factor":1.0},{"name":"rotation","degrees":2.0}]}
    assert validate_candidate_provenance(candidate,{}, {},lock)
    candidate["augmentation_parameters"].reverse()
    assert not validate_candidate_provenance(candidate,{}, {},lock)


def test_preflight_stage_qc_cannot_bypass_provenance_by_omitting_lock(tmp_path):
    import json
    from geoguessr_benchmark.artifacts import _REQUIRED,atomic_write_bytes,canonical_json_bytes,sha256_file
    source,candidate=setup_fixture(tmp_path)
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for name in _REQUIRED["preflight"]:
        if not (tmp_path/name).exists(): atomic_write_bytes(tmp_path/name,b"fixture")
        lock["artifacts"][name]=sha256_file(tmp_path/name)
    lock["stage"]="preflight"
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    assert check_candidate(tmp_path,candidate,source,{}, {})["reason"] == "invalid_provenance"


def test_basic_qc_thresholds_are_inclusive_for_std_and_strict_for_saturation():
    from geoguessr_benchmark.augmentation.qc import check_basic_intensities
    thresholds={"grayscale_std_min":5.0,"saturation_fraction_max":.98,"near_black_max":2,"near_white_min":253}
    assert check_basic_intensities(np.array([[100,110]],dtype=np.uint8),thresholds)[0]
    assert not check_basic_intensities(np.array([[100,109]],dtype=np.uint8),thresholds)[0]
    assert not check_basic_intensities(np.array([[0]*49+[100]],dtype=np.uint8),thresholds)[0]
    assert check_basic_intensities(np.array([[0]*48+[100,100]],dtype=np.uint8),thresholds)[0]
