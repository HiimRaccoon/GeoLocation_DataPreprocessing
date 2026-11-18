from pathlib import Path
import shutil

import pytest
import yaml

from geoguessr_benchmark.config import load_config

PROJECT = Path(__file__).resolve().parents[1]


def copy_configs(root):
    shutil.copytree(PROJECT / "configs", root / "configs")
    shutil.copytree(PROJECT / "augmentation", root / "augmentation")


@pytest.mark.parametrize("file,key,value", [
    ("configs/benchmark.yaml", "split_seed", 77),
    ("configs/benchmark.yaml", "amp", True),
    ("configs/benchmark.yaml", "input_width", 512),
    ("configs/benchmark.yaml", "benchmark_id", "other"),
    ("augmentation/da_fusion/config.yaml", "strength", 0.5),
    ("augmentation/da_fusion/config.yaml", "generation_multiple", 16),
    ("augmentation/da_fusion/config.yaml", "generation_downsample", True),
    ("augmentation/da_fusion/config.yaml", "generation_canvas_width", 384),
    ("augmentation/da_fusion/config.yaml", "prompt",
     "A realistic street-level road photograph in {class_name}, Google Street View style, "
     "preserve the original road layout, traffic side, lane markings, existing signs and visible writing, "
     "buildings, utility poles, vegetation and vehicle configuration, natural lighting, subtle realistic variation."),
    ("augmentation/da_fusion/config.yaml", "negative_prompt",
     "illustration, painting, cartoon, flipped road, reversed traffic, altered writing, invented road signs, "
     "distorted geometry, impossible objects, deformed vehicles, duplicated objects, severe blur, "
     "compression blocks, artificial textures, new watermark"),
    ("augmentation/da_fusion/config.yaml", "qc", {"grayscale_std_min": 1.0}),
])
def test_locked_configuration_changes_are_rejected(tmp_path, file, key, value):
    copy_configs(tmp_path)
    path = tmp_path / file
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config[key] = value
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_config_does_not_prematurely_materialize_batch(tmp_path):
    copy_configs(tmp_path)
    config = load_config(tmp_path)
    assert config["benchmark"]["batch_size_preferred"] == 128
    assert config["benchmark"].get("batch_size_effective") is None
    assert config["benchmark"]["conditions"] == ["original", "augmented"]
    assert config["benchmark"]["training_seeds"] == [42, 123, 2026]
    assert config["benchmark"]["components"] == {
        "E0": "original", "E1": "traditional", "E2": "da_fusion", "E3": "geo_disc_ds",
    }


def test_qc_thresholds_are_explicit_and_frozen(tmp_path):
    copy_configs(tmp_path)
    config=load_config(tmp_path)
    assert config["traditional"]["qc"]["grayscale_std_min"] == 5.0
    assert config["da_fusion"]["qc"]["blur_source_ratio"] == .5


def test_diffusion_prompts_and_parameters_are_frozen():
    config = load_config(PROJECT)["da_fusion"]
    assert config["prompt"] == "A realistic street-level road photograph in {class_name}, consistent with the visual characteristics of the country. Create a natural variation of the source scene while keeping it geographically plausible. Allow realistic changes in viewpoint, lighting, weather, road surface, vegetation, buildings, vehicles, street furniture, and local visual details. Keep the scene coherent and photorealistic without copying the source image too literally."
    assert config["negative_prompt"] == "illustration, painting, cartoon, surreal scene, impossible road geometry, reversed traffic, malformed roads, distorted perspective, deformed vehicles, duplicated objects, nonsensical signage, unreadable text artifacts, added country-name text, added flags, label watermark, severe blur, compression blocks, artificial textures"
    assert config["strength"] == 0.35
    assert config["guidance_scale"] == 5.0
    assert config["num_inference_steps"] == 40
    assert config["eta"] == 0.0
    assert config["precision"] == "float32"
