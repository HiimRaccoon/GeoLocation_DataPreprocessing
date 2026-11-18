"""Read and validate the frozen benchmark configuration."""
from pathlib import Path
import yaml
from .artifacts import BENCHMARK_ID

COMPONENTS = {"E0": "original", "E1": "traditional", "E2": "da_fusion", "E3": "geo_disc_ds"}

_BENCHMARK = {
    "benchmark_id": BENCHMARK_ID, "conditions": ["original", "augmented"],
    "components": COMPONENTS,
    "model": "efficientnet_b0", "weights": "IMAGENET1K_V1", "num_classes": 124,
    "input_width": 384, "input_height": 160, "optimizer": "AdamW", "learning_rate": 0.0003,
    "weight_decay": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8, "epochs": 20,
    "scheduler": "linear_warmup_then_cosine", "warmup_epochs": 1, "warmup_start_factor": 0.1,
    "minimum_learning_rate": 1e-6, "loss": "CrossEntropyLoss", "loss_reduction": "mean",
    "label_smoothing": 0.0, "class_weights": None, "batch_size_preferred": 128,
    "batch_size_fallback_order": [64, 32, 16, 8, 4, 2], "gradient_accumulation_steps": 1,
    "gradient_clipping": None, "precision": "float32", "amp": False, "shuffle_train": True,
    "replacement_sampling": False, "drop_last": False, "num_workers": 4, "pin_memory": True,
    "persistent_workers": False, "split_seed": 42, "augmentation_seed": 314159,
    "training_seeds": [42, 123, 2026], "early_stopping": False,
    "checkpoint_metric": "val_macro_f1_evaluable", "checkpoint_mode": "max", "checkpoint_tie_break": "earliest_epoch",
    "groups": {"many": {"min_train": 100, "method": "none"},
               "medium": {"min_train": 20, "max_train": 99, "method": "traditional", "target": 100},
               "tail": {"max_train": 19, "method": "da_fusion", "multiplier": 5, "maximum_target": 100}},
}
_TRADITIONAL = {
    "augmentation_seed": 314159, "method_variant": "traditional_fixed", "choose_k": [1, 2],
    "transform_order": ["brightness", "contrast", "gamma", "gaussian_noise", "gaussian_blur", "rotation", "jpeg_variation"],
    "brightness": [0.9, 1.1], "contrast": [0.9, 1.1], "gamma": [0.9, 1.1],
    "gaussian_noise_sigma": [1, 3], "gaussian_blur_sigma": [0.1, 0.6], "gaussian_blur_kernel": 5,
    "rotation_degrees": [-3, 3], "jpeg_quality": [85, 95], "output_quality": 95,
    "output_subsampling": 0, "attempt_cap_multiplier": 10,
    "qc": {"grayscale_std_min":5.0, "saturation_fraction_max":0.98, "near_black_max":2, "near_white_min":253},
}
_DIFFUSION = {
    "augmentation_seed": 314159, "method_variant": "sd15_img2img_fixed",
    "diffusion_model": "stable-diffusion-v1-5/stable-diffusion-v1-5", "pipeline": "StableDiffusionImg2ImgPipeline",
    "scheduler": "DDIMScheduler", "scheduler_source": "diffusion_model_scheduler_config", "eta": 0.0,
    "strength": 0.35, "guidance_scale": 5.0, "num_inference_steps": 40, "num_images_per_prompt": 1,
    "generation_batch_size": 1, "generation_resolution_policy": "native_pad_to_multiple",
    "generation_multiple": 8, "generation_padding_mode": "reflect", "generation_downsample": False,
    "precision": "float32", "attention_backend": "standard", "xformers": False, "textual_inversion": False,
    "lora": False, "diffusion_finetuning": False, "output_format": "jpeg", "output_quality": 95,
    "output_subsampling": 0, "attempt_cap_multiplier": 10,
    "qc": {"grayscale_std_min":5.0, "saturation_fraction_max":0.98, "near_black_max":2, "near_white_min":253, "blur_min":5.0, "blur_source_ratio":0.5},
    "prompt": "A realistic street-level road photograph in {class_name}, consistent with the visual characteristics of the country. Create a natural variation of the source scene while keeping it geographically plausible. Allow realistic changes in viewpoint, lighting, weather, road surface, vegetation, buildings, vehicles, street furniture, and local visual details. Keep the scene coherent and photorealistic without copying the source image too literally.",
    "negative_prompt": "illustration, painting, cartoon, surreal scene, impossible road geometry, reversed traffic, malformed roads, distorted perspective, deformed vehicles, duplicated objects, nonsensical signage, unreadable text artifacts, added country-name text, added flags, label watermark, severe blur, compression blocks, artificial textures",
}


def load_config(root):
    root = Path(root)
    result = {}
    for name, path, fixed in (
        ("benchmark", "configs/benchmark.yaml", _BENCHMARK),
        ("traditional", "augmentation/traditional/config.yaml", _TRADITIONAL),
        ("da_fusion", "augmentation/da_fusion/config.yaml", _DIFFUSION),
    ):
        with (root / path).open(encoding="utf-8-sig") as file:
            value = yaml.safe_load(file)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid config mapping: {path}")
        for key, expected in fixed.items():
            if key not in value or value[key] != expected or type(value[key]) is not type(expected):
                raise ValueError(f"Frozen config mismatch: {path}:{key}")
        extra = set(value) - set(fixed)
        if name == "benchmark" and extra == {"batch_size_effective"}:
            if value["batch_size_effective"] not in [128, 64, 32, 16, 8, 4, 2] or type(value["batch_size_effective"]) is not int:
                raise ValueError("invalid effective batch size")
        elif extra:
            raise ValueError(f"Unexpected config keys: {path}: {sorted(extra)}")
        result[name] = value
    return result
