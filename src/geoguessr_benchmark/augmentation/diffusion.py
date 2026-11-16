"""Stable Diffusion canvas preparation and content-only output geometry."""
from collections.abc import Mapping

import numpy as np
from PIL import Image

GENERATION_MULTIPLE = 8
GENERATION_PADDING_MODE = "reflect"


def canonical_scheduler_config(config):
    """Copy scheduler metadata in JSON form, sorting only unordered default names."""
    if not isinstance(config, Mapping):
        raise TypeError("scheduler config must be a mapping")

    def normalize(value):
        if isinstance(value, Mapping):
            normalized = {}
            for key, item in value.items():
                if key == "_use_default_values":
                    if not isinstance(item, (list, tuple)) or any(not isinstance(name, str) for name in item):
                        raise ValueError("scheduler _use_default_values must be a sequence of names")
                    normalized[key] = sorted(item)
                else:
                    normalized[key] = normalize(item)
            return normalized
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(config)


def native_generation_geometry(width, height):
    """Describe the native content and the smallest centered VAE canvas."""
    if min(width, height) < 1:
        raise ValueError("generation dimensions must be positive")
    extra_w = -width % GENERATION_MULTIPLE
    extra_h = -height % GENERATION_MULTIPLE
    left, top = extra_w // 2, extra_h // 2
    return {"source_content_width": width, "source_content_height": height,
            "generation_canvas_width": width + extra_w, "generation_canvas_height": height + extra_h,
            "source_bbox": [left, top, left + width, top + height],
            "padding_mode": GENERATION_PADDING_MODE,
            "padding": {"left": left, "right": extra_w - left, "top": top, "bottom": extra_h - top},
            "output_content_only": True, "output_width": width, "output_height": height}


def prepare_generation_canvas(rgb):
    if rgb.mode != "RGB":
        raise ValueError("generation source must be RGB")
    geometry = native_generation_geometry(*rgb.size)
    pad = geometry["padding"]
    padded = np.pad(np.asarray(rgb), ((pad["top"], pad["bottom"]), (pad["left"], pad["right"]), (0, 0)),
                    mode=GENERATION_PADDING_MODE)
    return Image.fromarray(padded, "RGB"), geometry


def crop_generated_content(canvas, geometry):
    expected = native_generation_geometry(geometry["source_content_width"], geometry["source_content_height"])
    if canvas.mode != "RGB" or canvas.size != (expected["generation_canvas_width"], expected["generation_canvas_height"]) or geometry != expected:
        raise ValueError("invalid generated native canvas or geometry")
    return canvas.crop(expected["source_bbox"])


def generate_diffusion(source, attempt, config, snapshot):
    """Run a supplied pinned pipeline and return its cropped content plus provenance.

    The caller is responsible for resolving and verifying the pipeline snapshot before
    calling this adapter; no model is downloaded or selected here.
    """
    import torch

    if not isinstance(source, Image.Image):
        from ..images import decode_rgb
        source = decode_rgb(source)
    canvas, geometry = prepare_generation_canvas(source)
    pipeline = snapshot["pipeline"]
    generator = torch.Generator(device=snapshot.get("device", "cuda")).manual_seed(attempt["generation_seed"])
    timesteps = []
    def on_step(_pipeline, _index, timestep, callback_kwargs):
        timesteps.append(int(timestep.item() if hasattr(timestep, "item") else timestep))
        return callback_kwargs
    device_type = torch.device(snapshot.get("device", "cuda")).type
    with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=False):
        result = pipeline(prompt=config["prompt"].format(class_name=attempt["class_name"]),
                          negative_prompt=config["negative_prompt"], image=canvas,
                          strength=config["strength"], guidance_scale=config["guidance_scale"],
                          num_inference_steps=config["num_inference_steps"], eta=config["eta"],
                          num_images_per_prompt=1, generator=generator, callback_on_step_end=on_step)
    flags = getattr(result, "nsfw_content_detected", None)
    if len(result.images) != 1 or flags is None or len(flags) != 1 or not timesteps:
        raise ValueError("diffusion output missing image, safety flag or actual timesteps")
    generated = result.images[0]
    content = crop_generated_content(generated, geometry)
    import hashlib
    from ..artifacts import canonical_json_bytes
    entries = snapshot["files"]["sd_weights"]
    if not isinstance(entries, list):
        entries = [entries]
    weights = [entry for entry in entries if entry["path"].endswith(".safetensors") and ".fp16." not in entry["path"]]
    if not weights:
        weights = [entry for entry in entries if entry["path"].endswith(".bin") and ".fp16." not in entry["path"]]
    if not weights:
        raise ValueError("snapshot does not identify FP32 UNet weight files")
    weight_hashes = {entry["path"]:entry["sha256"] for entry in sorted(weights, key=lambda item:item["path"])}
    weights_sha256 = next(iter(weight_hashes.values())) if len(weight_hashes) == 1 else hashlib.sha256(canonical_json_bytes(weight_hashes)).hexdigest()
    scheduler_config = canonical_scheduler_config(pipeline.scheduler.config)
    metadata = {**geometry,
                "actual_timesteps": timesteps, "safety_flag": bool(flags[0]),
                "diffusion_model": config.get("diffusion_model"), "model_revision": snapshot["sd_revision"],
                "weights_sha256":weights_sha256, "weights_files_sha256":weight_hashes, "scheduler_config":scheduler_config,
                "prompt": config["prompt"].format(class_name=attempt["class_name"]),
                "negative_prompt": config["negative_prompt"], "strength": config["strength"],
                "guidance_scale": config["guidance_scale"], "num_inference_steps": config["num_inference_steps"],
                "eta": config["eta"], "precision": "float32"}
    return content, metadata


def load_pinned_diffusion(snapshot_root, snapshot, device="cuda"):
    """Load only verified local snapshot files with DDIM and standard FP32 attention."""
    import torch
    from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
    from diffusers.models.attention_processor import AttnProcessor
    from ..preflight import validate_model_snapshot, validate_sd_directory

    validate_model_snapshot(snapshot_root, snapshot)
    directory = validate_sd_directory(snapshot_root, snapshot)
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(str(directory), local_files_only=True, torch_dtype=torch.float32, use_safetensors=True)
    if pipeline.safety_checker is None:
        raise ValueError("diffusion safety checker is required")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.set_attn_processor(AttnProcessor())
    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "set_attn_processor"):
        vae.set_attn_processor(AttnProcessor())
    pipeline = pipeline.to(device)
    for name in ("unet", "vae", "text_encoder", "safety_checker"):
        module = getattr(pipeline, name, None)
        if module is not None and hasattr(module, "parameters"):
            for parameter in module.parameters():
                if parameter.is_floating_point() and parameter.dtype != torch.float32:
                    raise ValueError(f"diffusion component {name} is not FP32")
    return pipeline
