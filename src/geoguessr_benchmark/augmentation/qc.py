"""Automatic acceptance checks on final JPEGs."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ..artifacts import _resolve_artifact, canonical_json_bytes, resolve_cleaned_image, sha256_file, verify_stage
from ..images import decode_rgb, pixel_sha256
from .diffusion import canonical_scheduler_config, native_generation_geometry
from .policy import plan_candidate


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class _QCContext:
    def __init__(self, root, train, holdout, lock):
        self.root = root
        self.lock = lock
        self.train_by_id = {row["image_id"]: row for row in train}
        self.train_by_class = {}
        for row in train:
            self.train_by_class.setdefault(row["class_id"], []).append(row)
        self.train_file = {row["sha256"] for row in train}
        self.train_pixel = {row["decoded_pixel_sha256"] for row in train}
        self.holdout_file = {row["sha256"] for row in holdout}
        self.holdout_pixel = {row["decoded_pixel_sha256"] for row in holdout}


def build_qc_index(root):
    root = Path(root).resolve()
    lock = verify_stage(root, "splits")
    train = _rows(root / "splits/train.csv")
    holdout = _rows(root / "splits/val.csv") + _rows(root / "splits/test.csv")
    return _QCContext(root, train, holdout, lock)


def _blur(image):
    w, h = image.size
    scale = 384 / max(w, h)
    size = (max(1, int(w * scale + .5)), max(1, int(h * scale + .5)))
    pixels = np.asarray(image.resize(size, Image.Resampling.BILINEAR).convert("L"), dtype=np.float64)
    p = np.pad(pixels, 1, mode="reflect")
    lap = p[:-2,1:-1] + p[2:,1:-1] + p[1:-1,:-2] + p[1:-1,2:] - 4 * p[1:-1,1:-1]
    return float(lap.var())


def check_basic_intensities(gray, thresholds):
    values = np.asarray(gray, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("grayscale image is empty or non-finite")
    std = float(values.std())
    saturated = float(np.mean((values <= thresholds["near_black_max"]) | (values >= thresholds["near_white_min"])))
    scores = {"grayscale_std":std, "black_white_fraction":saturated}
    return std >= thresholds["grayscale_std_min"] and saturated < thresholds["saturation_fraction_max"], scores


def check_candidate(root, candidate, source, config, hash_index):
    root = Path(root).resolve()
    context = hash_index.get("context")
    if not isinstance(context, _QCContext) or context.root != root:
        context = build_qc_index(root)
    def fail(reason, scores=None, incident=None):
        return {"passed": False, "reason": reason, "scores": scores or {}, "incident": incident}
    match = context.train_by_id.get(source.get("image_id"))
    if match is None or source != match or source.get("split") != "train" or source.get("is_synthetic") != "false":
        return fail("invalid_source")
    if (candidate.get("source_split") != "train" or candidate.get("source_image_id") != source["image_id"] or
        candidate.get("source_image_path") != source["filepath"] or candidate.get("source_sha256") != source["sha256"] or
        candidate.get("class_name") != source["class_name"] or int(candidate.get("class_id", -1)) != int(source["class_id"])):
        return fail("invalid_source")
    try:
        source_path = resolve_cleaned_image(root, source["filepath"])
    except ValueError:
        return fail("invalid_source")
    if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
        return fail("invalid_source")
    method = candidate.get("augmentation_type")
    if (method, candidate.get("method_variant")) not in (("traditional", "traditional_fixed"), ("da_fusion", "sd15_img2img_fixed")):
        return fail("invalid_provenance")
    try:
        class_sources = context.train_by_class.get(source["class_id"], [])
        expected = plan_candidate(method, int(source["class_id"]), candidate["attempt_index"], class_sources)
    except (KeyError, TypeError, ValueError):
        return fail("invalid_provenance")
    if any(candidate.get(key) != value for key, value in expected.items()):
        return fail("invalid_provenance")
    if context.lock["stage"] != "splits" and not validate_candidate_provenance(candidate, source, config, context.lock):
        return fail("invalid_provenance")
    if candidate.get("safety_flag"):
        return fail("safety_flag")
    try:
        path = _resolve_artifact(root, candidate["synthetic_path"])
        with Image.open(path) as check:
            if check.format != "JPEG" or check.mode != "RGB":
                return fail("invalid_rgb")
        rgb = decode_rgb(path)
        original = decode_rgb(source_path)
        output_hash = sha256_file(path)
        output_pixel_hash = pixel_sha256(rgb)
    except (OSError, ValueError, KeyError):
        return fail("unreadable")
    if (output_hash in context.holdout_file or output_pixel_hash in context.holdout_pixel
        or output_hash in hash_index.get("holdout_file", set()) or output_pixel_hash in hash_index.get("holdout_pixel", set())):
        return fail("exact_duplicate", incident="holdout_collision")
    if candidate.get("output_sha256") != output_hash or candidate.get("output_pixel_sha256") != output_pixel_hash:
        return fail("invalid_provenance")
    if method == "traditional":
        if rgb.size != original.size:
            return fail("invalid_dimensions")
    else:
        geometry = native_generation_geometry(*original.size)
        actual = {key: candidate.get(key) for key in geometry}
        try:
            valid_geometry = canonical_json_bytes(actual) == canonical_json_bytes(geometry)
        except (TypeError, ValueError):
            valid_geometry = False
        if not valid_geometry or rgb.size != original.size:
            return fail("invalid_dimensions")
    gray = np.asarray(rgb.convert("L"), dtype=np.float64)
    thresholds = config.get("qc", {"grayscale_std_min":5.0, "saturation_fraction_max":.98,
                                   "near_black_max":2, "near_white_min":253, "blur_min":5.0, "blur_source_ratio":.5})
    valid_intensity, scores = check_basic_intensities(gray, thresholds)
    if not valid_intensity:
        return fail("blank", scores)
    if method == "da_fusion":
        source_blur, output_blur = _blur(original), _blur(rgb)
        scores.update({"source_laplacian_variance": source_blur, "output_laplacian_variance": output_blur})
        if output_blur < max(thresholds["blur_min"], thresholds["blur_source_ratio"] * source_blur):
            return fail("blur", scores)
    if (output_hash in context.train_file or output_pixel_hash in context.train_pixel
        or output_hash in hash_index.get("accepted_file", set()) or output_pixel_hash in hash_index.get("accepted_pixel", set())):
        return fail("exact_duplicate", scores)
    return {"passed": True, "reason": None, "scores": scores, "incident": None}


def validate_candidate_provenance(candidate, source, config, lock):
    """Validate fixed generator identity and sampled parameters against the lock."""
    import math
    method = candidate.get("augmentation_type")
    artifacts = lock.get("artifacts", {})
    if method not in ("traditional", "da_fusion") or candidate.get("augmentation_seed") != 314159:
        return False
    if (candidate.get("config_sha256") != artifacts.get(f"augmentation/{method}/config.yaml") or
        candidate.get("environment_lock_sha256") != artifacts.get("environment.lock.txt") or
        candidate.get("config_sha256") is None or candidate.get("environment_lock_sha256") is None):
        return False
    if method == "traditional":
        order = ("brightness", "contrast", "gamma", "gaussian_noise", "gaussian_blur", "rotation", "jpeg_variation")
        parameters = candidate.get("augmentation_parameters")
        if not isinstance(parameters, list) or len(parameters) not in (1,2) or any(not isinstance(item, dict) or item.get("name") not in order for item in parameters):
            return False
        names = [item["name"] for item in parameters]
        if names != sorted(set(names), key=order.index):
            return False
        rules = {"brightness":("factor",.9,1.1), "contrast":("factor",.9,1.1), "gamma":("factor",.9,1.1),
                 "gaussian_noise":("sigma_255",1,3), "gaussian_blur":("sigma",.1,.6),
                 "rotation":("degrees",-3,3), "jpeg_variation":("quality",85,95)}
        for item in parameters:
            key, low, high = rules[item["name"]]
            value = item.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
                return False
            if item["name"] == "jpeg_variation" and type(value) is not int:
                return False
            if item["name"] == "gaussian_blur" and item.get("kernel") != 5:
                return False
        return True
    try:
        probe = lock["diffusion_probe"]
        snapshot = lock["model_snapshot"]
        checks = {"diffusion_model":config["diffusion_model"], "model_revision":snapshot["sd_revision"],
                  "weights_sha256":probe["weights_sha256"],
                  "prompt":config["prompt"].format(class_name=source["class_name"]), "negative_prompt":config["negative_prompt"],
                  "strength":config["strength"], "guidance_scale":config["guidance_scale"],
                  "num_inference_steps":config["num_inference_steps"], "eta":config["eta"], "precision":"float32"}
        steps = candidate.get("actual_timesteps")
        candidate_scheduler = canonical_json_bytes(canonical_scheduler_config(candidate["scheduler_config"]))
        frozen_scheduler = canonical_json_bytes(canonical_scheduler_config(probe["scheduler_config"]))
        return (all(candidate.get(key) == value for key,value in checks.items()) and
                candidate_scheduler == frozen_scheduler and
                isinstance(steps,list) and bool(steps) and all(type(step) is int for step in steps) and
                steps == probe["actual_timesteps"] and
                type(candidate.get("safety_flag")) is bool)
    except (KeyError, TypeError, ValueError):
        return False
