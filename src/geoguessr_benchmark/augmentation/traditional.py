"""Fixed mild traditional augmentation, sampled once per offline candidate."""
import io
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

_ORDER = ("brightness", "contrast", "gamma", "gaussian_noise", "gaussian_blur", "rotation", "jpeg_variation")


def generate_traditional(source, attempt, config):
    if source.mode != "RGB":
        raise ValueError("traditional source must be RGB")
    seed = attempt["generation_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid generation seed")
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 3))
    indices = sorted(int(i) for i in rng.choice(len(_ORDER), size=k, replace=False))
    rgb = np.asarray(source, dtype=np.float32).copy() / 255.0
    params = []
    for index in indices:
        name = _ORDER[index]
        if name in ("brightness", "contrast", "gamma"):
            value = float(rng.uniform(.9, 1.1))
            if name == "brightness":
                rgb *= value
            elif name == "contrast":
                gray_mean = float((rgb @ np.array([.299, .587, .114], dtype=np.float32)).mean())
                rgb = gray_mean + value * (rgb - gray_mean)
            else:
                rgb = np.power(rgb, value)
            params.append({"name": name, "factor": value})
        elif name == "gaussian_noise":
            sigma = float(rng.uniform(1, 3))
            rgb += rng.normal(0, sigma / 255.0, rgb.shape).astype(np.float32)
            params.append({"name": name, "sigma_255": sigma})
        elif name == "gaussian_blur":
            sigma = float(rng.uniform(.1, .6))
            positions = torch.arange(-2, 3, dtype=torch.float32)
            kernel = torch.exp(-positions.square() / (2 * sigma * sigma))
            kernel /= kernel.sum()
            kernel2d = torch.outer(kernel, kernel).expand(3, 1, 5, 5)
            padded = np.pad(rgb, ((2,2), (2,2), (0,0)), mode="reflect")
            tensor = torch.from_numpy(padded.copy()).permute(2,0,1)[None]
            rgb = functional.conv2d(tensor, kernel2d, groups=3)[0].permute(1,2,0).numpy()
            params.append({"name": name, "sigma": sigma, "kernel": 5})
        elif name == "rotation":
            degrees = float(rng.uniform(-3, 3))
            angle = math.radians(degrees)
            h, w = rgb.shape[:2]
            c, s = math.cos(angle), math.sin(angle)
            matrix = torch.tensor([[[c, -s*h/w, 0], [s*w/h, c, 0]]], dtype=torch.float32)
            tensor = torch.from_numpy(rgb.copy()).permute(2,0,1)[None]
            grid = functional.affine_grid(matrix, tensor.shape, align_corners=False)
            rgb = functional.grid_sample(tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0].permute(1,2,0).numpy()
            params.append({"name": name, "degrees": degrees})
        else:
            quality = int(rng.integers(85, 96))
            buffer = io.BytesIO()
            Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as intermediate:
                rgb = np.asarray(intermediate.convert("RGB"), dtype=np.float32) / 255.0
            params.append({"name": name, "quality": quality})
        rgb = np.clip(rgb, 0, 1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8)), params
