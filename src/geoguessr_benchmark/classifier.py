"""Classifier model and shared deterministic preprocessing."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.models import efficientnet_b0

from .images import decode_rgb, letterbox_geometry


def classifier_transform(path):
    image = decode_rgb(path)
    geometry = letterbox_geometry(image.width, image.height, 384, 160)
    values = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(tensor, size=(geometry["height"], geometry["width"]), mode="bilinear", align_corners=False, antialias=True)[0]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    result = torch.zeros((3, 160, 384), dtype=torch.float32)
    left, top, right, bottom = geometry["source_bbox"]
    result[:, top:bottom, left:right] = (tensor - mean) / std
    return result


def build_model(weights_path):
    if not Path(weights_path).is_file():
        raise ValueError("local pretrained weights are required")
    model = efficientnet_b0(weights=None)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 124)
    return model


def build_optimizer(model, config):
    return torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                             lr=config["learning_rate"], weight_decay=config["weight_decay"],
                             betas=tuple(config["betas"]), eps=config["eps"])


def build_loss(config):
    if config["loss_reduction"] != "mean" or config["label_smoothing"] != 0.0 or config["class_weights"] is not None:
        raise ValueError("benchmark requires unweighted mean cross-entropy without smoothing")
    return torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
