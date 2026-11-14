import numpy as np
from PIL import Image
import torch
from torchvision.models import efficientnet_b0

from geoguessr_benchmark.classifier import classifier_transform, build_model, build_optimizer


def test_classifier_preprocessing_preserves_aspect_and_zero_padding(tmp_path):
    for width, height, left in [(1536,662,6), (1030,703,75), (371,160,6), (234,160,75)]:
        path = tmp_path / f"{width}.jpg"
        Image.new("RGB", (width,height), (80,120,160)).save(path)
        tensor = classifier_transform(path)
        assert tensor.dtype == torch.float32
        assert tuple(tensor.shape) == (3,160,384)
        assert torch.count_nonzero(tensor[:, :, :left]) == 0
        assert torch.equal(tensor, classifier_transform(path))


def test_model_uses_supplied_backbone_and_trainable_124_class_head(tmp_path):
    path = tmp_path / "backbone.pt"
    backbone = efficientnet_b0(weights=None)
    torch.save(backbone.state_dict(), path)
    model = build_model(path)
    assert model.classifier[1].out_features == 124
    assert model.classifier[0].p == 0.2
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert torch.equal(model.features[0][0].weight, backbone.features[0][0].weight)
    optimizer = build_optimizer(model, {"learning_rate": 0.0003, "weight_decay": 0.0001, "betas": [0.9,0.999], "eps": 1e-8})
    assert optimizer.defaults["weight_decay"] == 0.0001
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(list(model.parameters()))


def test_shared_loss_uses_unweighted_mean_cross_entropy():
    from geoguessr_benchmark.classifier import build_loss
    loss=build_loss({"loss_reduction":"mean","label_smoothing":0.0,"class_weights":None})
    assert isinstance(loss,torch.nn.CrossEntropyLoss)
    assert loss.reduction=="mean" and loss.label_smoothing==0.0 and loss.weight is None
