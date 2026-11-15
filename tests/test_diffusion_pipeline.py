from PIL import Image

from geoguessr_benchmark.augmentation.diffusion import generate_diffusion


class StubPipeline:
    scheduler = type("Scheduler", (), {"config":{"num_train_timesteps":1000,
        "_use_default_values":["thresholding", "prediction_type"], "trained_betas":(.01, .02)}})()
    def __call__(self, **kwargs):
        assert kwargs["image"].size == (384,160)
        assert kwargs["strength"] == .35 and kwargs["guidance_scale"] == 5.0
        assert kwargs["num_inference_steps"] == 40 and kwargs["eta"] == 0.0
        assert kwargs["negative_prompt"] == "negative"
        callback = kwargs["callback_on_step_end"]
        callback(self, 0, 99, {})
        callback(self, 1, 77, {})
        return type("Output", (), {"images":[Image.new("RGB",(384,160),"blue")], "nsfw_content_detected":[False]})()


def test_diffusion_records_actual_steps_and_crops_before_saving():
    source = Image.new("RGB",(1536,662),"green")
    attempt = {"generation_seed":123, "class_name":"Country"}
    config = {"prompt":"road {class_name}", "negative_prompt":"negative", "strength":.35,
              "guidance_scale":5.0, "num_inference_steps":40, "eta":0.0}
    snapshot={"pipeline":StubPipeline(), "device":"cpu", "sd_revision":"a"*40,
              "files":{"sd_weights":[{"path":"unet/model.safetensors","sha256":"b"*64}]}}
    image, metadata = generate_diffusion(source, attempt, config, snapshot)
    assert image.size == (371,160)
    assert metadata["source_bbox"] == [6,0,377,160]
    assert metadata["actual_timesteps"] == [99,77]
    assert metadata["output_content_only"] is True
    assert metadata["safety_flag"] is False
    assert metadata["weights_sha256"] == "b"*64
    assert metadata["scheduler_config"]["num_train_timesteps"] == 1000
    assert metadata["scheduler_config"]["_use_default_values"] == ["prediction_type", "thresholding"]
    assert metadata["scheduler_config"]["trained_betas"] == [.01, .02]
    assert snapshot["pipeline"].scheduler.config["_use_default_values"] == ["thresholding", "prediction_type"]


def test_pinned_pipeline_uses_local_files_ddim_and_standard_attention(tmp_path, monkeypatch):
    import hashlib
    import diffusers
    from diffusers.models import attention_processor
    from geoguessr_benchmark.augmentation.diffusion import load_pinned_diffusion
    sd = tmp_path / "sd"
    sd.mkdir()
    files = {}
    for component in ("sd_weights", "tokenizer", "vae", "scheduler", "dino_weights", "efficientnet_weights"):
        path = (sd if component in {"sd_weights","tokenizer","vae","scheduler"} else tmp_path) / component
        path.write_text(component, encoding="utf-8")
        files[component] = {"path":path.relative_to(tmp_path).as_posix(), "sha256":hashlib.sha256(component.encode()).hexdigest()}
    snapshot = {"sd_directory":"sd", "sd_revision":"a"*40, "dino_revision":"b"*40, "files":files}
    calls = {}
    class Unet:
        def set_attn_processor(self, processor):
            calls["attention"] = processor
    class FakePipeline:
        scheduler = type("Scheduler", (), {"config":{"foo": "bar"}})()
        unet = Unet()
        safety_checker = object()
        def to(self, device):
            calls["device"] = device
            return self
    def from_pretrained(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return FakePipeline()
    monkeypatch.setattr(diffusers.StableDiffusionImg2ImgPipeline, "from_pretrained", from_pretrained)
    monkeypatch.setattr(diffusers.DDIMScheduler, "from_config", lambda config: ("ddim", config))
    pipe = load_pinned_diffusion(tmp_path, snapshot, device="cpu")
    assert calls["local_files_only"] is True
    assert calls["device"] == "cpu"
    assert isinstance(calls["attention"], attention_processor.AttnProcessor)
    assert pipe.scheduler == ("ddim", {"foo":"bar"})


def test_diffusion_forces_inference_without_autocast():
    import torch
    class Check(StubPipeline):
        def __call__(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert not torch.is_autocast_enabled("cpu")
            return super().__call__(**kwargs)
    config = {"prompt":"road {class_name}", "negative_prompt":"negative", "strength":.35,
              "guidance_scale":5.0, "num_inference_steps":40, "eta":0.0}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        generate_diffusion(Image.new("RGB",(1536,662),"green"), {"generation_seed":123, "class_name":"Country"}, config, {"pipeline":Check(), "device":"cpu", "sd_revision":"a"*40, "files":{"sd_weights":[{"path":"unet/model.safetensors","sha256":"b"*64}]}})
