"""Preflight primitives; the benchmark batch may only change after CUDA OOM."""


def memory_ladder(probe, reset):
    import torch
    candidates = (128, 64, 32, 16, 8, 4, 2)
    for index, batch_size in enumerate(candidates):
        try:
            probe(batch_size)
        except torch.OutOfMemoryError:
            if index == len(candidates) - 1:
                raise
        else:
            return batch_size
        finally:
            reset()
    raise RuntimeError("unreachable preflight state")


def validate_model_snapshot(snapshot_root, snapshot):
    """Verify immutable revisions and every file declared by a resolved snapshot."""
    import re
    from pathlib import Path
    from .artifacts import _resolve_artifact, sha256_file

    root = Path(snapshot_root).resolve()
    for field in ("sd_revision", "dino_revision"):
        if not isinstance(snapshot.get(field), str) or not re.fullmatch(r"[0-9a-f]{40}", snapshot[field]):
            raise ValueError(f"immutable {field} revision required")
    files = snapshot.get("files")
    required = {"sd_weights", "tokenizer", "vae", "scheduler", "dino_weights", "efficientnet_weights"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("missing required model snapshot components")
    for component, values in files.items():
        entries = values if isinstance(values, list) else [values]
        if not entries:
            raise ValueError(f"empty snapshot component: {component}")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
                raise ValueError(f"invalid snapshot component: {component}")
            path = _resolve_artifact(root, entry["path"])
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"model checksum mismatch: {component}")
    return snapshot


def validate_sd_directory(snapshot_root, snapshot):
    """Ensure every file in the local SD snapshot is in the checksum inventory."""
    from pathlib import Path
    from .artifacts import _resolve_artifact, sha256_file

    root = Path(snapshot_root).resolve()
    directory = _resolve_artifact(root, snapshot["sd_directory"])
    if not directory.is_dir():
        raise ValueError("SD snapshot directory missing")
    declared = {}
    for values in snapshot["files"].values():
        for entry in values if isinstance(values, list) else [values]:
            path = _resolve_artifact(root, entry["path"])
            if path.is_relative_to(directory):
                declared[path] = entry["sha256"]
    actual = set()
    for path in directory.rglob("*"):
        if ".cache" in path.relative_to(directory).parts:
            continue
        if path.is_symlink():
            raise ValueError("symlinks not allowed in verified SD snapshot")
        if path.is_file():
            actual.add(path.resolve())
    if not actual or actual != set(declared):
        raise ValueError("SD snapshot has missing or unverified files")
    if any(sha256_file(path) != declared[path] for path in actual):
        raise ValueError("SD snapshot checksum mismatch")
    return directory


def _select_sd_files(files):
    required = {"model_index.json", "feature_extractor/preprocessor_config.json",
                "safety_checker/config.json", "safety_checker/model.safetensors", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "text_encoder/model.safetensors", "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
                "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors"}
    missing = required - set(files)
    if missing:
        raise ValueError(f"SD repository is missing required FP32 files: {sorted(missing)}")
    return sorted(required)


def resolve_sd_snapshot(snapshot_root, relative_directory):
    """Resolve SD 1.5 to an immutable revision before downloading its local copy."""
    import re
    import json
    from pathlib import Path
    from huggingface_hub import HfApi, snapshot_download
    from .artifacts import _resolve_artifact, atomic_write_bytes, canonical_json_bytes, sha256_file

    root = Path(snapshot_root).resolve()
    directory = _resolve_artifact(root, relative_directory)
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    marker = directory.parent / (directory.name + ".resolution.json")
    if marker.exists():
        resolution = json.loads(marker.read_text(encoding="utf-8"))
        if resolution.get("repo_id") != repo_id or not isinstance(resolution.get("files"), list):
            raise ValueError("invalid SD snapshot resolution marker")
        commit = resolution.get("revision")
        expected_files = resolution["files"]
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("SD snapshot destination is not owned by this preflight")
        info = HfApi().model_info(repo_id, revision="main")
        commit = info.sha
        expected_files = _select_sd_files(item.rfilename for item in info.siblings)
        if not expected_files or len(expected_files) != len(set(expected_files)):
            raise ValueError("SD repository file inventory is unavailable")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("could not resolve immutable SD revision")
        atomic_write_bytes(marker, canonical_json_bytes({"repo_id":repo_id, "revision":commit, "files":expected_files}))
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("could not resolve immutable SD revision")
    snapshot_download(repo_id=repo_id, revision=commit, local_dir=str(directory), allow_patterns=expected_files)
    components = {"sd_weights": [], "tokenizer": [], "vae": [], "scheduler": [], "sd_other": []}
    prefixes = {"unet": "sd_weights", "tokenizer": "tokenizer", "vae": "vae", "scheduler": "scheduler"}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in resolved SD snapshot")
        if not path.is_file() or ".cache" in path.relative_to(directory).parts:
            continue
        component = prefixes.get(path.relative_to(directory).parts[0], "sd_other")
        components[component].append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    if any(not components[name] for name in ("sd_weights", "tokenizer", "vae", "scheduler")):
        raise ValueError("resolved SD snapshot missing required components")
    actual_files = sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*")
                          if path.is_file() and ".cache" not in path.relative_to(directory).parts)
    if actual_files != sorted(expected_files):
        raise ValueError("SD snapshot differs from immutable repository file inventory")
    if not components["sd_other"]:
        del components["sd_other"]
    result = {"sd_directory": relative_directory, "sd_revision": commit, "files": components}
    validate_sd_directory(root, result)
    return result


def resolve_all_model_snapshots(root):
    """Resolve and locally fingerprint the fixed pretrained dependencies."""
    import json
    from pathlib import Path
    from torchvision.models import EfficientNet_B0_Weights
    from .artifacts import atomic_write_bytes, canonical_json_bytes, sha256_file

    root = Path(root).resolve()
    manifest_path = root / "models/model_snapshot.json"
    if manifest_path.exists():
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_model_snapshot(root, snapshot)
        validate_sd_directory(root, snapshot)
        return snapshot
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    snapshot = resolve_sd_snapshot(root, "models/sd15")
    code, revision = _resolve_dino_source(root)
    code_files = []
    for path in sorted(code.rglob("*")):
        relative = path.relative_to(code)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("DINO source contains symlink")
        if path.is_file():
            code_files.append({"path":path.relative_to(root).as_posix(), "sha256":sha256_file(path)})
    snapshot["dino_revision"] = revision
    snapshot["dino_code_directory"] = "models/dino_code"
    snapshot["files"]["dino_code"] = code_files
    snapshot["files"]["dino_weights"] = _download_snapshot_file(root, "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", "models/dino_vits16.pth")
    snapshot["files"]["efficientnet_weights"] = _download_snapshot_file(root, EfficientNet_B0_Weights.IMAGENET1K_V1.url, "models/efficientnet_b0_imagenet1k_v1.pth")
    validate_model_snapshot(root, snapshot)
    atomic_write_bytes(manifest_path, canonical_json_bytes(snapshot))
    return snapshot


def _configure_determinism():
    import os
    if "torch" in __import__("sys").modules:
        import torch
        if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU is required for benchmark preflight")


def _reset_probe_state():
    import torch
    torch.cuda.empty_cache()


def _classifier_probe_batch(root, snapshot, config, size):
    import torch
    from .artifacts import _resolve_artifact
    from .classifier import build_loss, build_model, build_optimizer
    weights = _resolve_artifact(root, snapshot["files"]["efficientnet_weights"]["path"])
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = build_model(weights).to("cuda").train()
    optimizer = build_optimizer(model, config)
    inputs = torch.zeros((size,3,160,384), dtype=torch.float32, device="cuda")
    labels = torch.zeros(size, dtype=torch.long, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=False):
        logits = model(inputs)
        loss = build_loss(config)(logits, labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def _diffusion_probe(root, snapshot, config):
    import csv
    import torch
    from .artifacts import resolve_cleaned_image, sha256_file
    from .images import decode_rgb
    from .augmentation.diffusion import generate_diffusion, load_pinned_diffusion, native_generation_geometry
    with (root / "splits/train.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "reports/dataset_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = {row["filepath"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError("preflight requires a cleaned original train source")

    def probe_order(row):
        entry = inventory[row["filepath"]]
        geometry = native_generation_geometry(int(entry["width"]), int(entry["height"]))
        canvas_area = geometry["generation_canvas_width"] * geometry["generation_canvas_height"]
        return (-canvas_area, -int(entry["width"]) * int(entry["height"]), row["filepath"])

    if any(row["filepath"] not in inventory or row["split"] != "train" or row["is_synthetic"] != "false" for row in rows):
        raise ValueError("diffusion probe requires verified cleaned train rows")
    source = min(rows, key=probe_order)
    path = resolve_cleaned_image(root, source["filepath"])
    if sha256_file(path) != source["sha256"] or source["sha256"] != inventory[source["filepath"]]["sha256"]:
        raise ValueError("diffusion probe source hash differs from inventory or train manifest")
    image = decode_rgb(path)
    entry = inventory[source["filepath"]]
    if image.size != (int(entry["width"]), int(entry["height"])):
        raise ValueError("diffusion probe source dimensions differ from inventory")
    geometry = native_generation_geometry(*image.size)
    attempt = {"generation_seed":314159, "class_name":source["class_name"]}
    try:
        pipeline = load_pinned_diffusion(root, snapshot)
        output, details = generate_diffusion(image, attempt, config, {**snapshot, "pipeline":pipeline, "device":"cuda"})
    except torch.OutOfMemoryError as error:
        width, height = geometry["generation_canvas_width"], geometry["generation_canvas_height"]
        raise ValueError(f"CUDA OOM during native FP32 diffusion preflight at {width}x{height}, batch 1") from error
    if (output.mode != "RGB" or output.size != image.size or details.get("precision") != "float32"
            or any(details.get(key) != value for key, value in geometry.items())):
        raise ValueError("diffusion preflight native geometry or precision mismatch")
    return {**details, "probe_source_path": source["filepath"], "probe_source_sha256": source["sha256"]}


def source_code_sha256(root):
    import hashlib
    from pathlib import Path
    root = Path(root).resolve()
    paths = sorted([p for p in (root / "src").rglob("*.py") if p.is_file()] + [root / "pyproject.toml", root / "docs/specs.md"])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def verify_source(root):
    import json
    from pathlib import Path
    root = Path(root).resolve()
    frozen = json.loads((root / "environment.lock.txt").read_text(encoding="utf-8"))
    if frozen.get("code_sha256") != source_code_sha256(root):
        raise ValueError("runtime code hash differs from preflight")


def _build_environment(root, snapshot):
    import importlib.metadata
    import platform
    import subprocess
    import sys
    import torch
    from pathlib import Path
    from PIL import features
    root = Path(root)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        raise ValueError("unable to record GPU driver version") from error
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "PyYAML", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"required package missing: {name}") from error
    return {"python":sys.version, "os":platform.platform(), "packages":packages, "gpu":torch.cuda.get_device_name(0),
            "driver":driver, "cuda":torch.version.cuda, "cudnn":torch.backends.cudnn.version(),
            "image_backend":"Pillow/libjpeg + PyTorch bilinear antialias", "jpeg_version":features.version("jpg"),
            "rounding":"half-up geometry; numpy rint uint8", "code_sha256":source_code_sha256(root),
            "sd_revision":snapshot["sd_revision"], "dino_revision":snapshot["dino_revision"],
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark":torch.backends.cudnn.benchmark,
            "cudnn_deterministic":torch.backends.cudnn.deterministic,
            "tf32":bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32), "precision":"float32"}


def verify_runtime(root, lock):
    """Reject changed source code, libraries, hardware, or deterministic settings."""
    import json
    from pathlib import Path
    root = Path(root).resolve()
    frozen = json.loads((root / "environment.lock.txt").read_text(encoding="utf-8"))
    current = _build_environment(root, lock["model_snapshot"])
    for key, expected in frozen.items():
        if key in ("batch_size_effective", "diffusion_probe"):
            continue
        if key not in current or current[key] != expected:
            raise ValueError(f"runtime mismatch: {key}")
    return current


def run_preflight(root):
    """Publish preflight only after CUDA classifier and SD probes both succeed."""
    import random
    from pathlib import Path
    import numpy as np
    import yaml
    from .artifacts import atomic_write_bytes, canonical_json_bytes, sha256_file, verify_stage
    from .config import load_config
    root = Path(root).resolve()
    lock = verify_stage(root, "splits")
    if lock["stage"] != "splits":
        verify_source(root)
        return verify_stage(root, "preflight")
    config = load_config(root)
    _configure_determinism()
    _require_cuda()
    import torch
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        snapshot = resolve_all_model_snapshots(root)
        validate_model_snapshot(root, snapshot)
        size = memory_ladder(lambda batch:_classifier_probe_batch(root,snapshot,config["benchmark"],batch), _reset_probe_state)
        diffusion = _diffusion_probe(root,snapshot,config["da_fusion"])
        environment = _build_environment(root,snapshot)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
        _reset_probe_state()
    config["benchmark"]["batch_size_effective"] = size
    config_bytes = yaml.safe_dump(config["benchmark"], sort_keys=False, allow_unicode=True).encode("utf-8")
    environment.update({"batch_size_effective":size, "diffusion_probe":diffusion})
    atomic_write_bytes(root / "configs/benchmark.yaml", config_bytes)
    atomic_write_bytes(root / "environment.lock.txt", canonical_json_bytes(environment))
    artifacts = dict(lock["artifacts"])
    for name in ("configs/benchmark.yaml", "augmentation/traditional/config.yaml", "augmentation/da_fusion/config.yaml", "environment.lock.txt"):
        artifacts[name] = sha256_file(root / name)
    for values in snapshot["files"].values():
        for entry in values if isinstance(values, list) else [values]:
            if sha256_file(root / entry["path"]) != entry["sha256"]:
                raise ValueError("model artifact changed during preflight")
            artifacts[entry["path"]] = entry["sha256"]
    result = dict(lock, stage="preflight", artifacts=artifacts, model_snapshot=snapshot,
                  batch_size_effective=size, diffusion_probe=diffusion)
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(result))
    return result


def _download_snapshot_file(root, url, relative):
    """Download once with an ownership marker and safe interrupted-transfer retry."""
    import json
    import os
    from pathlib import Path
    import tempfile
    import urllib.request
    from .artifacts import _resolve_artifact, atomic_write_bytes, canonical_json_bytes, sha256_file
    root = Path(root).resolve()
    destination = _resolve_artifact(root, relative)
    destination.parent.mkdir(parents=True,exist_ok=True)
    marker = destination.with_name(destination.name + ".download.json")
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("url") != url:
            raise ValueError("weight destination is owned by another download")
        if state.get("sha256"):
            if not destination.is_file() or sha256_file(destination) != state["sha256"]:
                raise ValueError("downloaded weight checksum mismatch")
            return {"path":relative,"sha256":state["sha256"]}
    else:
        if destination.exists():
            raise ValueError("weight destination is not owned by this preflight")
        atomic_write_bytes(marker,canonical_json_bytes({"url":url,"sha256":None}))
    temp_name = None
    try:
        with urllib.request.urlopen(url) as response, tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as output:
            temp_name=output.name
            for block in iter(lambda:response.read(1024*1024),b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        new_hash=sha256_file(temp_name)
        if destination.exists():
            if sha256_file(destination) != new_hash:
                raise ValueError("incomplete owned weight differs from verified download")
        else:
            os.replace(temp_name,destination)
        atomic_write_bytes(marker,canonical_json_bytes({"url":url,"sha256":new_hash}))
        return {"path":relative,"sha256":new_hash}
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _resolve_dino_source(root):
    """Resolve DINO once, preserving interrupted owned clones for inspection."""
    import json
    import re
    import subprocess
    import uuid
    from pathlib import Path
    from .artifacts import atomic_write_bytes, canonical_json_bytes
    root=Path(root).resolve()
    models=root/"models"
    models.mkdir(parents=True,exist_ok=True)
    code=models/"dino_code"
    marker=models/"dino_code.resolution.json"
    remote="https://github.com/facebookresearch/dino.git"
    if marker.exists():
        saved=json.loads(marker.read_text(encoding="utf-8"))
        revision=saved.get("revision")
        if saved.get("remote") != remote or not isinstance(revision,str) or not re.fullmatch(r"[0-9a-f]{40}",revision):
            raise ValueError("invalid DINO resolution marker")
    else:
        if code.exists():
            raise ValueError("DINO source directory is not owned by this preflight")
        try:
            result=subprocess.run(["git","ls-remote",remote,"HEAD"],check=True,capture_output=True,text=True)
            revision=result.stdout.split()[0]
        except (OSError,subprocess.CalledProcessError,IndexError) as error:
            raise ValueError("unable to resolve DINO source revision") from error
        if not re.fullmatch(r"[0-9a-f]{40}",revision):
            raise ValueError("invalid DINO source revision")
        atomic_write_bytes(marker,canonical_json_bytes({"remote":remote,"revision":revision}))
    if code.exists():
        try:
            head=subprocess.run(["git","-C",str(code),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
            status=subprocess.run(["git","-C",str(code),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout
        except (OSError,subprocess.CalledProcessError):
            head,status=None,"incomplete"
        if head == revision and not status.strip():
            return code,revision
        code.rename(models/("dino_code.incomplete."+uuid.uuid4().hex))
    try:
        subprocess.run(["git","clone","--no-checkout",remote,str(code)],check=True,capture_output=True)
        subprocess.run(["git","-C",str(code),"checkout","--detach",revision],check=True,capture_output=True)
    except (OSError,subprocess.CalledProcessError) as error:
        raise ValueError("unable to complete DINO source snapshot") from error
    return code,revision
