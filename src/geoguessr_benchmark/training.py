"""Training utilities for the fixed epoch-progress schedule."""
import math


def learning_rate_at_step(step, steps_per_epoch, config):
    if type(step) is not int or type(steps_per_epoch) is not int or steps_per_epoch <= 0 or not 0 <= step < config["epochs"] * steps_per_epoch:
        raise ValueError("step must be inside the training schedule")
    u = step / steps_per_epoch
    base = config["learning_rate"]
    if u < 1:
        return base * (0.1 + 0.9 * u)
    minimum = config["minimum_learning_rate"]
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * (u - 1) / (config["epochs"] - 1)))


def epoch_permutation(size, generator):
    import torch
    if type(size) is not int or size < 1:
        raise ValueError("dataset size must be positive")
    return torch.randperm(size, generator=generator).tolist()


def worker_seed(initial_seed):
    if type(initial_seed) is not int or initial_seed < 0:
        raise ValueError("worker seed must be a non-negative integer")
    return initial_seed % (2**32)


def _cpu_snapshot(value):
    import copy
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


from .artifacts import verify_stage


def _training_device():
    import torch
    if not torch.cuda.is_available():
        raise ValueError("training requires the preflight CUDA device")
    return torch.device("cuda")


def _init_worker(_worker_id):
    import random
    import numpy as np
    import torch
    seed = worker_seed(torch.initial_seed())
    random.seed(seed)
    np.random.seed(seed)


class _ManifestDataset:
    def __init__(self, root, rows):
        self.root, self.rows = root, rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        from .classifier import classifier_transform
        row = self.rows[index]
        return classifier_transform(self.root / row["filepath"]), int(row["class_id"]), row["image_id"]


def _capture_rng(generator, device):
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python":random.getstate(), "numpy":[state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            "torch":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader":generator.get_state()}


def _restore_rng(state, generator, device):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def _publish_training_state(run_dir, state):
    import io
    import json
    import torch
    from .artifacts import atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes
    def save(path, value):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        atomic_write_bytes(path, buffer.getvalue())
    save(run_dir / "last_checkpoint.pt", state)
    save(run_dir / "best_checkpoint.pt", state["best_checkpoint"])
    history_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "learning_rate", "steps", "duration_seconds", "sample_ids"]
    history = [{**row, "sample_ids":json.dumps(row["sample_ids"], separators=(",", ":"))} for row in state["history"]]
    atomic_write_bytes(run_dir / "history.csv", canonical_csv_bytes(history, history_fields))
    atomic_write_bytes(run_dir / "lr_steps.csv", canonical_csv_bytes(state["lr_steps"], ["global_step", "epoch", "step_in_epoch", "learning_rate"]))
    evaluation = state["best_evaluation"]
    atomic_write_bytes(run_dir / "metrics.json", canonical_json_bytes({"validation_at_best":evaluation["metrics"], "final_test":None}))
    atomic_write_bytes(run_dir / "group_metrics.json", canonical_json_bytes({"validation_at_best":evaluation["groups"], "final_test":None}))
    per_class_fields = ["class_id", "support", "tp", "fp", "fn", "precision", "recall", "f1", "evaluable", "status"]
    atomic_write_bytes(run_dir / "per_class_metrics_val.csv", canonical_csv_bytes(evaluation["per_class"], per_class_fields))
    fields = ["true_class_id"] + [str(i) for i in range(124)]
    matrix = [{"true_class_id":i, **{str(j):value for j,value in enumerate(row)}} for i,row in enumerate(evaluation["confusion"])]
    atomic_write_bytes(run_dir / "confusion_matrix_val.csv", canonical_csv_bytes(matrix, fields))


def train_run(root, condition, seed):
    """Train a single paired run, with epoch-boundary checkpoint recovery."""
    import copy
    import csv
    import json
    import os
    import random
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    import numpy as np
    import torch
    from .artifacts import _resolve_artifact, atomic_write_bytes, canonical_json_bytes, sha256_file
    from .classifier import build_loss, build_model, build_optimizer
    from .config import load_config
    from .metrics import evaluate_predictions
    from .preflight import _configure_determinism, verify_runtime

    root = Path(root).resolve()
    if condition not in ("original", "augmented") or seed not in (42, 123, 2026):
        raise ValueError("invalid training run identity")
    if os.environ.get("PYTHONHASHSEED") != str(seed) or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("training requires the seeded launcher environment")
    lock = verify_stage(root, "initialized")
    if lock["stage"] != "initialized":
        raise ValueError("training cannot change after checkpoints are frozen")
    config = load_config(root)["benchmark"]
    batch_size = config["batch_size_effective"]
    _configure_determinism()
    device = _training_device()
    verify_runtime(root, lock)
    names = {"train":"splits/train.csv" if condition == "original" else "splits/train_augmented.csv", "val":"splits/val.csv"}
    manifests = {}
    for split, path in names.items():
        with (root / path).open(encoding="utf-8", newline="") as stream:
            manifests[split] = list(csv.DictReader(stream))
        if not manifests[split]:
            raise ValueError(f"empty {split} manifest")
        for row in manifests[split]:
            file = _resolve_artifact(root, row["filepath"])
            if not file.is_file() or sha256_file(file) != row["sha256"] or row["split"] != split:
                raise ValueError(f"{split} manifest file changed")
            if split == "val" and row["is_synthetic"] != "false":
                raise ValueError("validation must contain only real images")
    meta = json.loads((root / "splits/split_metadata.json").read_text(encoding="utf-8"))
    with (root / "splits/class_groups.csv").open(encoding="utf-8", newline="") as stream:
        class_groups = {name:[] for name in ("many", "medium", "tail")}
        for row in csv.DictReader(stream):
            class_groups[row["group"]].append(int(row["class_id"]))
    init_name = f"initializations/seed_{seed}.pt"
    identity = {"condition":condition, "seed":seed, "train_sha256":lock["artifacts"][names["train"]],
                "val_sha256":lock["artifacts"][names["val"]], "init_sha256":lock["artifacts"][init_name],
                "config_sha256":lock["artifacts"]["configs/benchmark.yaml"],
                "environment_sha256":lock["artifacts"]["environment.lock.txt"],
                "group_sha256":lock["artifacts"]["splits/class_groups.csv"],
                "split_metadata_sha256":lock["artifacts"]["splits/split_metadata.json"]}
    weights = lock["model_snapshot"]["files"]["efficientnet_weights"]
    if lock["artifacts"].get(weights["path"]) != weights["sha256"]:
        raise ValueError("pretrained weights are not locked")
    model = build_model(_resolve_artifact(root, weights["path"]))
    initial = torch.load(root / init_name, map_location="cpu", weights_only=True)
    model.load_state_dict(initial, strict=True)
    model = model.to(device)
    optimizer = build_optimizer(model, config)
    loss_function = build_loss(config)
    train_generator = torch.Generator().manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    run_dir = root / "runs" / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    state_path = run_dir / "last_checkpoint.pt"
    history, lr_steps, start_epoch, global_step = [], [], 0, 0
    best_score, best_checkpoint, best_evaluation = None, None, None
    failures, resumes = [], []
    start_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_identity = json.loads((root / "environment.lock.txt").read_text(encoding="utf-8"))
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        previous_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous_meta.get("identity") != identity:
            raise ValueError("existing run metadata is incompatible")
        failures = previous_meta.get("failure_history", [])
        resumes = previous_meta.get("resume_history", [])
        start_time = previous_meta.get("start_timestamp", start_time)
    atomic_write_bytes(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=True).encode("utf-8"))
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("identity") != identity:
            raise ValueError("resume checkpoint identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        _restore_rng(state["rng"], train_generator, device)
        history, lr_steps = state["history"], state["lr_steps"]
        start_epoch, global_step = state["epoch"], state["global_step"]
        best_score, best_checkpoint, best_evaluation = state["best_score"], state["best_checkpoint"], state["best_evaluation"]
        if len(history) != start_epoch or len(lr_steps) != global_step:
            raise ValueError("checkpoint history or step count is inconsistent")
        _publish_training_state(run_dir, state)
        if start_epoch < config["epochs"]:
            resumes.append({"timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "after_epoch":start_epoch})
    total_steps_per_epoch = math.ceil(len(manifests["train"]) / batch_size)
    try:
        for epoch in range(start_epoch + 1, config["epochs"] + 1):
            started = time.monotonic()
            model.train()
            loader = torch.utils.data.DataLoader(_ManifestDataset(root, manifests["train"]), batch_size=batch_size, shuffle=True,
                generator=train_generator, num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                persistent_workers=False, drop_last=False, worker_init_fn=_init_worker)
            total_loss, total_items, sample_ids = 0.0, 0, []
            for step_in_epoch, (images, labels, ids) in enumerate(loader):
                images, labels = images.to(device=device, dtype=torch.float32), labels.to(device=device, dtype=torch.long)
                lr = learning_rate_at_step(global_step, total_steps_per_epoch, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model(images)
                    loss = loss_function(logits, labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise ValueError("non-finite training logits or loss")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item()) * len(labels)
                total_items += len(labels)
                sample_ids.extend(ids)
                lr_steps.append({"global_step":global_step, "epoch":epoch, "step_in_epoch":step_in_epoch, "learning_rate":lr})
                global_step += 1
            if total_items != len(manifests["train"]) or len(set(sample_ids)) != total_items:
                raise ValueError("training traversal did not visit each record once")
            rng_before_validation = _capture_rng(train_generator, device)
            model.eval()
            val_generator = torch.Generator().manual_seed(seed + epoch)
            val_loader = torch.utils.data.DataLoader(_ManifestDataset(root, manifests["val"]), batch_size=batch_size,
                shuffle=False, generator=val_generator, num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                persistent_workers=False, drop_last=False, worker_init_fn=_init_worker)
            predictions, val_loss, val_items = [], 0.0, 0
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                for images, labels, ids in val_loader:
                    images, labels = images.to(device=device, dtype=torch.float32), labels.to(device=device, dtype=torch.long)
                    logits = model(images)
                    loss = loss_function(logits, labels)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise ValueError("non-finite validation logits or loss")
                    ranked = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :5].cpu().tolist()
                    for true, ranking in zip(labels.cpu().tolist(), ranked):
                        predictions.append({"true_id":true, "pred_id":ranking[0], "top5_ids":ranking})
                    val_loss += float(loss.item()) * len(labels)
                    val_items += len(labels)
            _restore_rng(rng_before_validation, train_generator, device)
            evaluation = evaluate_predictions(predictions, meta["evaluable_ids"]["val"], class_groups)
            score = evaluation["metrics"]["macro_f1"]
            history.append({"epoch":epoch, "train_loss":total_loss/total_items, "val_loss":val_loss/val_items,
                            "val_macro_f1":score, "learning_rate":lr, "steps":total_steps_per_epoch,
                            "duration_seconds":time.monotonic()-started, "sample_ids":sample_ids})
            if best_score is None or score > best_score:
                best_score, best_evaluation = score, evaluation
                best_checkpoint = {"identity":identity, "epoch":epoch, "global_step":global_step,
                                   "model":_cpu_snapshot(model.state_dict()), "optimizer":_cpu_snapshot(optimizer.state_dict()),
                                   "rng":_capture_rng(train_generator, device), "val_macro_f1":score}
            state = {"identity":identity, "epoch":epoch, "global_step":global_step, "model":_cpu_snapshot(model.state_dict()),
                     "optimizer":_cpu_snapshot(optimizer.state_dict()), "rng":_capture_rng(train_generator, device),
                     "best_score":best_score, "best_checkpoint":best_checkpoint, "best_evaluation":best_evaluation,
                     "history":history, "lr_steps":lr_steps}
            _publish_training_state(run_dir, state)
    except Exception as error:
        failures.append({"timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "error":str(error)})
        atomic_write_bytes(metadata_path, canonical_json_bytes({"identity":identity, "condition":condition, "seed":seed,
                                                                 "status":"failed", "start_timestamp":start_time,
                                                                 "failure_history":failures, "resume_history":resumes}))
        raise
    details = {"identity":identity, "condition":condition, "seed":seed, "status":"completed", "start_timestamp":start_time,
               "end_timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "duration_seconds":sum(row["duration_seconds"] for row in history), "epochs_completed":config["epochs"],
               "optimizer_steps":global_step, "selected_epoch":best_checkpoint["epoch"], "batch_size":batch_size,
               "train_count":len(manifests["train"]), "original_count":sum(row["is_synthetic"] == "false" for row in manifests["train"]),
               "synthetic_count":sum(row["is_synthetic"] == "true" for row in manifests["train"]),
               "val_count":len(manifests["val"]), "test_count":meta.get("counts", {}).get("test"),
               "val_evaluable_ids":meta["evaluable_ids"]["val"], "test_evaluable_ids":meta["evaluable_ids"]["test"],
               "runtime_identity":runtime_identity, "failure_history":failures, "resume_history":resumes,
               "initial_sha256":identity["init_sha256"],
               "best_checkpoint_sha256":sha256_file(run_dir / "best_checkpoint.pt")}
    atomic_write_bytes(metadata_path, canonical_json_bytes(details))
    return details
