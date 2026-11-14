"""Paired initialization and experiment orchestration primitives."""
import io
from pathlib import Path

from .artifacts import _resolve_artifact, atomic_write_bytes, canonical_json_bytes, sha256_file, verify_stage


def initialize_pairs(root):
    """Freeze one shared pretrained initial state for each training seed."""
    import torch
    from .classifier import build_model
    from .config import load_config
    from .preflight import _configure_determinism, verify_runtime

    root = Path(root).resolve()
    lock = verify_stage(root, "augmented")
    if lock["stage"] != "augmented":
        return verify_stage(root, "initialized")
    _configure_determinism()
    verify_runtime(root, lock)
    config = load_config(root)["benchmark"]
    if config.get("batch_size_effective") not in (128, 64, 32, 16, 8, 4, 2):
        raise ValueError("preflight has not frozen the effective batch size")
    try:
        entry = lock["model_snapshot"]["files"]["efficientnet_weights"]
        name, expected = entry["path"], entry["sha256"]
        weights_path = _resolve_artifact(root, name)
    except (KeyError, TypeError) as error:
        raise ValueError("pretrained classifier snapshot missing from lock") from error
    if lock["artifacts"].get(name) != expected or not weights_path.is_file() or sha256_file(weights_path) != expected:
        raise ValueError("pretrained classifier snapshot checksum mismatch")
    new_artifacts = dict(lock["artifacts"])
    for seed in (42, 123, 2026):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = build_model(weights_path)
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        relative = f"initializations/seed_{seed}.pt"
        path = root / relative
        if path.exists():
            try:
                previous = torch.load(path, map_location="cpu", weights_only=True)
                if set(previous) != set(state) or any(not torch.equal(previous[key], state[key]) for key in state):
                    raise ValueError("initialization differs from deterministic state")
            except (RuntimeError, OSError, TypeError, KeyError) as error:
                raise ValueError("existing initialization cannot be recovered") from error
        else:
            buffer = io.BytesIO()
            torch.save(state, buffer)
            atomic_write_bytes(path, buffer.getvalue())
        new_artifacts[relative] = sha256_file(path)
    new_lock = dict(lock, stage="initialized", artifacts=new_artifacts,
                    initial_state_sha256={str(seed): new_artifacts[f"initializations/seed_{seed}.pt"] for seed in (42, 123, 2026)})
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(new_lock))
    return new_lock


def freeze_checkpoints(root):
    """Lock six complete runs and their earliest best-validation checkpoints."""
    import csv
    import json
    import math
    import torch
    import yaml
    from .metrics import evaluate_predictions
    root = Path(root).resolve()
    lock = verify_stage(root, "initialized")
    if lock["stage"] != "initialized":
        return verify_stage(root, "trained")
    with (root / "configs/benchmark.yaml").open(encoding="utf-8") as stream:
        fixed_config = yaml.safe_load(stream)
    if not isinstance(fixed_config, dict) or fixed_config.get("epochs") != 20 or type(fixed_config.get("batch_size_effective")) is not int or fixed_config["batch_size_effective"] < 1:
        raise ValueError("training configuration does not lock 20 epochs and batch size")
    train_counts = {}
    for condition, path in (("original", "splits/train.csv"), ("augmented", "splits/train_augmented.csv")):
        with (root / path).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or any(row.get("is_synthetic") not in ("true", "false") for row in rows) or len({r["image_id"] for r in rows}) != len(rows):
            raise ValueError("invalid training manifest")
        original = sum(row["is_synthetic"] == "false" for row in rows)
        train_counts[condition] = (len(rows), original, len(rows) - original)
    if train_counts["original"][2] or train_counts["original"][1] != train_counts["augmented"][1]:
        raise ValueError("training conditions have inconsistent original counts")
    with (root / "splits/val.csv").open(encoding="utf-8", newline="") as stream:
        val_manifest = list(csv.DictReader(stream))
    split_meta = json.loads((root / "splits/split_metadata.json").read_text(encoding="utf-8"))
    val_support = [0] * 124
    for row in val_manifest:
        class_id = int(row["class_id"])
        if not 0 <= class_id < 124:
            raise ValueError("invalid validation class ID")
        val_support[class_id] += 1
    if [i for i,n in enumerate(val_support) if n] != split_meta["evaluable_ids"]["val"]:
        raise ValueError("validation support differs from split metadata")
    groups = {name:[] for name in ("many", "medium", "tail")}
    with (root / "splits/class_groups.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["group"]].append(int(row["class_id"]))
    artifacts = dict(lock["artifacts"])
    required = ("config.yaml", "metrics.json", "group_metrics.json", "history.csv", "lr_steps.csv", "best_checkpoint.pt", "last_checkpoint.pt", "confusion_matrix_val.csv", "per_class_metrics_val.csv", "run_metadata.json")
    for seed in (42, 123, 2026):
        for condition in ("original", "augmented"):
            base = root / "runs" / condition / f"seed_{seed}"
            if any(not (base / name).is_file() for name in required):
                raise ValueError(f"incomplete run: {condition}/{seed}")
            with (base / "run_metadata.json").open(encoding="utf-8") as stream:
                info = json.load(stream)
            train_path = "splits/train.csv" if condition == "original" else "splits/train_augmented.csv"
            identity = {"condition":condition, "seed":seed, "train_sha256":artifacts[train_path],
                        "val_sha256":artifacts["splits/val.csv"], "init_sha256":artifacts[f"initializations/seed_{seed}.pt"],
                        "config_sha256":artifacts["configs/benchmark.yaml"], "environment_sha256":artifacts["environment.lock.txt"],
                        "group_sha256":artifacts["splits/class_groups.csv"], "split_metadata_sha256":artifacts["splits/split_metadata.json"]}
            if (info.get("identity") != identity or info.get("condition") != condition or info.get("seed") != seed or
                info.get("status") != "completed" or info.get("epochs_completed") != 20):
                raise ValueError("run identity or completed epochs are invalid")
            if tuple(info.get(key) for key in ("train_count", "original_count", "synthetic_count")) != train_counts[condition]:
                raise ValueError("run train counts do not match manifest")
            batch = fixed_config["batch_size_effective"]
            steps = 20 * math.ceil(train_counts[condition][0] / batch)
            if info.get("batch_size") != batch or info.get("optimizer_steps") != steps:
                raise ValueError("run batch or step count is inconsistent")
            with (base / "config.yaml").open(encoding="utf-8") as stream:
                if yaml.safe_load(stream) != fixed_config:
                    raise ValueError("run configuration differs from frozen configuration")
            with (base / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            with (base / "lr_steps.csv").open(encoding="utf-8", newline="") as stream:
                lr_rows = list(csv.DictReader(stream))
            try:
                if [int(row["epoch"]) for row in history] != list(range(1,21)) or [int(row["global_step"]) for row in lr_rows] != list(range(steps)):
                    raise ValueError("run history or LR steps are incomplete")
                scores = [float(row["val_macro_f1"]) for row in history]
                if not all(math.isfinite(value) for value in scores):
                    raise ValueError("non-finite validation score")
                selected = scores.index(max(scores)) + 1
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid training history") from error
            if info.get("selected_epoch") != selected or info.get("best_checkpoint_sha256") != sha256_file(base / "best_checkpoint.pt"):
                raise ValueError("best checkpoint does not match validation history")
            try:
                best = torch.load(base / "best_checkpoint.pt", map_location="cpu", weights_only=True)
                last = torch.load(base / "last_checkpoint.pt", map_location="cpu", weights_only=True)
                initial = torch.load(root / f"initializations/seed_{seed}.pt", map_location="cpu", weights_only=True)
                metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
                group_metrics = json.loads((base / "group_metrics.json").read_text(encoding="utf-8"))
                if (best.get("identity") != identity or best.get("epoch") != selected or best.get("val_macro_f1") != scores[selected-1] or
                    last.get("identity") != identity or last.get("epoch") != 20 or last.get("global_step") != steps or
                    metrics["validation_at_best"]["macro_f1"] != scores[selected-1]):
                    raise ValueError("checkpoint state is inconsistent")
                if not isinstance(initial, dict) or not initial:
                    raise ValueError("initial state is empty")
                for checkpoint in (best, last):
                    model_state = checkpoint.get("model")
                    if not isinstance(model_state, dict) or set(model_state) != set(initial):
                        raise ValueError("checkpoint model keys differ from paired initial state")
                    for key, original in initial.items():
                        value = model_state[key]
                        if (not isinstance(original, torch.Tensor) or not isinstance(value, torch.Tensor) or
                            value.shape != original.shape or value.dtype != original.dtype or
                            (value.is_floating_point() and not torch.isfinite(value).all())):
                            raise ValueError("checkpoint model tensor is invalid")
            except (OSError, KeyError, TypeError, RuntimeError) as error:
                raise ValueError("invalid checkpoint or metrics") from error
            with (base / "per_class_metrics_val.csv").open(encoding="utf-8", newline="") as stream:
                per_class = list(csv.DictReader(stream))
            with (base / "confusion_matrix_val.csv").open(encoding="utf-8", newline="") as stream:
                matrix = list(csv.DictReader(stream))
            if len(per_class) != 124 or [int(row["class_id"]) for row in per_class] != list(range(124)) or len(matrix) != 124 or any(len(row) != 125 for row in matrix):
                raise ValueError("validation class or confusion artifacts are incomplete")
            try:
                if [int(row["true_class_id"]) for row in matrix] != list(range(124)):
                    raise ValueError("invalid confusion row IDs")
                counts = [[int(row[str(j)]) for j in range(124)] for row in matrix]
                if any(value < 0 for row in counts for value in row) or [sum(row) for row in counts] != val_support:
                    raise ValueError("confusion support differs from validation manifest")
                pseudo = []
                for true, row in enumerate(counts):
                    for pred, count in enumerate(row):
                        ranking = [pred] + [i for i in range(5) if i != pred][:4]
                        pseudo.extend([{"true_id":true,"pred_id":pred,"top5_ids":ranking}] * count)
                computed = evaluate_predictions(pseudo, split_meta["evaluable_ids"]["val"], groups)
                for name in ("top1_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"):
                    if not math.isclose(float(metrics["validation_at_best"][name]), computed["metrics"][name], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("validation metrics do not match confusion matrix")
                top5 = float(metrics["validation_at_best"]["top5_accuracy"])
                if not math.isfinite(top5) or not 0 <= top5 <= 1:
                    raise ValueError("invalid validation top-5 accuracy")
                for actual, expected in zip(per_class, computed["per_class"]):
                    if any(int(actual[key]) != expected[key] for key in ("class_id","support","tp","fp","fn")):
                        raise ValueError("per-class counts differ from confusion matrix")
                    if actual["evaluable"] != str(expected["evaluable"]).lower() or actual["status"] != expected["status"]:
                        raise ValueError("invalid per-class evaluability")
                    for key in ("precision","recall","f1"):
                        if expected[key] is None:
                            if actual[key] != "":
                                raise ValueError("unsupported class metric must be null")
                        elif not math.isclose(float(actual[key]), expected[key], rel_tol=1e-12, abs_tol=1e-12):
                            raise ValueError("per-class metric differs from confusion matrix")
                for name, expected in computed["groups"].items():
                    actual = group_metrics["validation_at_best"][name]
                    if actual["support"] != expected["support"] or actual["evaluable_classes"] != expected["evaluable_classes"]:
                        raise ValueError("group support differs from confusion matrix")
                    for key in ("macro_f1", "macro_recall"):
                        if expected[key] is None:
                            if actual[key] is not None: raise ValueError("empty group metric must be null")
                        elif not math.isclose(float(actual[key]), expected[key], rel_tol=1e-12, abs_tol=1e-12):
                            raise ValueError("group metric differs from confusion matrix")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid validation artifacts") from error
            for name in required:
                relative = (base / name).relative_to(root).as_posix()
                artifacts[relative] = sha256_file(base / name)
    result = dict(lock, stage="trained", artifacts=artifacts)
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(result))
    return result




def train_all(root):
    """Launch the fixed six-run matrix in isolated Python processes."""
    import os
    import subprocess
    import sys
    root = Path(root).resolve()
    sequence = [(condition, seed) for seed in (42, 123, 2026) for condition in ("original", "augmented")]
    for condition, seed in sequence:
        current = verify_stage(root, "initialized")
        if current["stage"] != "initialized":
            raise ValueError("the training matrix has already been frozen")
        variables = dict(os.environ)
        variables.update({"PYTHONHASHSEED":str(seed), "CUBLAS_WORKSPACE_CONFIG":":4096:8"})
        variables["PYTHONPATH"] = os.pathsep.join((str(root / "src"), variables.get("PYTHONPATH", "")))
        argv = [sys.executable, "-m", "geoguessr_benchmark", "--root", str(root), "_train-run", "--condition", condition, "--seed", str(seed)]
        completed = subprocess.run(argv, env=variables, cwd=root)
        if completed.returncode:
            raise ValueError(f"run failed: {condition}/seed_{seed}; retry resumes this run")
    return True
