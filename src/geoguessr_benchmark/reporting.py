"""Seed-paired aggregation helpers for final reporting."""
import math
import statistics


def aggregate_paired(original, augmented):
    def by_seed(rows):
        result = {}
        for row in rows:
            seed = row["seed"]
            if seed in result:
                raise ValueError("duplicate seed")
            value = row["value"]
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError("invalid metric value")
            result[seed] = value
        if set(result) != {42, 123, 2026}:
            raise ValueError("exactly the three benchmark seeds are required")
        return result
    a, b = by_seed(original), by_seed(augmented)
    if any((a[s] is None) != (b[s] is None) for s in a):
        raise ValueError("paired metrics have inconsistent support")
    if any(value is None for value in a.values()):
        if not all(value is None for value in a.values()):
            raise ValueError("group support differs across seeds")
        empty = {"mean": None, "std": None}
        return {"original": empty.copy(), "augmented": empty.copy(), "delta": empty.copy(), "delta_pp": empty.copy()}
    def summarize(values):
        return {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    seeds = (42, 123, 2026)
    delta = [b[s] - a[s] for s in seeds]
    return {"original": summarize([a[s] for s in seeds]), "augmented": summarize([b[s] for s in seeds]),
            "delta": summarize(delta), "delta_pp": summarize([100 * x for x in delta])}


def build_report(root):
    """Recompute and publish the final report from six locked prediction files."""
    import csv
    import json
    from pathlib import Path
    from .artifacts import atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, sha256_file, verify_stage
    from .evaluate import load_test_predictions
    from .metrics import evaluate_predictions
    from .preflight import verify_source

    root = Path(root).resolve()
    lock = verify_stage(root, "tested")
    verify_source(root)
    with (root / "splits/class_groups.csv").open(encoding="utf-8", newline="") as file:
        class_rows = list(csv.DictReader(file))
    try:
        ids = [int(row["class_id"]) for row in class_rows]
        if sorted(ids) != list(range(124)) or any(row["group"] not in ("many", "medium", "tail") for row in class_rows):
            raise ValueError("group mapping must cover exactly 124 classes")
        groups = {name: [int(row["class_id"]) for row in class_rows if row["group"] == name] for name in ("many", "medium", "tail")}
    except (KeyError, TypeError) as error:
        raise ValueError("invalid class groups") from error
    metadata = json.loads((root / "splits/split_metadata.json").read_text(encoding="utf-8"))
    test_ids = metadata["evaluable_ids"]["test"]
    qc = json.loads((root / "augmentation/manifests/qc_summary.json").read_text(encoding="utf-8"))
    manifests = {}
    for condition, path in (("original", "splits/train.csv"), ("augmented", "splits/train_augmented.csv")):
        with (root / path).open(encoding="utf-8", newline="") as file:
            entries = list(csv.DictReader(file))
        if not entries or any(row.get("is_synthetic") not in ("true", "false") for row in entries):
            raise ValueError("invalid training manifest counts")
        if len({row["image_id"] for row in entries}) != len(entries):
            raise ValueError("duplicate training manifest IDs")
        original = sum(row["is_synthetic"] == "false" for row in entries)
        manifests[condition] = {"train_count": len(entries), "original_count": original, "synthetic_count": len(entries) - original}
    if manifests["original"]["synthetic_count"] or manifests["original"]["original_count"] != manifests["augmented"]["original_count"]:
        raise ValueError("training manifest counts do not match the benchmark conditions")
    metric_names = ("top1_accuracy", "top5_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")
    results, run_meta, raw = {}, {}, []
    for condition in ("original", "augmented"):
        for seed in (42, 123, 2026):
            key = (condition, seed)
            rows = load_test_predictions(root, condition, seed)
            results[key] = evaluate_predictions(rows, test_ids, groups)
            path = f"runs/{condition}/seed_{seed}/run_metadata.json"
            if path not in lock["artifacts"]:
                raise ValueError(f"run metadata is not locked: {path}")
            details = json.loads((root / path).read_text(encoding="utf-8"))
            if details.get("condition") != condition or details.get("seed") != seed or details.get("epochs_completed") != 20:
                raise ValueError(f"invalid completed run metadata: {path}")
            for field in ("duration_seconds", "optimizer_steps", "selected_epoch", "batch_size", "train_count", "original_count", "synthetic_count"):
                value = details.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"missing or invalid run field: {field}")
            if not 1 <= details["selected_epoch"] <= 20 or details["train_count"] != details["original_count"] + details["synthetic_count"]:
                raise ValueError("inconsistent run counts or selected epoch")
            if any(details[field] != value for field, value in manifests[condition].items()):
                raise ValueError("run counts differ from locked training manifest")
            if details["batch_size"] <= 0 or details["optimizer_steps"] != 20 * math.ceil(details["train_count"] / details["batch_size"]):
                raise ValueError("run step counts do not match fixed epochs and batch size")
            run_meta[key] = details
            values = results[key]["metrics"]
            raw.append({"condition": condition, "seed": seed, **{name: values[name] for name in metric_names},
                        "selected_epoch": details["selected_epoch"], "optimizer_steps": details["optimizer_steps"],
                        "train_count": details["train_count"], "image_exposures": details["train_count"] * 20,
                        "duration_seconds": details["duration_seconds"]})
    if len({details["batch_size"] for details in run_meta.values()}) != 1:
        raise ValueError("batch size differs across paired runs")
    def paired(getter):
        return aggregate_paired([{"seed": seed, "value": getter(results[("original", seed)])} for seed in (42,123,2026)],
                                [{"seed": seed, "value": getter(results[("augmented", seed)])} for seed in (42,123,2026)])
    aggregate = {"schema_version": 1, "num_classifier_classes": 124,
                 "validation_evaluable_ids": metadata["evaluable_ids"]["val"], "test_evaluable_ids": test_ids,
                 "metrics": {name: paired(lambda result, name=name: result["metrics"][name]) for name in metric_names},
                 "groups": {}, "per_class": []}
    group_rows = []
    for name in ("many", "medium", "tail"):
        sample = results[("original", 42)]["groups"][name]
        for result in results.values():
            if result["groups"][name]["support"] != sample["support"] or result["groups"][name]["evaluable_classes"] != sample["evaluable_classes"]:
                raise ValueError("group support differs across runs")
        f1 = paired(lambda result: result["groups"][name]["macro_f1"])
        recall = paired(lambda result: result["groups"][name]["macro_recall"])
        aggregate["groups"][name] = {"support":sample["support"], "evaluable_classes":sample["evaluable_classes"], "macro_f1":f1, "macro_recall":recall}
        group_rows.append({"group":name, "evaluable_classes":sample["evaluable_classes"], "support":sample["support"],
                           "original_f1_mean":f1["original"]["mean"], "original_f1_std":f1["original"]["std"],
                           "augmented_f1_mean":f1["augmented"]["mean"], "augmented_f1_std":f1["augmented"]["std"],
                           "delta_f1_pp_mean":f1["delta_pp"]["mean"], "delta_f1_pp_std":f1["delta_pp"]["std"],
                           "original_recall_mean":recall["original"]["mean"], "original_recall_std":recall["original"]["std"],
                           "augmented_recall_mean":recall["augmented"]["mean"], "augmented_recall_std":recall["augmented"]["std"]})
    names = {int(row["class_id"]): row for row in class_rows}
    for class_id in range(124):
        per_class = results[("original", 42)]["per_class"][class_id]
        for result in results.values():
            if result["per_class"][class_id]["support"] != per_class["support"]:
                raise ValueError("class support differs across runs")
        aggregate["per_class"].append({"class_id":class_id, "class_name":names[class_id]["class_name"],
                                        "group":names[class_id]["group"], "support":per_class["support"],
                                        "f1":paired(lambda result, i=class_id: result["per_class"][i]["f1"]),
                                        "recall":paired(lambda result, i=class_id: result["per_class"][i]["recall"])})
    def fmt(value, scale=100):
        return "không đánh giá được" if value["mean"] is None else f'{scale * value["mean"]:.2f} ± {scale * value["std"]:.2f}'
    lines = ["# Báo cáo benchmark GeoGuessr-50k", "", "Kết quả test từ best checkpoint được chọn bằng validation. Không làm tròn trước khi tổng hợp.", "",
             f"Số lớp đầu ra: 124; lớp có support validation: {len(metadata['evaluable_ids']['val'])}; test: {len(test_ids)}.", "",
             "A (Original) dùng E0. B_DAFusion dùng E0 + E1 Traditional + E2 DA-Fusion.", "",
             "## Chỉ số chính", "", "| Chỉ số | A (%) | B_DAFusion (%) | B_DAFusion − A (pp) |", "|---|---:|---:|---:|"]
    labels = {"top1_accuracy":"Top-1", "top5_accuracy":"Top-5", "macro_f1":"Macro-F1", "weighted_f1":"Weighted-F1", "macro_precision":"Macro Precision", "macro_recall":"Macro Recall"}
    for name in metric_names:
        item = aggregate["metrics"][name]
        lines.append(f'| {labels[name]} | {fmt(item["original"])} | {fmt(item["augmented"])} | {fmt(item["delta_pp"],1)} |')
    lines += ["", "## Nhóm theo số ảnh train gốc", "", "| Nhóm | Lớp có support | Support | Original F1 (%) | Augmented F1 (%) | Delta F1 (pp) | Original recall (%) | Augmented recall (%) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("many", "medium", "tail"):
        item=aggregate["groups"][name]; f1=item["macro_f1"]; recall=item["macro_recall"]
        lines.append(f'| {name} | {item["evaluable_classes"]} | {item["support"]} | {fmt(f1["original"])} | {fmt(f1["augmented"])} | {fmt(f1["delta_pp"],1)} | {fmt(recall["original"])} | {fmt(recall["augmented"])} |')
    lines += ["", "## Từng lớp", "", "| ID | Lớp | Nhóm | Support | Original F1 (%) | Augmented F1 (%) | Delta F1 (pp) | Original recall (%) | Augmented recall (%) | Delta recall (pp) |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate["per_class"]:
        f1=row["f1"]; recall=row["recall"]; safe_name=row["class_name"].replace("|", "\\|")
        lines.append(f'| {row["class_id"]} | {safe_name} | {row["group"]} | {row["support"]} | {fmt(f1["original"])} | {fmt(f1["augmented"])} | {fmt(f1["delta_pp"],1)} | {fmt(recall["original"])} | {fmt(recall["augmented"])} | {fmt(recall["delta_pp"],1)} |')
    lines += ["", "## Kết quả từng seed", "", "| Condition | Seed | Top-1 | Top-5 | Macro-F1 | Weighted-F1 | Macro precision | Macro recall | Best epoch | Steps | Exposures | Giây |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in raw:
        vals = " | ".join(f'{100*row[name]:.2f}' for name in metric_names)
        lines.append(f'| {row["condition"]} | {row["seed"]} | {vals} | {row["selected_epoch"]} | {row["optimizer_steps"]} | {row["image_exposures"]} | {row["duration_seconds"]:.2f} |')
    lines += ["", "## QC và thông tin chạy", "", "```json", json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2), "```", "", "```json", json.dumps([run_meta[key] for key in sorted(run_meta)], ensure_ascii=False, sort_keys=True, indent=2), "```", "",
              "## Giới hạn diễn giải", "", "Hai điều kiện dùng cùng 20 epoch nhưng tập augmented có thể tạo thêm bước tối ưu và lượt xem ảnh. Đây là so sánh toàn bộ chính sách, không tách tác động riêng của từng phương pháp và không phải equal-compute.",
              "Độ lệch chuẩn mẫu phản ánh ba seed trên một split cố định; không chứng minh ý nghĩa thống kê hay tính độc lập địa lý. Các lớp support thấp hoặc không có support cần được diễn giải thận trọng. Review diffusion có tính chủ quan; còn rủi ro near-copy, domain gap, trùng dữ liệu tiền huấn luyện và leakage theo vị trí chưa được kiểm chứng.", ""]
    raw_fields = ["condition", "seed", *metric_names, "selected_epoch", "optimizer_steps", "train_count", "image_exposures", "duration_seconds"]
    outputs = {"reports/per_seed_metrics.csv":canonical_csv_bytes(raw, raw_fields),
               "reports/aggregate_metrics.json":canonical_json_bytes(aggregate),
               "reports/group_comparison.csv":canonical_csv_bytes(group_rows, list(group_rows[0])),
               "reports/final_report.md":"\n".join(lines).encode("utf-8")}
    updated = dict(lock["artifacts"])
    for name, payload in outputs.items():
        atomic_write_bytes(root / name, payload)
        updated[name] = sha256_file(root / name)
    new_lock = dict(lock, stage="reported", artifacts=updated)
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(new_lock))
    return new_lock
