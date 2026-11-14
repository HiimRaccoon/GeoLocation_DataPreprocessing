"""Metrics on the full 124-class prediction space."""
import math

NUM_CLASSES = 124


def rank_logits(logits):
    values = [float(value) for value in logits]
    if len(values) != NUM_CLASSES or not all(math.isfinite(value) for value in values):
        raise ValueError("expected 124 finite logits")
    return sorted(range(NUM_CLASSES), key=lambda class_id: (-values[class_id], class_id))


def evaluate_predictions(rows, evaluable_ids, class_groups):
    rows = list(rows)
    if not rows:
        raise ValueError("empty prediction split")
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    top5_correct = 0
    for row in rows:
        true = int(row["true_id"])
        predicted = int(row["pred_id"])
        top5 = [int(x) for x in row["top5_ids"]]
        if not (0 <= true < NUM_CLASSES and 0 <= predicted < NUM_CLASSES) or not top5 or predicted != top5[0] or len(top5) != 5 or len(set(top5)) != len(top5) or any(x < 0 or x >= NUM_CLASSES for x in top5):
            raise ValueError("invalid prediction row")
        confusion[true][predicted] += 1
        top5_correct += true in top5
    per_class = []
    for class_id in range(NUM_CLASSES):
        tp = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        fn = support - tp
        fp = sum(row[class_id] for row in confusion) - tp
        if support:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        else:
            precision = recall = f1 = None
        per_class.append({"class_id": class_id, "support": support, "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1, "evaluable": bool(support),
                          "status": "evaluable" if support else "not evaluable on this split"})
    observed = [r["class_id"] for r in per_class if r["support"]]
    if sorted(evaluable_ids) != observed:
        raise ValueError("evaluable IDs do not match split support")
    selected = [per_class[class_id] for class_id in observed]
    mean = lambda items, key: sum(item[key] for item in items) / len(items) if items else None
    metrics = {"top1_accuracy": sum(confusion[i][i] for i in range(NUM_CLASSES)) / len(rows),
               "top5_accuracy": top5_correct / len(rows), "macro_f1": mean(selected, "f1"),
               "macro_precision": mean(selected, "precision"), "macro_recall": mean(selected, "recall"),
               "weighted_f1": sum(item["support"] * item["f1"] for item in selected) / len(rows),
               "num_classifier_classes": NUM_CLASSES, "evaluable_ids": observed}
    groups = {}
    for name, ids in class_groups.items():
        if any(type(i) is not int or i < 0 or i >= NUM_CLASSES for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("invalid group class IDs")
        subset = [per_class[i] for i in ids if per_class[i]["support"]]
        groups[name] = {"macro_f1": mean(subset, "f1"), "macro_recall": mean(subset, "recall"),
                        "evaluable_classes": len(subset), "support": sum(x["support"] for x in subset)}
    return {"metrics": metrics, "per_class": per_class, "groups": groups, "confusion": confusion}

