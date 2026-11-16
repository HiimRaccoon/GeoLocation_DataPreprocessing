"""Deterministic candidate identities, seeds and train-only source ordering."""
import hashlib
from ..artifacts import BENCHMARK_ID, CLEANED_DATASET


def plan_candidate(augmentation_type, class_id, attempt_index, sources):
    if augmentation_type not in {"traditional", "da_fusion"} or type(class_id) is not int or class_id < 0 or type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("invalid candidate identity")
    sources = list(sources)
    if not sources:
        raise ValueError("candidate requires train sources")
    for source in sources:
        if (source.get("split") != "train" or source.get("is_synthetic") not in (False, "false")
            or int(source.get("class_id", -1)) != class_id or not source.get("image_id")
            or not str(source.get("filepath", "")).startswith(CLEANED_DATASET + "/")):
            raise ValueError("candidate sources must be cleaned original train images of the same class")
    sources.sort(key=lambda row: row["filepath"])
    if len({row["image_id"] for row in sources}) != len(sources):
        raise ValueError("duplicate source IDs")
    source = sources[attempt_index % len(sources)]
    seed_bytes = hashlib.sha256(f"aug-v1|314159|{augmentation_type}|{class_id}|{attempt_index}".encode()).digest()
    identifier = hashlib.sha256(f"{BENCHMARK_ID}|{augmentation_type}|{class_id}|{attempt_index}".encode()).hexdigest()
    return {"benchmark_id": BENCHMARK_ID, "synthetic_id": identifier, "augmentation_type": augmentation_type, "class_id": class_id,
            "attempt_index": attempt_index, "generation_seed": int.from_bytes(seed_bytes[:4], "big"),
            "source_image_id": source["image_id"], "source_image_path": source["filepath"], "source_split": "train"}


def next_candidate(group, sources, attempts):
    """Choose the next attempt in order, or stop once the accepted quota is met."""
    group_name = group["group"]
    class_id = int(group["class_id"])
    budget = int(group["required_accepted"])
    if budget < 0 or group_name not in ("many", "medium", "tail"):
        raise ValueError("invalid group budget")
    if group_name == "many":
        if budget != 0 or attempts:
            raise ValueError("many group cannot have augmentation attempts")
        return None
    method = "traditional" if group_name == "medium" else "da_fusion"
    attempts = sorted(attempts, key=lambda row: row["attempt_index"])
    accepted = 0
    for index, attempt in enumerate(attempts):
        expected = plan_candidate(method, class_id, index, sources)
        if any(attempt.get(key) != value for key, value in expected.items()):
            raise ValueError("invalid candidate history")
        status, flag = attempt.get("filter_status"), attempt.get("accepted")
        if status == "pending" and flag is False:
            raise ValueError("review_required")
        if status not in ("accepted", "rejected") or flag is not (status == "accepted"):
            raise ValueError("invalid candidate status")
        accepted += status == "accepted"
        if accepted >= budget and index != len(attempts) - 1:
            raise ValueError("attempts exist after quota was reached")
    if accepted >= budget:
        return None
    if len(attempts) >= 10 * budget:
        raise ValueError("shortage: candidate attempt cap exhausted")
    return plan_candidate(method, class_id, len(attempts), sources)
