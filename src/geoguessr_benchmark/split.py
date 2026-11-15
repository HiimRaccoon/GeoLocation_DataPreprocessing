"""Deterministic real-image splits and frozen class frequency groups."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .artifacts import BENCHMARK_ID, atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, sha256_file

SPLIT_FIELDS = ["image_id", "filepath", "class_name", "class_id", "split", "sha256", "decoded_pixel_sha256", "is_synthetic"]
GROUP_FIELDS = ["class_id", "class_name", "original_train_count", "group", "target", "required_accepted", "accepted_actual"]
EXACT_HASH_FIELDS = ("sha256", "decoded_pixel_sha256")


def allocate_split(n):
    if type(n) is not int or n < 1:
        raise ValueError("class count must be a positive integer")
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 0, 1)
    if n < 10:
        return (n - 2, 1, 1)
    q = max(1, math.floor(0.1 * n + 0.5))
    return (n - 2*q, q, q)


def group_for_count(n):
    if type(n) is not int or n < 1:
        raise ValueError("train count must be a positive integer")
    if n >= 100:
        return ("many", n, 0)
    if n >= 20:
        return ("medium", 100, 100 - n)
    target = min(100, 5 * n)
    return ("tail", target, target - n)


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _duplicate_groups(ordered):
    """Connect rows sharing either exact hash, keeping the original rank of each member."""
    parents = list(range(len(ordered)))
    first = {}

    def representative(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(ordered):
        for key in EXACT_HASH_FIELDS:
            other = first.setdefault((key, row[key]), index)
            parents[representative(index)] = representative(other)
    groups = defaultdict(list)
    for index in range(len(ordered)):
        groups[representative(index)].append(index)
    return groups.values()


def _place_duplicate_groups(groups, original, allocation):
    """Find a quota-compatible placement, trying the original assignments first."""
    preferences = []
    for members in groups:
        votes = Counter(original[index] for index in members)
        preferences.append(sorted(range(len(allocation)), key=lambda split: (-votes[split], split)))

    # Search whole-group placements only. Singletons can fill any remaining slots.
    # Use an explicit stack for classes with many duplicate groups and skip failed states.
    failed = set()
    placement = []
    stack = [(allocation, iter(preferences[0]))]
    while stack:
        index = len(placement)
        remaining, choices = stack[-1]
        size = len(groups[index])
        for split in choices:
            if size > remaining[split]:
                continue
            after = tuple(capacity - (size if slot == split else 0) for slot, capacity in enumerate(remaining))
            next_index = index + 1
            if next_index == len(groups):
                return placement + [split]
            if len(groups[next_index]) <= max(after) and (next_index, after) not in failed:
                placement.append(split)
                stack.append((after, iter(preferences[next_index])))
                break
        else:
            failed.add((index, remaining))
            stack.pop()
            if placement:
                placement.pop()
    raise ValueError(f"cannot place exact duplicate groups within split quotas: {allocation}")


def _class_split_assignments(ordered, allocation):
    original = [split for split, size in enumerate(allocation) for _ in range(size)]
    components = _duplicate_groups(ordered)
    duplicates = [members for members in components if len(members) > 1]
    if all(original[members[0]] == original[members[-1]] for members in duplicates):
        return original

    duplicates.sort(key=lambda members: (-len(members), members[0]))
    placement = _place_duplicate_groups(duplicates, original, allocation)
    assignments = [None] * len(ordered)
    remaining = list(allocation)
    for members, split in zip(duplicates, placement):
        for index in members:
            assignments[index] = split
        remaining[split] -= len(members)

    displaced = []
    for index, preferred in enumerate(original):
        if assignments[index] is not None:
            continue
        if remaining[preferred]:
            assignments[index] = preferred
            remaining[preferred] -= 1
        else:
            displaced.append(index)
    slots = (split for split, size in enumerate(remaining) for _ in range(size))
    for index, split in zip(displaced, slots):
        assignments[index] = split
    return assignments


def _geographic_audit(root, splits):
    """Audit optional location metadata; 25 m is a diagnostic, not a filter."""
    from itertools import product
    path = root / "data/location_metadata.csv"
    if not path.exists():
        return "not supplied; exact-duplicate audit cannot establish geographic independence"
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"filepath", "panorama_id", "latitude", "longitude"}.issubset(reader.fieldnames or []):
            raise ValueError("location metadata requires filepath, panorama_id, latitude, longitude")
        rows = list(reader)
    real = {row["filepath"]: split for split, values in splits.items() for row in values}
    seen, panoramas, points = set(), defaultdict(list), []
    for row in rows:
        filepath = row["filepath"]
        if filepath not in real or filepath in seen:
            raise ValueError("location metadata contains unknown or duplicate filepath")
        seen.add(filepath)
        if row["panorama_id"]:
            panoramas[row["panorama_id"]].append((filepath, real[filepath]))
        lat, lon = row["latitude"].strip(), row["longitude"].strip()
        if bool(lat) != bool(lon):
            raise ValueError("location metadata has incomplete coordinates")
        if lat:
            try:
                latitude, longitude = float(lat), float(lon)
            except ValueError as error:
                raise ValueError("invalid location coordinates") from error
            if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError("invalid location coordinates")
            phi, lam = math.radians(latitude), math.radians(longitude)
            point = (math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi))
            points.append((filepath, real[filepath], point))
    shared = [(key, values) for key,values in sorted(panoramas.items()) if len({item[1] for item in values}) > 1]
    radius, earth = 25.0, 6371008.8
    chord = 2 * math.sin(radius / (2 * earth))
    buckets, nearby_count, nearby_examples = defaultdict(list), 0, []
    for filepath, split, point in points:
        cell = tuple(math.floor((coordinate + 1) / chord) for coordinate in point)
        for delta in product((-1, 0, 1), repeat=3):
            adjacent = tuple(cell[i] + delta[i] for i in range(3))
            for old_path, old_split, old_point in buckets.get(adjacent, []):
                distance_squared = sum((point[i] - old_point[i])**2 for i in range(3))
                if old_split != split and distance_squared <= chord**2:
                    nearby_count += 1
                    if len(nearby_examples) < 100:
                        distance = 2 * earth * math.asin(min(1.0, math.sqrt(distance_squared) / 2))
                        nearby_examples.append({"filepaths":[old_path, filepath], "splits":[old_split, split], "distance_m":distance})
        buckets[cell].append((filepath, split, point))
    return {"status":"audited", "metadata_path":"data/location_metadata.csv", "metadata_rows":len(rows),
            "covered_real_images":len(seen), "total_real_images":len(real), "images_with_coordinates":len(points),
            "radius_m":radius, "diagnostic_only":True, "cross_split_same_panorama_count":len(shared),
            "same_panorama_examples":[{"panorama_id":key,"files":values} for key,values in shared[:100]],
            "cross_split_nearby_pair_count":nearby_count, "nearby_examples":nearby_examples,
            "examples_truncated":len(shared)>100 or nearby_count>100,
            "limitation":"metadata coverage and proximity do not establish geographic independence"}


def build_split(root):
    root = Path(root).resolve()
    if (root / "benchmark_lock.json").exists():
        raise ValueError("benchmark lock already exists; refusing to replace frozen splits")
    inventory_path = root / "reports/dataset_inventory.csv"
    if not (root / "docs/specs.md").is_file():
        raise ValueError("frozen specification is missing")
    rows = _read_csv(inventory_path)
    audit = json.loads((root / "reports/split_audit.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "splits/class_to_idx.json").read_text(encoding="utf-8"))
    if audit.get("unreadable") or audit.get("conflicting_duplicates") or sha256_file(inventory_path) != audit.get("dataset_fingerprint"):
        raise ValueError("inventory is not current or audit failed")
    if len(rows) != audit["baseline"]["images"] or len(mapping) != audit["baseline"]["classes"]:
        raise ValueError("inventory baseline mismatch")
    dataset = root / "data/compressed_dataset"
    actual_classes = {path.name for path in dataset.iterdir() if path.is_dir()}
    actual_paths = {path.relative_to(root).as_posix() for name in actual_classes
                    for path in (dataset / name).rglob("*") if path.is_file() and path.suffix.lower() == ".jpg"}
    if actual_classes != set(mapping) or actual_paths != {row["filepath"] for row in rows}:
        raise ValueError("dataset membership changed since inventory")
    by_class = defaultdict(list)
    hash_classes = {}
    for row in rows:
        name, path = row["class_name"], row["filepath"]
        expected_id = hashlib.sha256(path.encode("utf-8")).hexdigest()
        file = (root / path).resolve()
        if not file.is_relative_to(root / "data/compressed_dataset") or not file.is_file() or sha256_file(file) != row["sha256"]:
            raise ValueError(f"inventory file changed: {path}")
        if mapping.get(name) != int(row["class_id"]) or row["image_id"] != expected_id or row["readable"] != "true":
            raise ValueError(f"invalid inventory row: {path}")
        for key in EXACT_HASH_FIELDS:
            if hash_classes.setdefault((key, row[key]), name) != name:
                raise ValueError(f"exact duplicate images have conflicting labels: {path}")
        by_class[name].append(row)
    if set(by_class) != set(mapping) or len({r["filepath"] for r in rows}) != len(rows):
        raise ValueError("inventory mapping or paths invalid")
    splits = {"train": [], "val": [], "test": []}
    split_names = tuple(splits)
    groups, class_counts = [], {}
    for name, class_id in sorted(mapping.items(), key=lambda pair: pair[1]):
        ordered = sorted(by_class[name], key=lambda r: (hashlib.sha256(("split-v1|42|" + r["filepath"]).encode("utf-8")).digest(), r["filepath"]))
        allocation = allocate_split(len(ordered))
        group, target, budget = group_for_count(allocation[0])
        groups.append({"class_id": class_id, "class_name": name, "original_train_count": allocation[0], "group": group,
                       "target": target, "required_accepted": budget, "accepted_actual": 0})
        class_counts[str(class_id)] = dict(zip(("train", "val", "test"), allocation))
        for row, index in zip(ordered, _class_split_assignments(ordered, allocation)):
            split = split_names[index]
            splits[split].append({key: (split if key == "split" else False if key == "is_synthetic" else row[key]) for key in SPLIT_FIELDS})
    collisions = []
    for key in ("image_id", *EXACT_HASH_FIELDS):
        seen = defaultdict(set)
        for split, values in splits.items():
            for row in values:
                seen[row[key]].add(split)
        for digest, locations in seen.items():
            if len(locations) > 1:
                collisions.append({"hash_type": key, "hash": digest, "splits": sorted(locations)})
    audit["cross_split_duplicates"] = collisions
    audit["split_counts"] = {name: len(values) for name, values in splits.items()}
    audit["split_support"] = {name: len({row["class_id"] for row in values}) for name, values in splits.items()}
    audit["expected_counts"] = {"train":39978, "val":5007, "test":5012}
    audit["expected_support"] = {"train":124, "val":106, "test":111}
    audit["counts_match_reference"] = audit["split_counts"] == audit["expected_counts"]
    audit["support_matches_reference"] = audit["split_support"] == audit["expected_support"]
    try:
        audit["geographic_metadata"] = _geographic_audit(root, splits)
    except ValueError as error:
        audit["geographic_metadata"] = {"status":"invalid", "error":str(error)}
        atomic_write_bytes(root / "reports/split_audit.json", canonical_json_bytes(audit))
        raise
    atomic_write_bytes(root / "reports/split_audit.json", canonical_json_bytes(audit))
    if collisions:
        raise ValueError("cross-split duplicate detected")
    artifacts = {"docs/specs.md": sha256_file(root / "docs/specs.md"),
                 "reports/dataset_inventory.csv": sha256_file(inventory_path),
                 "reports/split_audit.json": sha256_file(root / "reports/split_audit.json"),
                 "splits/class_to_idx.json": sha256_file(root / "splits/class_to_idx.json")}
    if (root / "data/location_metadata.csv").is_file():
        artifacts["data/location_metadata.csv"] = sha256_file(root / "data/location_metadata.csv")
    for name, values in splits.items():
        values.sort(key=lambda r: (int(r["class_id"]), r["filepath"]))
        relative = f"splits/{name}.csv"
        atomic_write_bytes(root / relative, canonical_csv_bytes(values, SPLIT_FIELDS))
        artifacts[relative] = sha256_file(root / relative)
    group_path = root / "splits/class_groups.csv"
    atomic_write_bytes(group_path, canonical_csv_bytes(groups, GROUP_FIELDS))
    artifacts["splits/class_groups.csv"] = sha256_file(group_path)
    projection = canonical_csv_bytes([{k: row[k] for k in GROUP_FIELDS[:-1]} for row in groups], GROUP_FIELDS[:-1])
    immutable_hash = hashlib.sha256(projection).hexdigest()
    metadata = {"schema_version": 1, "split_seed": 42, "ordering": "SHA256(split-v1|42|filepath), filepath",
                "allocation": "n>=10: n-2q/q/q; n=3..9: n-2/1/1; n=2: 1/0/1; n=1: 1/0/0; q=floor(0.1*n+0.5)",
                "dataset_fingerprint": audit["dataset_fingerprint"], "per_class_counts": class_counts,
                "counts": audit["split_counts"], "evaluable_ids": {name: sorted({int(row["class_id"]) for row in values}) for name, values in splits.items()},
                "artifacts": dict(artifacts), "group_immutable_sha256": immutable_hash}
    atomic_write_bytes(root / "splits/split_metadata.json", canonical_json_bytes(metadata))
    artifacts["splits/split_metadata.json"] = sha256_file(root / "splits/split_metadata.json")
    lock = {"schema_version": 1, "benchmark_id": BENCHMARK_ID, "stage": "splits", "dataset_fingerprint": audit["dataset_fingerprint"],
            "artifacts": artifacts, "initial_group_sha256": artifacts["splits/class_groups.csv"],
            "initial_split_metadata_sha256": artifacts["splits/split_metadata.json"], "group_immutable_sha256": immutable_hash}
    atomic_write_bytes(root / "benchmark_lock.json", canonical_json_bytes(lock))
    return lock
