"""Inventory and audit original images before constructing fixed splits."""
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

from .artifacts import CLEANED_DATASET, CLEANING_MANIFEST, atomic_write_bytes, canonical_csv_bytes, canonical_json_bytes, resolve_cleaned_image, sha256_file
from .cleaning import read_cleaning_manifest
from .images import decode_rgb, pixel_sha256

FIELDS = ["image_id", "filepath", "class_name", "class_id", "width", "height", "bytes", "sha256", "decoded_pixel_sha256", "readable"]
DEFAULT_BASELINE = {"images": 49997, "classes": 124}


def build_inventory(root, *, baseline=None):
    root = Path(root).resolve()
    if (root / "benchmark_lock.json").exists():
        raise ValueError("benchmark lock already exists; refusing to overwrite inventory")
    expected_classes, cleaning = read_cleaning_manifest(root)
    dataset = root / CLEANED_DATASET
    if not dataset.is_dir():
        raise ValueError(f"dataset directory missing: {dataset}")
    baseline = DEFAULT_BASELINE if baseline is None else baseline
    classes = sorted(p.name for p in dataset.iterdir() if p.is_dir())
    if classes != expected_classes:
        raise ValueError("cleaned class directories differ from cleaning manifest")
    cleaning_sha = sha256_file(root / CLEANING_MANIFEST)
    mapping = {name: index for index, name in enumerate(classes)}
    mapping_path = root / "splits/class_to_idx.json"
    if mapping_path.exists():
        existing = json.loads(mapping_path.read_text(encoding="utf-8"))
        if existing != mapping:
            raise ValueError("existing class mapping conflicts with dataset")
    rows, errors = [], []
    discovered = set()
    by_hash = {"sha256": defaultdict(list), "decoded_pixel_sha256": defaultdict(list)}
    for name in classes:
        directory = dataset / name
        for path in sorted((p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"), key=lambda p: p.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            discovered.add(relative)
            if relative not in cleaning:
                raise ValueError(f"cleaned dataset membership differs from cleaning manifest: {relative}")
            try:
                path = resolve_cleaned_image(root, relative)
                payload = path.read_bytes()
                image = decode_rgb(BytesIO(payload))
                raw_path = cleaning[relative]["raw_path"]
                row = {"image_id": hashlib.sha256(raw_path.encode("utf-8")).hexdigest(),
                       "filepath": relative, "class_name": name, "class_id": mapping[name],
                       "width": image.width, "height": image.height, "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest(), "decoded_pixel_sha256": pixel_sha256(image), "readable": True}
                rows.append(row)
                for key in by_hash:
                    by_hash[key][row[key]].append(row)
            except (OSError, ValueError) as error:
                errors.append({"filepath": relative, "reason": str(error)})
    if discovered != set(cleaning):
        raise ValueError("cleaned dataset membership differs from cleaning manifest")
    rows.sort(key=lambda row: row["filepath"])
    duplicates, conflicts = [], []
    for key, grouped in by_hash.items():
        for digest, entries in grouped.items():
            if len(entries) > 1:
                duplicate = {"hash_type": key, "hash": digest, "files": [x["filepath"] for x in entries], "classes": sorted({x["class_name"] for x in entries})}
                duplicates.append(duplicate)
                if len(duplicate["classes"]) > 1:
                    conflicts.append(duplicate)
    counts = Counter(row["class_name"] for row in rows)
    resolutions = Counter(f'{r["width"]}x{r["height"]}' for r in rows)
    payload = canonical_csv_bytes(rows, FIELDS)
    fingerprint = hashlib.sha256(payload).hexdigest()
    audit = {"schema_version": 1, "dataset_fingerprint": fingerprint, "baseline": baseline,
             "cleaning_manifest_sha256": cleaning_sha,
             "image_count": len(rows) + len(errors), "class_count": len(classes),
             "class_counts": dict(sorted(counts.items())), "resolutions": dict(sorted(resolutions.items())),
             "reference_checks": {"1536x662": resolutions.get("1536x662", 0) == 49942,
                                  "1030x703": resolutions.get("1030x703", 0) == 55,
                                  "united_states": counts.get("United States", 0) == 12014,
                                  "minimum_class_support": min((counts.get(c, 0) for c in classes), default=0)},
             "unreadable": errors, "duplicates": duplicates, "conflicting_duplicates": conflicts,
             "geographic_metadata": "not supplied; exact-duplicate audit cannot establish geographic independence"}
    atomic_write_bytes(root / "reports/split_audit.json", canonical_json_bytes(audit))
    if errors:
        raise ValueError(f"unreadable image(s): {len(errors)}")
    for row in rows:
        entry = cleaning[row["filepath"]]
        if any(row[field] != entry[key] for field, key in (
            ("sha256", "cleaned_sha256"), ("decoded_pixel_sha256", "cleaned_pixel_sha256"),
            ("width", "width"), ("height", "height"),
        )):
            raise ValueError(f"cleaned image differs from cleaning manifest: {row['filepath']}")
    if len(rows) != baseline["images"] or len(classes) != baseline["classes"]:
        raise ValueError("dataset baseline image or class count mismatch")
    if conflicts:
        raise ValueError("duplicate images have conflicting labels")
    atomic_write_bytes(root / "reports/dataset_inventory.csv", payload)
    atomic_write_bytes(mapping_path, canonical_json_bytes(mapping))
    return audit

