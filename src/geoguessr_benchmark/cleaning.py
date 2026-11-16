"""Publish native-resolution, HUD-masked GeoGuessr JPEGs and their provenance."""
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .artifacts import (
    BENCHMARK_ID, CLEANED_DATASET, CLEANING_MANIFEST, atomic_write_bytes,
    canonical_json_bytes, sha256_file,
)
from .images import decode_rgb, pixel_sha256, write_output_jpeg

RAW_DATASET = "data/compressed_dataset"
POLICY = "geoguessr_fixed_hud_neutral_fill_v1"
FILL = (128, 128, 128)
HUD_BOXES = {
    (1536, 662): ((1253, 15, 1505, 73), (1257, 448, 1505, 603)),
    (1030, 703): ((747, 15, 999, 73), (832, 554, 1000, 646)),
}


def clean_dataset(root):
    root = Path(root).resolve()
    raw, cleaned = root / RAW_DATASET, root / CLEANED_DATASET
    manifest = root / CLEANING_MANIFEST
    existing = next((name for name in (
        CLEANED_DATASET, CLEANING_MANIFEST, "benchmark_lock.json", "environment.lock.txt",
        "reports/dataset_inventory.csv", "reports/split_audit.json", "splits", "data/augmented_dataset",
        "augmentation/manifests", "augmentation/review", "initializations", "runs",
    ) if (root / name).exists() or (root / name).is_symlink()), None)
    if existing is not None:
        raise ValueError(f"existing runtime artifact: {existing}; use a fresh v1.2 run root")
    if not raw.is_dir():
        raise ValueError(f"raw dataset directory missing: {raw}")
    classes = sorted(path.name for path in raw.iterdir() if path.is_dir())
    if not classes:
        raise ValueError("raw dataset has no class directories")
    entries = []
    with TemporaryDirectory(prefix=".cleaning-", dir=cleaned.parent) as staging:
        destination = Path(staging) / "dataset"
        destination.mkdir()
        for name in classes:
            (destination / name).mkdir()
            paths = sorted(path for path in (raw / name).rglob("*") if path.is_file() and path.suffix.lower() == ".jpg")
            if not paths:
                raise ValueError(f"raw class has no JPG images: {name}")
            for path in paths:
                relative = path.relative_to(raw)
                try:
                    payload = path.read_bytes()
                    image = decode_rgb(BytesIO(payload))
                except (OSError, ValueError) as error:
                    raise ValueError(f"unreadable raw image: {relative.as_posix()}: {error}") from error
                if image.size not in HUD_BOXES:
                    raise ValueError(f"unsupported raw resolution {image.width}x{image.height}: {relative.as_posix()}")
                boxes = HUD_BOXES[image.size]
                raw_pixels = pixel_sha256(image)
                for box in boxes:
                    image.paste(FILL, box)
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                write_output_jpeg(output, image)
                entries.append({
                    "raw_path": (PurePosixPath(RAW_DATASET) / relative.as_posix()).as_posix(),
                    "cleaned_path": (PurePosixPath(CLEANED_DATASET) / relative.as_posix()).as_posix(),
                    "width": image.width, "height": image.height, "bboxes": boxes,
                    "raw_sha256": hashlib.sha256(payload).hexdigest(), "raw_pixel_sha256": raw_pixels,
                    "cleaned_sha256": sha256_file(output), "cleaned_pixel_sha256": pixel_sha256(decode_rgb(output)),
                })
        destination.rename(cleaned)
    result = {"schema_version": 1, "benchmark_id": BENCHMARK_ID, "policy": POLICY, "fill_rgb": FILL,
              "classes": classes, "images": sorted(entries, key=lambda entry: entry["cleaned_path"])}
    atomic_write_bytes(manifest, canonical_json_bytes(result))
    return result


def read_cleaning_manifest(root):
    path = root / CLEANING_MANIFEST
    if not path.is_file():
        raise ValueError("cleaning manifest missing; run clean first in a v1.2 run root")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("benchmark_id") != BENCHMARK_ID
            or manifest.get("policy") != POLICY or manifest.get("fill_rgb") != list(FILL)):
        raise ValueError("incompatible cleaning manifest")
    classes, images = manifest.get("classes"), manifest.get("images")
    if (not isinstance(classes, list) or not classes or any(not isinstance(name, str) for name in classes)
            or classes != sorted(set(classes)) or not isinstance(images, list) or not images):
        raise ValueError("invalid classes or images in cleaning manifest")
    entries = {}
    for entry in images:
        try:
            path = entry["cleaned_path"]
            parts = PurePosixPath(path).parts
            expected_boxes = HUD_BOXES.get((entry["width"], entry["height"]))
            boxes = tuple(map(tuple, entry["bboxes"]))
            hashes = [entry[f"{kind}_{field}"] for kind in ("raw", "cleaned") for field in ("sha256", "pixel_sha256")]
            raw_path = entry["raw_path"]
        except (KeyError, TypeError) as error:
            raise ValueError("invalid row in cleaning manifest") from error
        if (path in entries or len(parts) < 4 or parts[:2] != ("data", "cleaned_dataset") or parts[2] not in classes
                or raw_path != (PurePosixPath(RAW_DATASET) / PurePosixPath(*parts[2:])).as_posix()
                or boxes != expected_boxes
                or any(not isinstance(digest, str) or len(digest) != 64
                       or any(char not in "0123456789abcdef" for char in digest) for digest in hashes)):
            raise ValueError(f"invalid cleaning provenance: {path}")
        entries[path] = entry
    return classes, entries
