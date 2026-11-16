"""Publish native-resolution, HUD-masked GeoGuessr JPEGs and their provenance."""
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .artifacts import (
    BENCHMARK_ID, CLEANED_DATASET, CLEANING_MANIFEST, atomic_write_bytes,
    canonical_json_bytes,
)
from .images import decode_rgb, encode_output_jpeg, pixel_sha256

RAW_DATASET = "data/compressed_dataset"
POLICY = "geoguessr_fixed_hud_neutral_fill_v1"
FILL = (128, 128, 128)
HUD_BOXES = {
    (1536, 662): ((1253, 15, 1505, 73), (1257, 448, 1505, 603)),
    (1030, 703): ((747, 15, 999, 73), (832, 554, 1000, 646)),
}


def _check_cleaning_publication(root):
    cleaned, manifest = root / CLEANED_DATASET, root / CLEANING_MANIFEST
    dataset_present = cleaned.exists() or cleaned.is_symlink()
    manifest_present = manifest.exists() or manifest.is_symlink()
    if dataset_present != manifest_present:
        present, missing = (cleaned, manifest) if dataset_present else (manifest, cleaned)
        raise ValueError(
            f"incomplete cleaning publication: {present.relative_to(root)} exists but "
            f"{missing.relative_to(root)} is missing; neither artifact set is verified. "
            "Preserve the incomplete output and rerun clean in a fresh v1.2 run root"
        )


def clean_dataset(root):
    root = Path(root).resolve()
    raw, cleaned = root / RAW_DATASET, root / CLEANED_DATASET
    manifest = root / CLEANING_MANIFEST
    _check_cleaning_publication(root)
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
                jpeg_bytes = encode_output_jpeg(image)
                decoded = decode_rgb(BytesIO(jpeg_bytes))
                if decoded.size != image.size:
                    raise ValueError(f"cleaned dimensions changed: {relative.as_posix()}")
                cleaned_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()
                cleaned_pixels = pixel_sha256(decoded)
                atomic_write_bytes(output, jpeg_bytes)
                entries.append({
                    "raw_path": (PurePosixPath(RAW_DATASET) / relative.as_posix()).as_posix(),
                    "cleaned_path": (PurePosixPath(CLEANED_DATASET) / relative.as_posix()).as_posix(),
                    "width": image.width, "height": image.height, "bboxes": boxes,
                    "raw_sha256": hashlib.sha256(payload).hexdigest(), "raw_pixel_sha256": raw_pixels,
                    "cleaned_sha256": cleaned_sha256, "cleaned_pixel_sha256": cleaned_pixels,
                })
        result = {"schema_version": 1, "benchmark_id": BENCHMARK_ID, "policy": POLICY, "fill_rgb": FILL,
                  "classes": classes, "images": sorted(entries, key=lambda entry: entry["cleaned_path"])}
        payload = canonical_json_bytes(result)
        # The manifest is the commit marker. An interruption between these two
        # publications is detected by both clean and its consumers.
        destination.rename(cleaned)
        atomic_write_bytes(manifest, payload)
    return result


def read_cleaning_manifest(root):
    """Read a published mapping and check its raw sources before split setup."""
    _check_cleaning_publication(root)
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
    cleaned_prefix = PurePosixPath(CLEANED_DATASET).parts
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
        if (path in entries or len(parts) < 4 or parts[:2] != cleaned_prefix or parts[2] not in classes
                or PurePosixPath(path).as_posix() != path or ".." in parts or "\\" in path or ":" in path
                or raw_path != (PurePosixPath(RAW_DATASET) / PurePosixPath(*parts[2:])).as_posix()
                or boxes != expected_boxes
                or any(not isinstance(digest, str) or len(digest) != 64
                       or any(char not in "0123456789abcdef" for char in digest) for digest in hashes)):
            raise ValueError(f"invalid cleaning provenance: {path}")
        entries[path] = entry
    _verify_raw_provenance(root, classes, entries.values())
    return classes, entries


def _verify_raw_provenance(root, classes, entries):
    raw = root / RAW_DATASET
    if not raw.is_dir():
        raise ValueError(f"raw dataset directory missing: {RAW_DATASET}")
    if sorted(path.name for path in raw.iterdir() if path.is_dir()) != classes:
        raise ValueError("raw class directories differ from cleaning manifest")
    expected = {entry["raw_path"]: entry for entry in entries}
    paths = {
        path.relative_to(root).as_posix(): path
        for name in classes for path in (raw / name).rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    }
    if paths.keys() != expected.keys():
        missing = sorted(expected.keys() - paths.keys())
        extra = sorted(paths.keys() - expected.keys())
        raise ValueError(f"raw dataset membership differs from cleaning manifest: missing {missing[:1]}, extra {extra[:1]}")
    raw_root = raw.resolve()
    cleaned_root = (root / CLEANED_DATASET).resolve()
    if raw_root.is_relative_to(cleaned_root) or cleaned_root.is_relative_to(raw_root):
        raise ValueError("raw and cleaned dataset roots must be separate")
    for name, path in paths.items():
        if path.resolve() != raw_root / path.relative_to(raw):
            raise ValueError(f"raw image must resolve to its own path under {RAW_DATASET}: {name}")
        entry = expected[name]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["raw_sha256"]:
            raise ValueError(f"raw SHA-256 mismatch: {name}")
        try:
            image = decode_rgb(BytesIO(payload))
        except (OSError, ValueError) as error:
            raise ValueError(f"unreadable raw image: {name}: {error}") from error
        if pixel_sha256(image) != entry["raw_pixel_sha256"]:
            raise ValueError(f"raw decoded pixel SHA-256 mismatch: {name}")
        if image.size != (entry["width"], entry["height"]):
            raise ValueError(f"raw dimensions differ from cleaning manifest: {name}")
