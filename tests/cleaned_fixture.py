"""Bind hand-crafted cleaned JPEGs for inventory and split unit tests."""
from geoguessr_benchmark.artifacts import (
    BENCHMARK_ID, CLEANED_DATASET, CLEANING_MANIFEST, atomic_write_bytes,
    canonical_json_bytes, sha256_file,
)
from geoguessr_benchmark.cleaning import FILL, HUD_BOXES, POLICY, RAW_DATASET
from geoguessr_benchmark.images import decode_rgb, pixel_sha256


def seal_cleaned_fixture(root):
    dataset = root / CLEANED_DATASET
    classes = sorted(path.name for path in dataset.iterdir() if path.is_dir())
    entries = []
    for path in sorted(dataset.rglob("*.jpg")):
        image = decode_rgb(path)
        digest, pixels = sha256_file(path), pixel_sha256(image)
        entries.append({"raw_path": f"{RAW_DATASET}/{path.relative_to(dataset).as_posix()}",
                        "cleaned_path": path.relative_to(root).as_posix(),
                        "width": image.width, "height": image.height, "bboxes": HUD_BOXES[image.size],
                        "raw_sha256": digest, "raw_pixel_sha256": pixels,
                        "cleaned_sha256": digest, "cleaned_pixel_sha256": pixels})
    atomic_write_bytes(root / CLEANING_MANIFEST, canonical_json_bytes({
        "schema_version": 1, "benchmark_id": BENCHMARK_ID, "policy": POLICY,
        "fill_rgb": FILL, "classes": classes, "images": entries,
    }))
