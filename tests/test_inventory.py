import csv
import hashlib
import json

from PIL import Image
import pytest

from geoguessr_benchmark.inventory import build_inventory
from cleaned_fixture import seal_cleaned_fixture


def image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1030, 703), color).save(path, quality=95)


def test_inventory_is_canonical_for_unicode_paths_and_checks_baseline(tmp_path):
    image(tmp_path / "data/cleaned_dataset/Việt Nam/b.jpg", (30, 40, 60))
    image(tmp_path / "data/cleaned_dataset/A/a.jpg", (190, 40, 60))
    seal_cleaned_fixture(tmp_path)
    baseline = {"images": 2, "classes": 2}
    build_inventory(tmp_path, baseline=baseline)
    mapping = json.loads((tmp_path / "splits/class_to_idx.json").read_text(encoding="utf-8"))
    assert mapping == {"A": 0, "Việt Nam": 1}
    with (tmp_path / "reports/dataset_inventory.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [r["class_id"] for r in rows] == ["0", "1"]
    assert rows[1]["filepath"] == "data/cleaned_dataset/Việt Nam/b.jpg"
    assert rows[1]["image_id"] == hashlib.sha256("data/compressed_dataset/Việt Nam/b.jpg".encode()).hexdigest()
    with pytest.raises(ValueError, match="baseline"):
        build_inventory(tmp_path, baseline={"images": 3, "classes": 2})
    (tmp_path / "splits/class_to_idx.json").write_text(json.dumps({"A": 1, "Việt Nam": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        build_inventory(tmp_path, baseline=baseline)


def test_conflicting_duplicates_and_unreadable_images_fail_closed(tmp_path):
    source = tmp_path / "data/cleaned_dataset/A/a.jpg"
    image(source, (10, 20, 30))
    target = tmp_path / "data/cleaned_dataset/B/b.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    seal_cleaned_fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        build_inventory(tmp_path, baseline={"images": 2, "classes": 2})
    audit = json.loads((tmp_path / "reports/split_audit.json").read_text(encoding="utf-8"))
    assert audit["conflicting_duplicates"]
    target.write_bytes(b"broken jpeg")
    with pytest.raises(ValueError, match="unreadable"):
        build_inventory(tmp_path, baseline={"images": 2, "classes": 2})


def test_distinct_jpeg_bytes_with_same_pixels_detect_label_conflict(tmp_path):
    source = tmp_path / "data/cleaned_dataset/A/a.jpg"
    target = tmp_path / "data/cleaned_dataset/B/b.jpg"
    image(source,(40,80,120))
    target.parent.mkdir(parents=True)
    # Appending bytes after the JPEG end marker preserves decoded RGB.
    target.write_bytes(source.read_bytes()+b"metadata-after-eoi")
    assert source.read_bytes()!=target.read_bytes()
    seal_cleaned_fixture(tmp_path)
    with pytest.raises(ValueError,match="duplicate"):
        build_inventory(tmp_path,baseline={"images":2,"classes":2})
    audit=json.loads((tmp_path/"reports/split_audit.json").read_text(encoding="utf-8"))
    assert any(row["hash_type"]=="decoded_pixel_sha256" for row in audit["conflicting_duplicates"])


@pytest.mark.parametrize("field,value,error", [
    ("raw_pixel_sha256", "0" * 64, "raw decoded pixel SHA-256 mismatch"),
    ("raw_path", "data/compressed_dataset/A/../../a.jpg", "invalid cleaning provenance"),
    ("raw_path", "data/compressed_dataset/A/a.jpg", "invalid cleaning provenance"),
])
def test_inventory_rejects_unverified_raw_pixel_hash_or_mapping(tmp_path, field, value, error):
    image(tmp_path / "data/cleaned_dataset/A/a.jpg", (30, 40, 60))
    image(tmp_path / "data/cleaned_dataset/A/b.jpg", (80, 160, 90))
    seal_cleaned_fixture(tmp_path)
    manifest_path = tmp_path / "reports/cleaning_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["images"][1][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        build_inventory(tmp_path, baseline={"images": 2, "classes": 1})
    assert not (tmp_path / "reports/dataset_inventory.csv").exists()
