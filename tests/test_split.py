import csv
import json

from PIL import Image
import pytest

from geoguessr_benchmark.artifacts import verify_stage
from geoguessr_benchmark.inventory import build_inventory
from geoguessr_benchmark.split import allocate_split, build_split, group_for_count


@pytest.mark.parametrize("n,expected", [(1,(1,0,0)), (2,(1,0,1)), (3,(1,1,1)), (9,(7,1,1)), (10,(8,1,1)), (15,(11,2,2))])
def test_allocation_boundaries(n, expected):
    assert allocate_split(n) == expected


@pytest.mark.parametrize("n,expected", [(1,("tail",5,4)), (19,("tail",95,76)), (20,("medium",100,80)), (99,("medium",100,1)), (100,("many",100,0))])
def test_group_boundaries(n, expected):
    assert group_for_count(n) == expected


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def create_images(root):
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs/specs.md").write_text("fixture spec", encoding="utf-8")
    directory = root / "data/compressed_dataset/A"
    directory.mkdir(parents=True)
    for i, name in enumerate("abcd"):
        Image.new("RGB", (5, 7), (20+i*40, 30+i*10, 50)).save(directory / f"{name}.jpg")
    return directory


def test_hash_order_and_published_lock(tmp_path):
    create_images(tmp_path)
    build_inventory(tmp_path, baseline={"images": 4, "classes": 1})
    build_split(tmp_path)
    assert [r["filepath"].split("/")[-1] for r in read_csv(tmp_path / "splits/train.csv")] == ["a.jpg", "b.jpg"]
    assert read_csv(tmp_path / "splits/val.csv")[0]["filepath"].endswith("/d.jpg")
    assert read_csv(tmp_path / "splits/test.csv")[0]["filepath"].endswith("/c.jpg")
    assert verify_stage(tmp_path, "splits")["stage"] == "splits"


def test_exact_duplicate_across_splits_blocks_lock(tmp_path):
    directory = create_images(tmp_path)
    (directory / "d.jpg").write_bytes((directory / "b.jpg").read_bytes())
    build_inventory(tmp_path, baseline={"images": 4, "classes": 1})
    with pytest.raises(ValueError, match="duplicate"):
        build_split(tmp_path)
    assert not (tmp_path / "benchmark_lock.json").exists()
    audit = json.loads((tmp_path / "reports/split_audit.json").read_text(encoding="utf-8"))
    assert audit["cross_split_duplicates"]


def test_inventory_does_not_overwrite_published_split_audit(tmp_path):
    create_images(tmp_path)
    build_inventory(tmp_path, baseline={"images": 4, "classes": 1})
    build_split(tmp_path)
    audit_before = (tmp_path / "reports/split_audit.json").read_bytes()
    with pytest.raises(ValueError, match="lock"):
        build_inventory(tmp_path, baseline={"images": 4, "classes": 1})
    assert (tmp_path / "reports/split_audit.json").read_bytes() == audit_before
    verify_stage(tmp_path, "splits")


def test_new_source_after_inventory_requires_new_audit(tmp_path):
    directory = create_images(tmp_path)
    build_inventory(tmp_path, baseline={"images":4,"classes":1})
    Image.new("RGB", (5,7), (7,88,159)).save(directory / "new.jpg")
    with pytest.raises(ValueError, match="inventory"):
        build_split(tmp_path)
    assert not (tmp_path / "benchmark_lock.json").exists()


def test_split_lock_includes_frozen_spec(tmp_path):
    create_images(tmp_path)
    docs = tmp_path / "docs"
    (docs / "specs.md").write_text("frozen spec", encoding="utf-8")
    build_inventory(tmp_path, baseline={"images":4,"classes":1})
    lock = build_split(tmp_path)
    assert "docs/specs.md" in lock["artifacts"]
    (docs / "specs.md").write_text("changed spec", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_stage(tmp_path,"splits")


def test_optional_geographic_metadata_reports_cross_split_nearby_and_panorama(tmp_path):
    create_images(tmp_path)
    build_inventory(tmp_path,baseline={"images":4,"classes":1})
    metadata = tmp_path / "data/location_metadata.csv"
    metadata.write_text("filepath,panorama_id,latitude,longitude\n"
                        "data/compressed_dataset/A/b.jpg,pano-1,10.0,106.0\n"
                        "data/compressed_dataset/A/d.jpg,pano-1,10.00001,106.0\n",encoding="utf-8")
    build_split(tmp_path)
    audit=json.loads((tmp_path/"reports/split_audit.json").read_text(encoding="utf-8"))
    geo=audit["geographic_metadata"]
    assert geo["metadata_rows"]==2 and geo["radius_m"]==25.0
    assert geo["cross_split_same_panorama_count"]==1
    assert geo["cross_split_nearby_pair_count"]==1
    assert "data/location_metadata.csv" in verify_stage(tmp_path,"splits")["artifacts"]
