import csv
import hashlib
import json

from PIL import Image
import pytest

from geoguessr_benchmark.artifacts import CLEANING_MANIFEST, canonical_csv_bytes, sha256_file, verify_stage
from geoguessr_benchmark.inventory import build_inventory as inventory_cleaned
from geoguessr_benchmark.split import allocate_split, build_split, group_for_count
from cleaned_fixture import seal_cleaned_fixture


def build_inventory(root, *, baseline):
    seal_cleaned_fixture(root)
    return inventory_cleaned(root, baseline=baseline)


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
    directory = root / "data/cleaned_dataset/A"
    directory.mkdir(parents=True)
    for i, name in enumerate("abcd"):
        Image.new("RGB", (1030, 703), (20+i*40, 30+i*10, 50)).save(directory / f"{name}.jpg")
    return directory


def test_hash_order_and_published_lock(tmp_path):
    create_images(tmp_path)
    build_inventory(tmp_path, baseline={"images": 4, "classes": 1})
    build_split(tmp_path)
    assert [r["filepath"].split("/")[-1] for r in read_csv(tmp_path / "splits/train.csv")] == ["a.jpg", "c.jpg"]
    assert read_csv(tmp_path / "splits/val.csv")[0]["filepath"].endswith("/d.jpg")
    assert read_csv(tmp_path / "splits/test.csv")[0]["filepath"].endswith("/b.jpg")
    assert verify_stage(tmp_path, "splits")["stage"] == "splits"


@pytest.mark.parametrize("holdout,pixel_only", [("d", False), ("b", True)])
def test_duplicate_across_train_and_holdout_is_colocated(tmp_path, holdout, pixel_only):
    snapshots = []
    for root in (tmp_path / "first", tmp_path / "second"):
        directory = create_images(root)
        original = (directory / "c.jpg").read_bytes()
        # A trailing byte changes the file digest without changing decoded pixels.
        (directory / f"{holdout}.jpg").write_bytes(original + b"x" if pixel_only else original)
        build_inventory(root, baseline={"images": 4, "classes": 1})
        build_split(root)
        files = [root / f"splits/{name}.csv" for name in ("train", "val", "test")]
        train, val, test = map(read_csv, files)
        assert [len(train), len(val), len(test)] == [2, 1, 1]
        assert {row["filepath"].rsplit("/", 1)[-1] for row in train} == {"c.jpg", f"{holdout}.jpg"}
        assert all(row["is_synthetic"] == "false" for row in train + val + test)
        audit = json.loads((root / "reports/split_audit.json").read_text(encoding="utf-8"))
        assert audit["cross_split_duplicates"] == []
        assert audit["split_support"] == {"train": 1, "val": 1, "test": 1}
        verify_stage(root, "splits")
        snapshots.append([path.read_bytes() for path in files])
    assert snapshots[0] == snapshots[1]


def test_duplicate_group_larger_than_any_quota_blocks_lock(tmp_path):
    directory = create_images(tmp_path)
    for name in ("c", "d"):
        (directory / f"{name}.jpg").write_bytes((directory / "b.jpg").read_bytes())
    build_inventory(tmp_path, baseline={"images": 4, "classes": 1})
    with pytest.raises(ValueError, match="duplicate"):
        build_split(tmp_path)
    assert not (tmp_path / "benchmark_lock.json").exists()


def test_split_can_reconsider_an_earlier_duplicate_group_placement(tmp_path):
    directory = create_images(tmp_path)
    paths = list(directory.glob("*.jpg")) + [directory / f"image-{n}.jpg" for n in range(21)]
    paths.sort(key=lambda path: hashlib.sha256(
        ("split-v1|42|" + path.relative_to(tmp_path).as_posix()).encode()).digest())
    # The first group of three prefers train, leaving one train slot. The other
    # triple fills val, so one of the two remaining pairs would be stranded.
    groups = [[8, 11, 14, 16, 20, 22], [0, 2, 4, 7, 9], [12, 15, 23, 24],
              [3, 5, 6], [18, 19, 21], [1, 13], [10, 17]]
    for color, members in enumerate(groups):
        first = paths[members[0]]
        Image.new("RGB", (1030, 703), (color * 35, 30, 50)).save(first)
        for index in members[1:]:
            paths[index].write_bytes(first.read_bytes())
    build_inventory(tmp_path, baseline={"images": 25, "classes": 1})
    build_split(tmp_path)
    splits = [read_csv(tmp_path / f"splits/{name}.csv") for name in ("train", "val", "test")]
    assert [len(rows) for rows in splits] == [19, 3, 3]
    by_name = {row["filepath"].rsplit("/", 1)[-1]: row["split"] for rows in splits for row in rows}
    for members in groups:
        assert len({by_name[paths[index].name] for index in members}) == 1


def test_split_rejects_conflicting_labels_even_if_audit_omits_them(tmp_path):
    directory = create_images(tmp_path)
    other_class = directory.with_name("B")
    other_class.mkdir()
    other_file = other_class / "a.jpg"
    Image.new("RGB", (1030, 703), "blue").save(other_file)
    build_inventory(tmp_path, baseline={"images": 5, "classes": 2})
    inventory = tmp_path / "reports/dataset_inventory.csv"
    rows = read_csv(inventory)
    source = next(row for row in rows if row["filepath"].endswith("/A/a.jpg"))
    duplicate = next(row for row in rows if row["class_name"] == "B")
    other_file.write_bytes((directory / "a.jpg").read_bytes())
    for key in ("sha256", "decoded_pixel_sha256"):
        duplicate[key] = source[key]
    inventory.write_bytes(canonical_csv_bytes(rows, list(source)))
    with pytest.raises(ValueError, match="conflicting labels"):
        build_inventory(tmp_path, baseline={"images": 5, "classes": 2})
    audit_path = tmp_path / "reports/split_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["conflicting_duplicates"] = []
    audit["dataset_fingerprint"] = sha256_file(inventory)
    audit["cleaning_manifest_sha256"] = sha256_file(tmp_path / CLEANING_MANIFEST)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting labels"):
        build_split(tmp_path)
    assert not (tmp_path / "benchmark_lock.json").exists()


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
                        "data/cleaned_dataset/A/c.jpg,pano-1,10.0,106.0\n"
                        "data/cleaned_dataset/A/d.jpg,pano-1,10.00001,106.0\n",encoding="utf-8")
    build_split(tmp_path)
    audit=json.loads((tmp_path/"reports/split_audit.json").read_text(encoding="utf-8"))
    geo=audit["geographic_metadata"]
    assert geo["metadata_rows"]==2 and geo["radius_m"]==25.0
    assert geo["cross_split_same_panorama_count"]==1
    assert geo["cross_split_nearby_pair_count"]==1
    assert "data/location_metadata.csv" in verify_stage(tmp_path,"splits")["artifacts"]
