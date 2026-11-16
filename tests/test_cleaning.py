import json
import builtins
import hashlib
from io import BytesIO
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import get_sampling
import pytest

from geoguessr_benchmark import __main__
from geoguessr_benchmark import cleaning
from geoguessr_benchmark.artifacts import BENCHMARK_ID, sha256_file
from geoguessr_benchmark.images import decode_rgb, pixel_sha256


@pytest.mark.parametrize("size, boxes", [
    ((1536, 662), [(1253, 15, 1505, 73), (1257, 448, 1505, 603)]),
    ((1030, 703), [(747, 15, 999, 73), (832, 554, 1000, 646)]),
])
def test_clean_cli_preserves_native_dimensions_and_raw(tmp_path, size, boxes, monkeypatch):
    root = tmp_path / "first"
    raw = root / "data/compressed_dataset/Country/sub/scene.jpg"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", size, (24, 60, 207)).save(raw)
    before = raw.read_bytes()
    source = decode_rgb(raw)
    expected = np.array(source)
    for x1, y1, x2, y2 in boxes:
        expected[y1:y2, x1:x2] = (128, 128, 128)
    encode = cleaning.encode_output_jpeg
    def check_mask(image):
        assert np.array_equal(np.asarray(image), expected)
        return encode(image)
    monkeypatch.setattr(cleaning, "encode_output_jpeg", check_mask)
    assert __main__.main(["--root", str(root), "clean"]) == 0
    cleaned = root / "data/cleaned_dataset/Country/sub/scene.jpg"
    with Image.open(cleaned) as image:
        assert image.size == size and image.mode == "RGB" and not image.getexif()
        assert get_sampling(image) == 0
        for x1, y1, x2, y2 in boxes:
            assert image.getpixel(((x1+x2)//2, (y1+y2)//2)) == (128, 128, 128)
        assert image.getpixel((0, 0)) == source.getpixel((0, 0))
    assert raw.read_bytes() == before
    manifest_path = root / "reports/cleaning_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark_id"] == BENCHMARK_ID and manifest["classes"] == ["Country"]
    assert manifest["images"] == [{
        "raw_path": raw.relative_to(root).as_posix(), "cleaned_path": cleaned.relative_to(root).as_posix(),
        "width": size[0], "height": size[1], "bboxes": [list(box) for box in boxes],
        "raw_sha256": sha256_file(raw), "cleaned_sha256": sha256_file(cleaned),
        "raw_pixel_sha256": pixel_sha256(source), "cleaned_pixel_sha256": pixel_sha256(decode_rgb(cleaned)),
    }]
    second = tmp_path / "second"
    shutil.copytree(root / "data/compressed_dataset", second / "data/compressed_dataset")
    assert __main__.main(["--root", str(second), "clean"]) == 0
    assert (second / cleaned.relative_to(root)).read_bytes() == cleaned.read_bytes()
    assert (second / manifest_path.relative_to(root)).read_bytes() == manifest_path.read_bytes()
    assert __main__.main(["--root", str(root), "clean"]) == 1
    assert raw.read_bytes() == before


def test_clean_cli_rejects_unknown_dimensions_without_publishing(tmp_path, capsys):
    raw = tmp_path / "data/compressed_dataset/A/a.jpg"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (99, 99)).save(raw)
    assert __main__.main(["--root", str(tmp_path), "clean"]) == 1
    assert "99" in capsys.readouterr().err
    assert not (tmp_path / "data/cleaned_dataset").exists()
    assert not (tmp_path / "reports/cleaning_manifest.json").exists()


def test_clean_refuses_legacy_runtime_even_if_lock_is_missing(tmp_path, capsys):
    legacy = tmp_path / "augmentation/manifests/candidates.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy candidate log")
    assert __main__.main(["--root", str(tmp_path), "clean"]) == 1
    assert "fresh v1.2 run root" in capsys.readouterr().err
    assert legacy.read_bytes() == b"legacy candidate log"


def test_clean_hashes_verified_bytes_without_output_readback(tmp_path, monkeypatch):
    raw = tmp_path / "data/compressed_dataset/A/a.jpg"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (1030, 703), (25, 60, 201)).save(raw)
    before = raw.read_bytes()
    decode = cleaning.decode_rgb
    real_open = Image.open
    real_path_open = Path.open
    real_file_open = builtins.open
    real_write = cleaning.atomic_write_bytes
    published = {}
    decoded_payloads = []

    def bytes_only_decode(source):
        assert isinstance(source, BytesIO), "cleaning must decode in-memory bytes"
        decoded_payloads.append(source.getvalue())
        return decode(source)

    def reject_output_readback(source, *args, **kwargs):
        if not isinstance(source, BytesIO):
            raise AssertionError("unexpected image path readback")
        return real_open(source, *args, **kwargs)

    def guard_path_open(path, mode="r", *args, **kwargs):
        if path.suffix == ".jpg" and path != raw and "r" in mode:
            raise AssertionError("unexpected output path readback")
        return real_path_open(path, mode, *args, **kwargs)

    def guard_file_open(path, mode="r", *args, **kwargs):
        if not isinstance(path, int) and Path(path).suffix == ".jpg" and Path(path) != raw and "r" in mode:
            raise AssertionError("unexpected output file readback")
        return real_file_open(path, mode, *args, **kwargs)

    def publish(path, payload):
        if path.suffix == ".jpg":
            assert payload in decoded_payloads, "verify before publishing"
            published[path.name] = payload
        return real_write(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(cleaning, "decode_rgb", bytes_only_decode)
        patch.setattr(Image, "open", reject_output_readback)
        patch.setattr(Path, "open", guard_path_open)
        patch.setattr(builtins, "open", guard_file_open)
        patch.setattr(cleaning, "atomic_write_bytes", publish)
        result = cleaning.clean_dataset(tmp_path)
        classes, entries = cleaning.read_cleaning_manifest(tmp_path)
    assert classes == ["A"] and len(entries) == 1
    payload = published["a.jpg"]
    entry = result["images"][0]
    assert entry["cleaned_sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["cleaned_pixel_sha256"] == pixel_sha256(decode(BytesIO(payload)))
    assert (tmp_path / entry["cleaned_path"]).read_bytes() == payload
    assert entry["raw_sha256"] == hashlib.sha256(before).hexdigest()
    assert entry["raw_pixel_sha256"] == pixel_sha256(decode(BytesIO(before)))
    assert raw.read_bytes() == before


@pytest.mark.parametrize("failure", ["encode", "decode", "dimensions", "write"])
def test_clean_output_failure_does_not_publish(tmp_path, monkeypatch, failure):
    raw = tmp_path / "data/compressed_dataset/A/a.jpg"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (1030, 703), "blue").save(raw)
    before = raw.read_bytes()

    def fail(*args):
        raise OSError("output failure")

    if failure == "encode":
        monkeypatch.setattr(cleaning, "encode_output_jpeg", fail)
    elif failure == "decode":
        monkeypatch.setattr(cleaning, "encode_output_jpeg", lambda image: b"not a JPEG")
    elif failure == "dimensions":
        encode = cleaning.encode_output_jpeg
        monkeypatch.setattr(cleaning, "encode_output_jpeg", lambda image: encode(image.resize((10, 10))))
    else:
        monkeypatch.setattr(cleaning, "atomic_write_bytes", fail)
    with pytest.raises((OSError, ValueError)):
        cleaning.clean_dataset(tmp_path)
    assert not (tmp_path / "data/cleaned_dataset").exists()
    assert not (tmp_path / "reports/cleaning_manifest.json").exists()
    assert not list((tmp_path / "data").glob(".cleaning-*"))
    assert raw.read_bytes() == before


def test_cleaned_disk_corruption_is_not_blessed_by_manifest(tmp_path, monkeypatch):
    from geoguessr_benchmark.inventory import build_inventory

    raw = tmp_path / "data/compressed_dataset/A/a.jpg"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (1030, 703), "blue").save(raw)
    write = cleaning.atomic_write_bytes

    def corrupt_output(path, payload):
        # Appended bytes keep decoding valid, so only the file hash catches this.
        return write(path, payload + b"corruption" if path.suffix == ".jpg" else payload)

    monkeypatch.setattr(cleaning, "atomic_write_bytes", corrupt_output)
    result = cleaning.clean_dataset(tmp_path)
    entry = result["images"][0]
    assert sha256_file(tmp_path / entry["cleaned_path"]) != entry["cleaned_sha256"]
    with pytest.raises(ValueError, match="cleaned image differs from cleaning manifest"):
        build_inventory(tmp_path, baseline={"images": 1, "classes": 1})
    assert not (tmp_path / "reports/dataset_inventory.csv").exists()


def test_interrupted_clean_publication_is_rejected_by_clean_and_inventory(tmp_path, monkeypatch):
    raw = tmp_path / "data/compressed_dataset/A/a.jpg"
    raw.parent.mkdir(parents=True)
    Image.new("RGB", (1030, 703), "blue").save(raw)
    before = raw.read_bytes()
    manifest = tmp_path / "reports/cleaning_manifest.json"
    published_image = tmp_path / "data/cleaned_dataset/A/a.jpg"
    write = cleaning.atomic_write_bytes

    def interrupt(path, _payload):
        if path != manifest:
            return write(path, _payload)
        assert path == manifest and published_image.is_file()
        raise OSError("interrupted before manifest publication")

    with monkeypatch.context() as patch:
        patch.setattr(cleaning, "atomic_write_bytes", interrupt)
        with pytest.raises(OSError, match="interrupted before manifest publication"):
            cleaning.clean_dataset(tmp_path)
    assert not manifest.exists()
    published_bytes = published_image.read_bytes()
    with pytest.raises(ValueError, match="incomplete cleaning publication"):
        cleaning.clean_dataset(tmp_path)
    from geoguessr_benchmark.inventory import build_inventory
    with pytest.raises(ValueError, match="incomplete cleaning publication"):
        build_inventory(tmp_path, baseline={"images": 1, "classes": 1})
    assert raw.read_bytes() == before and published_image.read_bytes() == published_bytes
