import json
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
    write = cleaning.write_output_jpeg
    def check_mask(path, image):
        assert np.array_equal(np.asarray(image), expected)
        write(path, image)
    monkeypatch.setattr(cleaning, "write_output_jpeg", check_mask)
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
