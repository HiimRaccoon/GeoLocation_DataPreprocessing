import hashlib
from io import BytesIO
import struct

import numpy as np
from PIL import Image
import pytest

from geoguessr_benchmark import images
from geoguessr_benchmark.images import decode_rgb, letterbox_geometry, pixel_sha256, write_output_jpeg


def test_letterbox_rounding_and_asymmetric_padding():
    a = letterbox_geometry(1536, 662, 384, 160)
    b = letterbox_geometry(1030, 703, 384, 160)
    assert a["source_bbox"] == [6, 0, 377, 160]
    assert (a["left"], a["right"]) == (6, 7)
    assert b["source_bbox"] == [75, 0, 309, 160]


def test_decode_applies_exif_and_pixel_hash_includes_dimensions(tmp_path):
    image = Image.new("RGB", (2, 3), (32, 80, 120))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.jpg"
    image.save(path, exif=exif)
    rgb = decode_rgb(path)
    assert rgb.size == (3, 2)
    assert pixel_sha256(rgb) == hashlib.sha256(struct.pack(">II", 3, 2) + rgb.tobytes()).hexdigest()
    assert pixel_sha256(Image.new("RGB", (1, 6), (0, 0, 0))) != pixel_sha256(Image.new("RGB", (2, 3), (0, 0, 0)))


def test_output_jpeg_is_rgb_without_exif_and_no_chroma_subsampling(tmp_path):
    path = tmp_path / "output.jpg"
    write_output_jpeg(path, Image.fromarray(np.full((8, 12, 3), 90, dtype=np.uint8)))
    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.size == (12, 8)
        assert not image.getexif()
        assert image.layer[0][1:3] == (1, 1)


def test_encode_output_jpeg_is_deterministic_and_preserves_policy():
    rgb = Image.fromarray(np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8))
    rgb.getexif()[274] = 6
    encoded = images.encode_output_jpeg(rgb)
    assert isinstance(encoded, bytes)
    assert encoded == images.encode_output_jpeg(rgb)
    reference = BytesIO()
    rgb.save(reference, format="JPEG", quality=95, subsampling=0, exif=b"")
    assert encoded == reference.getvalue()
    with Image.open(BytesIO(encoded)) as decoded:
        decoded.load()
        assert decoded.mode == "RGB" and decoded.size == rgb.size
        assert not decoded.getexif()
        assert decoded.layer[0][1:3] == (1, 1)
    with pytest.raises(ValueError, match="requires RGB"):
        images.encode_output_jpeg(rgb.convert("L"))
