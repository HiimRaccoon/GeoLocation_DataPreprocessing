import numpy as np
from PIL import Image
import pytest

from geoguessr_benchmark.augmentation.diffusion import prepare_generation_canvas, crop_generated_content


@pytest.mark.parametrize("size, expected_canvas, bbox", [
    ((1536, 662), (1536, 664), [0, 1, 1536, 663]),
    ((1030, 703), (1032, 704), [1, 0, 1031, 703]),
])
def test_native_generation_preserves_pixels_and_crop_excludes_padding(size, expected_canvas, bbox):
    pixels = np.random.default_rng(42).integers(0, 128, (size[1], size[0], 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    canvas, info = prepare_generation_canvas(source)
    assert canvas.size == expected_canvas
    assert info["source_bbox"] == bbox
    padded = np.asarray(canvas)
    x1, y1, x2, y2 = bbox
    assert np.array_equal(padded[y1:y2, x1:x2], pixels)
    if x1:
        assert np.array_equal(padded[y1:y2, 0], pixels[:, 1])
    if y1:
        assert np.array_equal(padded[0, x1:x2], pixels[1])
    sentinel = Image.new("RGB", expected_canvas, (255, 0, 0))
    sentinel.paste(source, (x1, y1))
    out = crop_generated_content(sentinel, info)
    assert out.size == size
    assert np.array_equal(np.asarray(out), pixels)
