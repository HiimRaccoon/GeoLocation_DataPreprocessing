from PIL import Image
import numpy as np
import pytest

from geoguessr_benchmark.augmentation.traditional import generate_traditional


@pytest.mark.parametrize("size", [(1536, 662), (1030, 703)])
def test_traditional_is_deterministic_and_preserves_size(size):
    width, height = size
    source = Image.fromarray(np.random.default_rng(4).integers(0, 256, (height,width,3), dtype=np.uint8))
    attempt = {"generation_seed": 1121161069}
    first, params = generate_traditional(source, attempt, {})
    again, params2 = generate_traditional(source, attempt, {})
    assert first.size == size and first.mode == "RGB"
    assert first.tobytes() == again.tobytes()
    assert params == params2
    assert len(params) in (1,2)
    names = [item["name"] for item in params]
    order = ["brightness", "contrast", "gamma", "gaussian_noise", "gaussian_blur", "rotation", "jpeg_variation"]
    assert names == sorted(set(names), key=order.index)


def test_all_seven_transforms_can_run_and_preserve_shape():
    source = Image.fromarray(np.random.default_rng(5).integers(0, 256, (16, 24, 3), dtype=np.uint8))
    seen = set()
    for seed in range(60):
        out, parameters = generate_traditional(source, {"generation_seed": seed}, {})
        assert out.size == source.size
        seen.update(item["name"] for item in parameters)
        if len(seen) == 7:
            break
    assert seen == {"brightness", "contrast", "gamma", "gaussian_noise", "gaussian_blur", "rotation", "jpeg_variation"}
