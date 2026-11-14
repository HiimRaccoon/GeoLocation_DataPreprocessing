from PIL import Image
from geoguessr_benchmark.augmentation.diffusion import prepare_generation_canvas, crop_generated_content


def test_generation_crop_excludes_padding_sentinel():
    source = Image.new("RGB", (1536, 662), "blue")
    canvas, info = prepare_generation_canvas(source)
    assert canvas.size == (384, 160)
    assert info["source_bbox"] == [6, 0, 377, 160]
    for x in list(range(6)) + list(range(377,384)):
        for y in range(160):
            canvas.putpixel((x,y), (255,0,0))
    crop = crop_generated_content(canvas, info["source_bbox"])
    assert crop.size == (371,160)
    assert crop.getextrema()[0] == (0, 0)
    _, info2 = prepare_generation_canvas(Image.new("RGB", (1030,703), "blue"))
    assert info2["source_bbox"] == [75,0,309,160]



