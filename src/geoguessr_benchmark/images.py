"""Image decoding, canonical pixel hashing and shared letterbox geometry."""
import hashlib
from io import BytesIO
import math
import struct

from PIL import Image, ImageOps


def decode_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def pixel_sha256(rgb):
    if rgb.mode != "RGB":
        raise ValueError("pixel hashing requires RGB")
    return hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()


def write_output_jpeg(path, rgb):
    if rgb.mode != "RGB":
        raise ValueError("JPEG output requires RGB")
    rgb.save(path, format="JPEG", quality=95, subsampling=0, exif=b"")


def encode_output_jpeg(rgb):
    """Encode with the shared output policy without touching the filesystem."""
    buffer = BytesIO()
    write_output_jpeg(buffer, rgb)
    return buffer.getvalue()


def letterbox_geometry(width, height, canvas_width, canvas_height):
    if min(width, height, canvas_width, canvas_height) < 1:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(canvas_width / width, canvas_height / height)
    new_width = min(canvas_width, max(1, math.floor(width * scale + 0.5)))
    new_height = min(canvas_height, max(1, math.floor(height * scale + 0.5)))
    left = (canvas_width - new_width) // 2
    top = (canvas_height - new_height) // 2
    return {
        "width": new_width, "height": new_height,
        "left": left, "right": canvas_width - new_width - left,
        "top": top, "bottom": canvas_height - new_height - top,
        "source_bbox": [left, top, left + new_width, top + new_height],
    }
