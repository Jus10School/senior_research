"""Load PNG images and extract their pixel data for encryption."""

from pathlib import Path
from typing import Literal

from PIL import Image


def load_png_pixel_bytes(
    input_path: str | Path, mode: Literal["L", "RGB"] = "RGB"
) -> bytes:
    """Return decoded PNG pixels as grayscale or RGB bytes."""
    with Image.open(input_path) as image:
        if image.format != "PNG":
            raise ValueError("Input must be a PNG image")
        return image.convert(mode).tobytes()
