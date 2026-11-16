"""Structural similarity calculations for original and encrypted pixels."""

import numpy as np
from skimage.metrics import structural_similarity


def calculate_ssim(
    original_data: bytes,
    encrypted_data: bytes,
    image_size: tuple[int, int],
    channels: int,
) -> float:
    """Return SSIM for two grayscale or RGB images with 8-bit pixels."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if channels not in (1, 3):
        raise ValueError("Pixel data must have one or three channels")

    expected_length = width * height * channels
    if len(original_data) != expected_length or len(encrypted_data) != expected_length:
        raise ValueError("Pixel data length does not match the image dimensions")

    image_shape = (height, width) if channels == 1 else (height, width, channels)
    original = np.frombuffer(original_data, dtype=np.uint8).reshape(image_shape)
    encrypted = np.frombuffer(encrypted_data, dtype=np.uint8).reshape(image_shape)

    window_size = min(7, width, height)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires images at least 3 pixels wide and high")

    return float(
        structural_similarity(
            original,
            encrypted,
            data_range=255,
            channel_axis=None if channels == 1 else -1,
            win_size=window_size,
        )
    )