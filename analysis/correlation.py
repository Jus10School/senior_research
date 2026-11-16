"""Adjacent-pixel Pearson correlation calculations."""

import numpy as np


_DIRECTIONS = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "diagonal": (1, 1),
}


def calculate_adjacent_pixel_correlation(
    pixel_data: bytes, image_size: tuple[int, int], channels: int
) -> dict[str, float | None]:
    """Return mean per-channel Pearson correlations for neighboring pixels.

    Correlations are ``None`` when the image is too small for a direction or
    none of its channels has variance among the corresponding pixel pairs.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if channels not in (1, 3):
        raise ValueError("Pixel data must have one or three channels")
    if len(pixel_data) != width * height * channels:
        raise ValueError("Pixel data length does not match the image dimensions")

    pixels = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
        height, width, channels
    )
    correlations: dict[str, float | None] = {}
    for direction, (vertical_offset, horizontal_offset) in _DIRECTIONS.items():
        if height <= vertical_offset or width <= horizontal_offset:
            correlations[direction] = None
            continue

        first = pixels[
            : height - vertical_offset, : width - horizontal_offset, :
        ]
        second = pixels[
            vertical_offset:, horizontal_offset:, :
        ]
        channel_correlations = []
        for channel in range(channels):
            first_values = first[:, :, channel].reshape(-1).astype(np.float64)
            second_values = second[:, :, channel].reshape(-1).astype(np.float64)
            first_centered = first_values - first_values.mean()
            second_centered = second_values - second_values.mean()
            denominator = np.linalg.norm(first_centered) * np.linalg.norm(
                second_centered
            )
            if denominator:
                correlation = np.dot(first_centered, second_centered) / denominator
                channel_correlations.append(float(np.clip(correlation, -1.0, 1.0)))

        correlations[direction] = (
            float(np.mean(channel_correlations)) if channel_correlations else None
        )

    return correlations