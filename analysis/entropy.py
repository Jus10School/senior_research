"""Shannon entropy calculations for encrypted image data."""

from math import log2


def calculate_shannon_entropy(data: bytes) -> float:
    """Return Shannon entropy in bits per byte for a byte sequence."""
    if not data:
        raise ValueError("Cannot calculate entropy for empty data")

    counts = [0] * 256
    for value in data:
        counts[value] += 1

    data_length = len(data)
    return -sum(
        (count / data_length) * log2(count / data_length)
        for count in counts
        if count
    )