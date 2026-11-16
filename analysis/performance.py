"""Encryption performance calculations."""


def calculate_cpu_throughput(
    data_size_bytes: int, cpu_time_seconds: float
) -> float:
    """Return throughput in decimal megabytes per CPU second."""
    if data_size_bytes < 0:
        raise ValueError("Data size cannot be negative")
    if cpu_time_seconds <= 0:
        raise ValueError("CPU time must be positive")

    return data_size_bytes / cpu_time_seconds / 1_000_000