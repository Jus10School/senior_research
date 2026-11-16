"""Run AES-128-ECB, CBC, CFB, OFB, and CTR experiments on the PNG images."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from secrets import token_bytes
from statistics import fmean
from time import process_time

from PIL import Image

from analysis.correlation import calculate_adjacent_pixel_correlation
from analysis.entropy import calculate_shannon_entropy
from analysis.performance import calculate_cpu_throughput
from analysis.ssim import calculate_ssim
from preprocessing import load_png_pixel_bytes

from encryption import (
    encrypt_cbc,
    encrypt_cfb,
    encrypt_ctr,
    encrypt_ecb,
    encrypt_ofb,
)




PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "results" / "encrypted"
ENTROPY_OUTPUT_PATH = PROJECT_DIR / "results" / "metrics" / "entropy.csv"
CORRELATION_OUTPUT_PATH = PROJECT_DIR / "results" / "metrics" / "correlation.csv"
SSIM_OUTPUT_PATH = PROJECT_DIR / "results" / "metrics" / "ssim.csv"
PERFORMANCE_OUTPUT_PATH = PROJECT_DIR / "results" / "metrics" / "performance.csv"
GRAYSCALE_IMAGE_STEMS = frozenset({"Pattern 1", "Pattern 2", "Pattern 3"})
CORRELATION_DIRECTIONS = ("horizontal", "vertical", "diagonal")


def _encrypt_ecb(
    pixel_data: bytes, key: bytes
) -> tuple[bytes, None]:
    return encrypt_ecb(pixel_data, key), None


ENCRYPTION_MODES = (
    ("ecb", _encrypt_ecb),
    ("cbc", encrypt_cbc),
    ("cfb", encrypt_cfb),
    ("ofb", encrypt_ofb),
    ("ctr", encrypt_ctr),
)


def _process_image_mode(
    *,
    image_path: Path,
    pixel_mode: str,
    pixel_data: bytes,
    channels: int,
    image_size: tuple[int, int],
    original_correlations: dict[str, float | None],
    cycle_key: bytes,
    mode_name: str,
    encrypt,
    output_dir: Path,
    is_final_cycle: bool,
):
    cpu_start = process_time()
    ciphertext, mode_parameter = encrypt(pixel_data, cycle_key)
    cpu_time = process_time() - cpu_start
    throughput = calculate_cpu_throughput(len(pixel_data), cpu_time)

    entropy = calculate_shannon_entropy(ciphertext)
    encrypted_pixels = ciphertext[: len(pixel_data)]
    encrypted_correlations = calculate_adjacent_pixel_correlation(
        encrypted_pixels, image_size, channels
    )
    correlation = tuple(
        original_correlations[direction] for direction in CORRELATION_DIRECTIONS
    ) + tuple(
        encrypted_correlations[direction] for direction in CORRELATION_DIRECTIONS
    )
    ssim = calculate_ssim(pixel_data, encrypted_pixels, image_size, channels)

    result = {
        "image_name": image_path.name,
        "mode_name": mode_name,
        "entropy": entropy,
        "correlation": correlation,
        "ssim": ssim,
        "performance": (cpu_time, throughput),
    }

    if is_final_cycle:
        mode_output_dir = output_dir / mode_name
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        ciphertext_path = mode_output_dir / f"{image_path.stem}.bin"
        ciphertext_path.write_bytes(ciphertext)
        if mode_parameter is not None:
            parameter_suffix = ".counter" if mode_name == "ctr" else ".iv"
            parameter_path = (
                mode_output_dir / f"{image_path.stem}{parameter_suffix}"
            )
            parameter_path.write_bytes(mode_parameter)

        preview_path = mode_output_dir / f"{image_path.stem}_encrypted.png"
        Image.frombytes(pixel_mode, image_size, encrypted_pixels).save(preview_path)

    return result


def run_experiment(
    key: bytes | None = None,
    image_dir: Path = IMAGE_DIR,
    output_dir: Path = OUTPUT_DIR,
    entropy_output_path: Path = ENTROPY_OUTPUT_PATH,
    correlation_output_path: Path = CORRELATION_OUTPUT_PATH,
    ssim_output_path: Path = SSIM_OUTPUT_PATH,
    performance_output_path: Path = PERFORMANCE_OUTPUT_PATH,
    cycles: int = 10,
    workers: int | None = None,
) -> None:
    if cycles <= 0:
        raise ValueError("Number of cycles must be positive")
    if workers is not None and workers <= 0:
        raise ValueError("Number of workers must be positive")

    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    image_inputs = []
    for image_path in images:
        pixel_mode = "L" if image_path.stem in GRAYSCALE_IMAGE_STEMS else "RGB"
        pixel_data = load_png_pixel_bytes(image_path, pixel_mode)
        channels = 1 if pixel_mode == "L" else 3
        with Image.open(image_path) as image:
            image_size = image.size
        original_correlations = calculate_adjacent_pixel_correlation(
            pixel_data, image_size, channels
        )
        image_inputs.append(
            (
                image_path,
                pixel_mode,
                pixel_data,
                channels,
                image_size,
                original_correlations,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    entropy_samples = {}
    correlation_samples = {}
    ssim_samples = {}
    performance_samples = {}
    for cycle in range(cycles):
        cycle_key = key if key is not None else token_bytes(16)
        tasks = []
        for (
            image_path,
            pixel_mode,
            pixel_data,
            channels,
            image_size,
            original_correlations,
        ) in image_inputs:
            for mode_name, encrypt in ENCRYPTION_MODES:
                tasks.append(
                    (
                        image_path,
                        pixel_mode,
                        pixel_data,
                        channels,
                        image_size,
                        original_correlations,
                        cycle_key,
                        mode_name,
                        encrypt,
                    )
                )

        max_workers = workers or min(8, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(
                lambda args: _process_image_mode(
                    image_path=args[0],
                    pixel_mode=args[1],
                    pixel_data=args[2],
                    channels=args[3],
                    image_size=args[4],
                    original_correlations=args[5],
                    cycle_key=args[6],
                    mode_name=args[7],
                    encrypt=args[8],
                    output_dir=output_dir,
                    is_final_cycle=(cycle == cycles - 1),
                ),
                tasks,
            ):
                metric_key = (result["image_name"], result["mode_name"])
                entropy_samples.setdefault(metric_key, []).append(
                    result["entropy"]
                )
                correlation_samples.setdefault(metric_key, []).append(
                    result["correlation"]
                )
                ssim_samples.setdefault(metric_key, []).append(result["ssim"])
                performance_samples.setdefault(metric_key, []).append(
                    result["performance"]
                )

        print(f"Cycle {cycle + 1}/{cycles} complete", flush=True)

    entropy_results = [
        (image_name, mode_name, fmean(samples))
        for (image_name, mode_name), samples in entropy_samples.items()
    ]
    correlation_results = [
        (
            image_name,
            mode_name,
            *(
                fmean(value for value in values if value is not None)
                if any(value is not None for value in values)
                else None
                for values in zip(*samples)
            ),
        )
        for (image_name, mode_name), samples in correlation_samples.items()
    ]
    ssim_results = [
        (image_name, mode_name, fmean(samples))
        for (image_name, mode_name), samples in ssim_samples.items()
    ]
    performance_results = [
        (
            image_name,
            mode_name,
            fmean(cpu_time for cpu_time, _ in samples),
            fmean(throughput for _, throughput in samples),
        )
        for (image_name, mode_name), samples in performance_samples.items()
    ]

    entropy_output_path.parent.mkdir(parents=True, exist_ok=True)
    with entropy_output_path.open("w", newline="", encoding="utf-8") as entropy_file:
        writer = csv.writer(entropy_file, lineterminator="\n")
        writer.writerow(("image", "mode", "shannon_entropy_bits_per_byte"))
        writer.writerows(entropy_results)

    correlation_output_path.parent.mkdir(parents=True, exist_ok=True)
    with correlation_output_path.open(
        "w", newline="", encoding="utf-8"
    ) as correlation_file:
        writer = csv.writer(correlation_file, lineterminator="\n")
        writer.writerow(
            (
                "image",
                "mode",
                *(f"original_{direction}" for direction in CORRELATION_DIRECTIONS),
                *(f"encrypted_{direction}" for direction in CORRELATION_DIRECTIONS),
            )
        )
        writer.writerows(
            (
                image_name,
                mode_name,
                *(
                    f"{value:.6f}" if value is not None else ""
                    for value in values
                ),
            )
            for image_name, mode_name, *values in correlation_results
        )

    ssim_output_path.parent.mkdir(parents=True, exist_ok=True)
    with ssim_output_path.open("w", newline="", encoding="utf-8") as ssim_file:
        writer = csv.writer(ssim_file, lineterminator="\n")
        writer.writerow(("image", "mode", "ssim"))
        writer.writerows(
            (image_name, mode_name, f"{score:.6f}")
            for image_name, mode_name, score in ssim_results
        )

    performance_output_path.parent.mkdir(parents=True, exist_ok=True)
    with performance_output_path.open(
        "w", newline="", encoding="utf-8"
    ) as performance_file:
        writer = csv.writer(performance_file, lineterminator="\n")
        writer.writerow(
            ("image", "mode", "cpu_time_seconds", "throughput_MB_per_second")
        )
        writer.writerows(
            (
                image_name,
                mode_name,
                f"{cpu_time:.6f}",
                f"{throughput:.6f}",
            )
            for image_name, mode_name, cpu_time, throughput in performance_results
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AES mode comparison experiment on the project images."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Number of random-key sampling cycles to run (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional thread-pool size for parallel image-mode evaluation.",
    )
    args = parser.parse_args()
    run_experiment(cycles=args.cycles, workers=args.workers)


if __name__ == "__main__":
    main()
