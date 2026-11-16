"""Run AES-128-ECB, CBC, CFB, OFB, and CTR experiments on the PNG images."""

import csv
from pathlib import Path
from secrets import token_bytes
from time import perf_counter

from PIL import Image

from analysis.correlation import calculate_adjacent_pixel_correlation
from analysis.entropy import calculate_shannon_entropy
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


def run_experiment(
    key: bytes,
    image_dir: Path = IMAGE_DIR,
    output_dir: Path = OUTPUT_DIR,
    entropy_output_path: Path = ENTROPY_OUTPUT_PATH,
    correlation_output_path: Path = CORRELATION_OUTPUT_PATH,
) -> None:
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    entropy_results = []
    correlation_results = []
    for image_path in images:
        pixel_mode = "L" if image_path.stem in GRAYSCALE_IMAGE_STEMS else "RGB"
        pixel_data = load_png_pixel_bytes(image_path, pixel_mode)
        channels = 1 if pixel_mode == "L" else 3
        with Image.open(image_path) as image:
            image_size = image.size

        original_correlations = calculate_adjacent_pixel_correlation(
            pixel_data, image_size, channels
        )
        for mode_name, encrypt in ENCRYPTION_MODES:
            start = perf_counter()
            ciphertext, mode_parameter = encrypt(pixel_data, key)
            elapsed = perf_counter() - start
            entropy = calculate_shannon_entropy(ciphertext)
            entropy_results.append((image_path.name, mode_name, entropy))
            encrypted_correlations = calculate_adjacent_pixel_correlation(
                ciphertext[: len(pixel_data)], image_size, channels
            )
            correlation_results.append(
                (
                    image_path.name,
                    mode_name,
                    *(
                        original_correlations[direction]
                        for direction in CORRELATION_DIRECTIONS
                    ),
                    *(
                        encrypted_correlations[direction]
                        for direction in CORRELATION_DIRECTIONS
                    ),
                )
            )

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
            Image.frombytes(
                pixel_mode, image_size, ciphertext[: len(pixel_data)]
            ).save(preview_path)

            print(
                f"{image_path.name} ({mode_name.upper()}): "
                f"{len(ciphertext)} encrypted bytes in {elapsed:.6f} seconds "
                f"with {entropy:.6f} bits/byte entropy "
                f"-> {ciphertext_path}; preview -> {preview_path}"
            )

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


def main() -> None:
    run_experiment(token_bytes(16))


if __name__ == "__main__":
    main()
