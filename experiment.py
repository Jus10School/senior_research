"""Run AES-128-ECB experiments on the PNG images in the Images folder."""

from pathlib import Path
from secrets import token_bytes
from time import perf_counter

from encryption.ecb import encrypt_ecb
from preprocessing import load_png_pixel_bytes

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "results" / "encrypted"


def run_experiment(
    key: bytes,
    image_dir: Path = IMAGE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        start = perf_counter()
        pixel_data = load_png_pixel_bytes(image_path)
        ciphertext = encrypt_ecb(pixel_data, key)
        elapsed = perf_counter() - start

        ciphertext_path = output_dir / f"{image_path.stem}.bin"
        ciphertext_path.write_bytes(ciphertext)

        with Image.open(image_path) as image:
            image_size = image.size
        preview_byte_count = image_size[0] * image_size[1] * 3
        preview_path = output_dir / f"{image_path.stem}_encrypted.png"
        Image.frombytes(
            "RGB", image_size, ciphertext[:preview_byte_count]
        ).save(preview_path)

        print(
            f"{image_path.name}: {len(ciphertext)} encrypted bytes "
            f"in {elapsed:.6f} seconds -> {ciphertext_path}; preview -> {preview_path}"
        )


def main() -> None:
    run_experiment(token_bytes(16))


if __name__ == "__main__":
    main()
