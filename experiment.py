"""Run AES-128-ECB, CBC, CFB, OFB, and CTR experiments on the PNG images."""

from pathlib import Path
from secrets import token_bytes
from time import perf_counter
from PIL import Image
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
) -> None:
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        pixel_data = load_png_pixel_bytes(image_path)
        with Image.open(image_path) as image:
            image_size = image.size
        preview_byte_count = image_size[0] * image_size[1] * 3

        for mode_name, encrypt in ENCRYPTION_MODES:
            start = perf_counter()
            ciphertext, mode_parameter = encrypt(pixel_data, key)
            elapsed = perf_counter() - start

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
                "RGB", image_size, ciphertext[:preview_byte_count]
            ).save(preview_path)

            print(
                f"{image_path.name} ({mode_name.upper()}): "
                f"{len(ciphertext)} encrypted bytes in {elapsed:.6f} seconds "
                f"-> {ciphertext_path}; preview -> {preview_path}"
            )


def main() -> None:
    run_experiment(token_bytes(16))


if __name__ == "__main__":
    main()
