import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from analysis.entropy import calculate_shannon_entropy
from experiment import ENCRYPTION_MODES, run_experiment


class ShannonEntropyTests(unittest.TestCase):
    def test_entropy_for_uniform_and_balanced_data(self):
        self.assertEqual(calculate_shannon_entropy(bytes([7] * 16)), 0.0)
        self.assertEqual(calculate_shannon_entropy(bytes([0, 1])), 1.0)
        self.assertEqual(calculate_shannon_entropy(bytes(range(256))), 8.0)

    def test_empty_data_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_shannon_entropy(b"")


class ExperimentTests(unittest.TestCase):
    def test_entropy_results_and_grayscale_pattern_encryption(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_directory = root / "images"
            image_directory.mkdir()
            for name in ("sample", "Pattern 1"):
                Image.new("RGB", (4, 4), color=(12, 34, 56)).save(
                    image_directory / f"{name}.png"
                )
            entropy_path = root / "metrics" / "entropy.csv"
            encrypted_directory = root / "encrypted"

            run_experiment(
                bytes(range(16)),
                image_dir=image_directory,
                output_dir=encrypted_directory,
                entropy_output_path=entropy_path,
            )

            with entropy_path.open(newline="", encoding="utf-8") as entropy_file:
                rows = list(csv.DictReader(entropy_file))

            self.assertEqual(len(rows), len(ENCRYPTION_MODES) * 2)
            self.assertEqual(
                {row["image"] for row in rows}, {"sample.png", "Pattern 1.png"}
            )
            self.assertEqual(
                {(row["image"], row["mode"]) for row in rows},
                {
                    (image_name, mode_name)
                    for image_name in ("sample.png", "Pattern 1.png")
                    for mode_name, _ in ENCRYPTION_MODES
                },
            )
            self.assertTrue(
                all(
                    0.0 <= float(row["shannon_entropy_bits_per_byte"]) <= 8.0
                    for row in rows
                )
            )
            for mode_name, _ in ENCRYPTION_MODES:
                pattern_preview = Image.open(
                    encrypted_directory
                    / mode_name
                    / "Pattern 1_encrypted.png"
                )
                sample_preview = Image.open(
                    encrypted_directory / mode_name / "sample_encrypted.png"
                )
                with pattern_preview, sample_preview:
                    self.assertEqual(pattern_preview.mode, "L")
                    self.assertEqual(sample_preview.mode, "RGB")
                self.assertEqual(
                    len(
                        (
                            encrypted_directory
                            / mode_name
                            / "Pattern 1.bin"
                        ).read_bytes()
                    ),
                    32 if mode_name in {"ecb", "cbc"} else 16,
                )
                self.assertEqual(
                    len(
                        (encrypted_directory / mode_name / "sample.bin").read_bytes()
                    ),
                    64 if mode_name in {"ecb", "cbc"} else 48,
                )


if __name__ == "__main__":
    unittest.main()
