import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from analysis.entropy import calculate_shannon_entropy
from analysis.performance import calculate_cpu_throughput
from experiment import ENCRYPTION_MODES, run_experiment


class CpuThroughputTests(unittest.TestCase):
    def test_calculates_decimal_megabytes_per_cpu_second(self):
        self.assertEqual(calculate_cpu_throughput(2_000_000, 2.0), 1.0)

    def test_rejects_non_positive_cpu_time(self):
        with self.assertRaises(ValueError):
            calculate_cpu_throughput(1024, 0)


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
            correlation_path = root / "metrics" / "correlation.csv"
            ssim_path = root / "metrics" / "ssim.csv"
            performance_path = root / "metrics" / "performance.csv"
            encrypted_directory = root / "encrypted"

            mode_spies = [
                (mode_name, Mock(wraps=encrypt))
                for mode_name, encrypt in ENCRYPTION_MODES
            ]
            with patch("experiment.ENCRYPTION_MODES", mode_spies):
                run_experiment(
                    bytes(range(16)),
                    image_dir=image_directory,
                    output_dir=encrypted_directory,
                    entropy_output_path=entropy_path,
                    correlation_output_path=correlation_path,
                    ssim_output_path=ssim_path,
                    performance_output_path=performance_path,
                    cycles=2,
                )
            self.assertTrue(all(encrypt.call_count == 4 for _, encrypt in mode_spies))

            with entropy_path.open(newline="", encoding="utf-8") as entropy_file:
                rows = list(csv.DictReader(entropy_file))
            with correlation_path.open(
                newline="", encoding="utf-8"
            ) as correlation_file:
                correlation_rows = list(csv.DictReader(correlation_file))
            with ssim_path.open(newline="", encoding="utf-8") as ssim_file:
                ssim_rows = list(csv.DictReader(ssim_file))
            with performance_path.open(
                newline="", encoding="utf-8"
            ) as performance_file:
                performance_rows = list(csv.DictReader(performance_file))

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
            self.assertEqual(len(correlation_rows), len(rows))
            self.assertEqual(
                {
                    (row["image"], row["mode"])
                    for row in correlation_rows
                },
                {(row["image"], row["mode"]) for row in rows},
            )
            self.assertTrue(
                all(
                    row[f"original_{direction}"] == ""
                    for row in correlation_rows
                    if row["image"] == "sample.png"
                    for direction in ("horizontal", "vertical", "diagonal")
                )
            )
            self.assertTrue(
                all(
                    not value or len(value.partition(".")[2]) == 6
                    for row in correlation_rows
                    for key, value in row.items()
                    if key.startswith(("original_", "encrypted_"))
                )
            )
            self.assertEqual(len(ssim_rows), len(rows))
            self.assertEqual(
                {(row["image"], row["mode"]) for row in ssim_rows},
                {(row["image"], row["mode"]) for row in rows},
            )
            self.assertTrue(
                all(
                    len(row["ssim"].partition(".")[2]) == 6
                    and -1.0 <= float(row["ssim"]) <= 1.0
                    for row in ssim_rows
                )
            )
            self.assertEqual(len(performance_rows), len(rows))
            self.assertEqual(
                {(row["image"], row["mode"]) for row in performance_rows},
                {(row["image"], row["mode"]) for row in rows},
            )
            self.assertTrue(
                all(
                    float(row["cpu_time_seconds"]) > 0
                    and float(row["throughput_MB_per_second"]) > 0
                    for row in performance_rows
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
