import unittest

from analysis.ssim import calculate_ssim


class SsimTests(unittest.TestCase):
    def test_identical_images_have_perfect_similarity(self):
        pixels = bytes(range(64))

        score = calculate_ssim(
            pixels, pixels, image_size=(8, 8), channels=1
        )

        self.assertAlmostEqual(score, 1.0)

    def test_rejects_mismatched_pixel_data_length(self):
        with self.assertRaises(ValueError):
            calculate_ssim(
                b"\x00", bytes(4), image_size=(2, 2), channels=1
            )

    def test_rejects_images_too_small_for_ssim(self):
        with self.assertRaises(ValueError):
            calculate_ssim(
                bytes(4), bytes(4), image_size=(2, 2), channels=1
            )


if __name__ == "__main__":
    unittest.main()
