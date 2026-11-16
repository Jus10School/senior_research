import unittest

from analysis.correlation import calculate_adjacent_pixel_correlation


class AdjacentPixelCorrelationTests(unittest.TestCase):
    def test_calculates_correlations_for_each_direction(self):
        pixels = bytes([0, 255, 0, 255] * 2)

        correlations = calculate_adjacent_pixel_correlation(
            pixels, image_size=(4, 2), channels=1
        )

        self.assertAlmostEqual(correlations["horizontal"], -1.0)
        self.assertAlmostEqual(correlations["vertical"], 1.0)
        self.assertAlmostEqual(correlations["diagonal"], -1.0)

    def test_returns_undefined_for_constant_pixels(self):
        correlations = calculate_adjacent_pixel_correlation(
            bytes([128] * 16), image_size=(4, 4), channels=1
        )

        self.assertEqual(
            correlations,
            {"horizontal": None, "vertical": None, "diagonal": None},
        )

    def test_rejects_mismatched_pixel_data_length(self):
        with self.assertRaises(ValueError):
            calculate_adjacent_pixel_correlation(
                b"\x00", image_size=(2, 2), channels=1
            )


if __name__ == "__main__":
    unittest.main()
