import unittest

from skyline.features import light_curve_features, lomb_scargle_peak
from skyline.types import Observation


def point(jd, magnitude, error=0.05, band="g"):
    return Observation("x", jd, band, magnitude, error, 10, -10, 0)


class FeatureTests(unittest.TestCase):
    def test_rising_object_has_positive_flux_slope(self):
        features = light_curve_features([point(0, 21), point(1, 20), point(2, 19), point(3, 18)])
        self.assertGreater(features["rise_rate"], 0)

    def test_periodicity_is_bounded(self):
        points = [point(index * 0.25, 20 + (index % 2)) for index in range(8)]
        self.assertGreaterEqual(lomb_scargle_peak(points), 0)
        self.assertLessEqual(lomb_scargle_peak(points), 1)

    def test_feature_contract_preserves_error_information(self):
        features = light_curve_features([point(0, 21, 0.1), point(1, 20, 0.3), point(2, 19, 0.2)])
        self.assertAlmostEqual(features["mean_magnitude_error"], 0.2)
