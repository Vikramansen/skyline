import unittest

from skyline.spatial import SphericalGridIndex, angular_distance_arcsec
from skyline.types import CatalogObject, Observation


class SpatialTests(unittest.TestCase):
    def test_angular_distance_at_equator(self):
        self.assertAlmostEqual(angular_distance_arcsec(0, 0, 1, 0), 3600, places=3)

    def test_match_radius_uses_combined_uncertainty(self):
        index = SphericalGridIndex([CatalogObject("Gaia", "nearby", 10.0002, 0, 1, "star")])
        alert = Observation("a", 1, "g", 20, 0.1, 10, 0, 1, 1)
        match = index.nearest(alert)
        self.assertIsNotNone(match)
        self.assertGreater(match.radius_arcsec, 3.0)

    def test_index_wraps_at_zero_right_ascension(self):
        index = SphericalGridIndex([CatalogObject("SIMBAD", "edge", 359.9999, 0, 0.2, "galaxy")])
        alert = Observation("a", 1, "g", 20, 0.1, 0.0001, 0, 1, 1)
        self.assertIsNotNone(index.nearest(alert))
