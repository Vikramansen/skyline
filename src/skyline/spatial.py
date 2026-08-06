"""A local spherical grid index with uncertainty-aware exact matching."""

import math
from collections import defaultdict
from typing import DefaultDict, Iterable, List, Optional, Tuple

from .types import CatalogObject, CrossMatch, Observation


def angular_distance_arcsec(ra_a: float, dec_a: float, ra_b: float, dec_b: float) -> float:
    """Great-circle separation, stable for small match radii."""
    ra_a, dec_a, ra_b, dec_b = map(math.radians, (ra_a, dec_a, ra_b, dec_b))
    cosine = math.sin(dec_a) * math.sin(dec_b) + math.cos(dec_a) * math.cos(dec_b) * math.cos(ra_a - ra_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600


class SphericalGridIndex:
    """Coarse candidate retrieval before an exact great-circle comparison.

    The interface mirrors a HEALPix-backed index and is deliberately small so
    production data can swap in ``healpy`` without changing ranker logic.
    """
    def __init__(self, catalog: Iterable[CatalogObject], cell_degrees: float = 0.05):
        self.cell_degrees = cell_degrees
        self._ra_cells = math.ceil(360 / cell_degrees)
        self._cells: DefaultDict[Tuple[int, int], List[CatalogObject]] = defaultdict(list)
        for item in catalog:
            self._cells[self._cell(item.ra_deg, item.dec_deg)].append(item)

    def _cell(self, ra_deg: float, dec_deg: float) -> Tuple[int, int]:
        return (int((ra_deg % 360) / self.cell_degrees), int((dec_deg + 90) / self.cell_degrees))

    def nearest(self, alert: Observation, sigma: float = 3.0, min_radius_arcsec: float = 1.0, max_radius_arcsec: float = 30.0) -> Optional[CrossMatch]:
        # Candidate cells are chosen for the largest legal radius. The final
        # acceptance radius adapts to the particular pair's two uncertainties.
        cell_radius = max(1, math.ceil((max_radius_arcsec / 3600) / self.cell_degrees))
        cell_x, cell_y = self._cell(alert.ra_deg, alert.dec_deg)
        nearest = None
        for x_offset in range(-cell_radius, cell_radius + 1):
            for y_offset in range(-cell_radius, cell_radius + 1):
                key = ((x_offset + cell_x) % self._ra_cells, y_offset + cell_y)
                for item in self._cells.get(key, []):
                    radius = min(max_radius_arcsec, max(min_radius_arcsec, sigma * math.hypot(alert.positional_error_arcsec, item.positional_error_arcsec)))
                    separation = angular_distance_arcsec(alert.ra_deg, alert.dec_deg, item.ra_deg, item.dec_deg)
                    if separation <= radius and (nearest is None or separation < nearest.separation_arcsec):
                        nearest = CrossMatch(item.catalog, item.object_id, item.kind, separation, radius)
        return nearest
