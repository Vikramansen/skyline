from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Observation:
    object_id: str
    jd: float
    band: str
    magnitude: float
    magnitude_error: float
    ra_deg: float
    dec_deg: float
    first_detected_jd: float
    positional_error_arcsec: float = 1.0

    @property
    def flux(self) -> float:
        """Relative flux; the common zero point cancels in feature ratios."""
        return 10 ** (-0.4 * self.magnitude)


@dataclass(frozen=True)
class CatalogObject:
    catalog: str
    object_id: str
    ra_deg: float
    dec_deg: float
    positional_error_arcsec: float
    kind: str


@dataclass(frozen=True)
class CrossMatch:
    catalog: str
    object_id: str
    kind: str
    separation_arcsec: float
    radius_arcsec: float


@dataclass
class Candidate:
    object_id: str
    ra_deg: float
    dec_deg: float
    age_days: float
    features: Dict[str, float]
    cross_match: Optional[CrossMatch]
    confidence: float
    value_score: float
    reasons: List[str]

    def as_dict(self) -> Dict[str, object]:
        match = None
        if self.cross_match:
            match = {
                "catalog": self.cross_match.catalog,
                "object_id": self.cross_match.object_id,
                "kind": self.cross_match.kind,
                "separation_arcsec": round(self.cross_match.separation_arcsec, 3),
                "match_radius_arcsec": round(self.cross_match.radius_arcsec, 3),
            }
        return {
            "object_id": self.object_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "age_days": round(self.age_days, 3),
            "features": {key: round(value, 5) for key, value in self.features.items()},
            "cross_match": match,
            "confidence": round(self.confidence, 5),
            "value_score": round(self.value_score, 5),
            "reasons": self.reasons,
        }
