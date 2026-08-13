import json
import tempfile
import unittest
from pathlib import Path

from skyline.alerce_ingest import detection_to_alert, fetch_alerts, fetch_object_alerts, write_snapshot


class FakeAlerce:
    def query_objects(self, **_kwargs):
        return [{"oid": "ZTF-test", "mjdstarthist": 60000.0}]

    def query_detections(self, oid, **_kwargs):
        self.oid = oid
        return [
            {"mjd": 60000.1, "magpsf": 20.2, "sigmapsf": 0.1, "ra": 10, "dec": -20, "fid": 1, "drb": 0.8},
            {"mjd": 60000.2, "magpsf": 20.3, "sigmapsf": 0.1, "ra": 10, "dec": -20, "fid": 2, "drb": 0.1},
        ]


class AlerceIngestTests(unittest.TestCase):
    def test_maps_ztf_fields_and_converts_mjd_to_jd(self):
        alert = detection_to_alert("ZTF-test", {"mjd": 60000, "magpsf": 20, "sigmapsf": 0.1, "ra": 1, "dec": 2, "fid": 2}, 59999)
        self.assertEqual(alert["band"], "r")
        self.assertEqual(alert["jd"], 2460000.5)
        self.assertEqual(alert["first_detected_jd"], 2459999.5)

    def test_filters_explicit_low_drb_score(self):
        alerts = fetch_alerts(FakeAlerce(), 60000, 60001, 10, 0.5)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["object_id"], "ZTF-test")

    def test_object_mode_keeps_the_recent_bounded_history(self):
        alerts = fetch_object_alerts(FakeAlerce(), ["ZTF-test"], 0.0, 1)
        self.assertEqual(len(alerts), 1)

    def test_bad_corrected_uncertainty_falls_back_to_raw_photometry(self):
        alert = detection_to_alert("ZTF-test", {"mjd": 60000, "magpsf": 20, "sigmapsf": 0.1, "magpsf_corr": 18, "sigmapsf_corr": 100, "corrected": True, "ra": 1, "dec": 2, "fid": 1}, 60000)
        self.assertEqual(alert["magnitude"], 20)

    def test_snapshot_writes_a_provenance_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_snapshot(temporary, [], 60000, 60001, 10, 0.5)
            receipt = json.loads((path / "receipt.json").read_text())
        self.assertEqual(receipt["source"], "ALeRCE ZTF public API")
