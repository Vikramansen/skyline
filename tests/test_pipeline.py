import json
import tempfile
import unittest
from pathlib import Path

from skyline.pipeline import rank_tonight, write_run


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_demo_ranks_unmatched_transient_above_minor_planet(self):
        ranked = rank_tonight(str(ROOT / "data/demo_alerts.jsonl"), str(ROOT / "data/demo_catalog.jsonl"))
        names = [candidate.object_id for candidate in ranked]
        self.assertEqual(names[0], "SKY-2026a")
        self.assertLess(names.index("SKY-2026a"), names.index("SKY-MPC-7"))

    def test_run_receipt_contains_delayed_label_metric(self):
        ranked = rank_tonight(str(ROOT / "data/demo_alerts.jsonl"), str(ROOT / "data/demo_catalog.jsonl"))
        with tempfile.TemporaryDirectory() as temporary:
            write_run(temporary, ranked, 5, {"SKY-2026a": "SN Ia"})
            receipt = json.loads((Path(temporary) / "tonight.json").read_text())
        self.assertIn("precision_at_k", receipt)
        self.assertEqual(receipt["objects_considered"], 5)
