import tempfile
import unittest
from pathlib import Path

from skyline.operations import ReviewStore, snapshot_id


class OperationsTests(unittest.TestCase):
    def test_snapshot_id_changes_when_model_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            alert = Path(directory) / "alerts.jsonl"
            alert.write_text("{}\n")
            self.assertNotEqual(snapshot_id([str(alert)], 20, "baseline"), snapshot_id([str(alert)], 20, "jax"))

    def test_review_is_persisted_and_bound_to_snapshot_member(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(str(Path(directory) / "skyline.sqlite"))
            store.save_snapshot("run", "baseline", 1, 20, [{"object_id": "target"}])
            review = store.record_review("run", "target", "follow_up", "early rise")
            self.assertEqual(review["decision"], "follow_up")
            self.assertEqual(store.decisions("run")["target"]["note"], "early rise")
            with self.assertRaises(LookupError):
                store.record_review("run", "other", "watch")
