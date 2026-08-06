import unittest

from skyline.model import FEATURE_NAMES, LinearRanker


class ModelTests(unittest.TestCase):
    def test_model_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            LinearRanker([1.0], 0.0, [0.0], [1.0])

    def test_model_round_trip_preserves_prediction(self):
        model = LinearRanker([0.1] * len(FEATURE_NAMES), -0.2, [0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES))
        restored = LinearRanker.from_dict(model.to_dict())
        self.assertAlmostEqual(model.predict({"novelty": 1.0}), restored.predict({"novelty": 1.0}))

    def test_model_rejects_feature_contract_drift(self):
        model = LinearRanker([0.1] * len(FEATURE_NAMES), -0.2, [0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES))
        payload = model.to_dict()
        payload["feature_names"] = list(reversed(FEATURE_NAMES))
        with self.assertRaises(ValueError):
            LinearRanker.from_dict(payload)
