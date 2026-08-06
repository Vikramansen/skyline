import importlib.util
import unittest

from skyline.model import FEATURE_NAMES


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX optional dependency is not installed")
class JaxTrainingTests(unittest.TestCase):
    def test_fitted_model_ranks_a_positive_above_a_negative(self):
        from skyline.jax_training import fit

        positive = {name: 0.0 for name in FEATURE_NAMES}
        negative = {name: 0.0 for name in FEATURE_NAMES}
        positive.update({"novelty": 1.0, "rise_rate": 2e-8, "quality": 1.0})
        negative.update({"periodicity": 1.0, "mean_magnitude_error": 0.4})
        model = fit([(positive, 1), (negative, 0), (positive, 1), (negative, 0)], epochs=80)
        self.assertGreater(model.predict(positive), model.predict(negative))
