import unittest
from itertools import combinations

import numpy as np
import torch

from batch_heat.model import fit_surrogate, predict_frame, split_batches
from batch_heat.simulation import FEATURES, generate_batches, prepare_data


class ModelTest(unittest.TestCase):
    def test_batch_split_and_trainable_models(self):
        torch.set_num_threads(1)
        data = prepare_data(generate_batches(nominal_batches=20, stress_batches=3))
        splits = split_batches(data, seed=41)
        for first, second in combinations(splits.values(), 2):
            self.assertTrue(set(first.batch_id).isdisjoint(second.batch_id))
        self.assertEqual(set(splits["stress"].regime), {"stress"})
        for family, value in (("ridge", 0.1), ("extra_trees", 2), ("torch", 16)):
            with self.subTest(family=family):
                model = fit_surrogate(family, value, splits["train"], FEATURES, 41)
                predictions = predict_frame(model, splits["test"])
                self.assertEqual(len(predictions), len(splits["test"]))
                self.assertTrue(np.isfinite(predictions).all())
