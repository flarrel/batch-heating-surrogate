import math
import unittest

from pandas.testing import assert_frame_equal

from batch_heat.simulation import Vessel, generate_batches, prepare_data


class SimulationTest(unittest.TestCase):
    def test_energy_balance_and_convergence(self):
        vessel = Vessel(20, 100, 0.1)
        self.assertGreater(vessel.advance(20, 8), 20)
        self.assertLess(vessel.advance(45, 0), 45)
        exact = 100 + (40 - 100) * math.exp(-0.1 * 60 / 100)
        errors = [abs(vessel.advance(40, 8, step_s=step) - exact) for step in (10, 5, 1)]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_data_are_seeded_and_reject_invalid_readings(self):
        data = generate_batches(nominal_batches=4, stress_batches=1)
        assert_frame_equal(data, generate_batches(nominal_batches=4, stress_batches=1))
        data.loc[0, "observation_s"] = data.loc[0, "decision_s"] + 5
        data.loc[1, "temperature_c"] = float("inf")
        data.loc[2, "power_kw"] = -1
        clean = prepare_data(data)
        self.assertTrue({0, 1, 2}.isdisjoint(clean.index))
        self.assertTrue(clean.observation_s.eq(clean.decision_s).all())
        self.assertIn("stress", set(clean.regime))
        with self.assertRaises(ValueError):
            prepare_data(data.drop(columns="target_c"))
