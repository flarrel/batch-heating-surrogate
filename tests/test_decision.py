import math
import unittest

from batch_heat.decision import evaluate_decisions, recommend, replay
from batch_heat.simulation import Vessel


class ThermalForecast:
    def __init__(self, vessel):
        self.vessel = vessel

    def predict(self, rows):
        return [
            Vessel(row["ambient_c"], row["capacity_kj_per_c"], row["loss_kw_per_c"]).advance(
                row["temperature_c"], row["power_kw"],
            )
            for row in rows
        ]

    def in_scope(self, row):
        return 0 < row["temperature_c"] < 70


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.vessel = Vessel(22, 100, 0.12)
        self.model = ThermalForecast(self.vessel)

    def decide(self, temperature, target, ceiling, reading_s=60):
        return recommend(
            self.model, self.vessel, temperature, reading_s, 60, math.nan, target, ceiling,
        )

    def test_minimum_feasible_setting_is_verified(self):
        result = self.decide(38, 41, 46)
        self.assertEqual(result["status"], "recommend")
        self.assertEqual(result["setting"]["power_kw"], 8)
        self.assertTrue(result["setting"]["feasible"])
        self.assertGreaterEqual(result["setting"]["end_c"], 41)

    def test_declines_when_missing_delayed_out_of_range_or_infeasible(self):
        for temperature, target, ceiling, reading_s, expected in (
            (None, 41, 46, 60, "invalid_or_stale_reading"),
            (38, 41, 46, 0, "invalid_or_stale_reading"),
            (75, 78, 85, 60, "outside_training_range"),
            (30, 45, 48, 60, "no_feasible_setting"),
        ):
            with self.subTest(expected=expected):
                self.assertEqual(self.decide(temperature, target, ceiling, reading_s)["status"], expected)
        with self.assertRaises(ValueError):
            self.decide(38, 46, 41)

    def test_replay_and_policy_accounting(self):
        events = replay(self.model, self.vessel)["events"]
        self.assertEqual(events[3]["status"], "invalid_or_stale_reading")
        first, second = (event["fixed_power_prediction_c"] for event in events[:2])
        self.assertNotEqual(first, second)
        report = evaluate_decisions(self.model, cases=30)
        stats = report["surrogate_optimizer"]
        self.assertEqual(stats["proposed"] + stats["declined"], 30)
        self.assertLessEqual(report["paired_feasible_cases"], stats["feasible"])
