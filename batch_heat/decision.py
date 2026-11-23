"""Local decision support for simulated batches."""

import math

import numpy as np

from .simulation import HORIZON_S, MAX_POWER_KW, SENSOR_LIMITS_C, STEP_S, sample_vessel

FIXED_POWER_KW = 10
SENSOR_MARGIN_C = 0.5
POWER_GRID = tuple(range(MAX_POWER_KW + 1))


def outcome(vessel, temperature_c, power_kw, target_c, ceiling_c):
    temperatures = [temperature_c]
    for _ in range(HORIZON_S // STEP_S):
        temperatures.append(vessel.advance(temperatures[-1], power_kw, STEP_S))
    crossing = next((i * STEP_S for i, value in enumerate(temperatures) if value >= target_c), None)
    return {
        "power_kw": power_kw,
        "energy_kwh": power_kw * HORIZON_S / 3600,
        "end_c": temperatures[-1],
        "peak_c": max(temperatures),
        "time_to_target_s": crossing,
        "feasible": temperatures[-1] >= target_c and max(temperatures) <= ceiling_c,
    }


def recommend(model, vessel, observed_c, observation_s, decision_s, history_c, target_c, ceiling_c):
    if not (
        all(math.isfinite(value) for value in (target_c, ceiling_c))
        and SENSOR_LIMITS_C[0] <= target_c < ceiling_c <= SENSOR_LIMITS_C[1]
    ):
        raise ValueError("Target and ceiling must be ordered and within the sensor range")
    if (
        observed_c is None
        or not math.isfinite(observed_c)
        or not SENSOR_LIMITS_C[0] <= observed_c <= SENSOR_LIMITS_C[1]
        or observation_s != decision_s
    ):
        return {"status": "invalid_or_stale_reading"}

    rows = [vessel.features(observed_c, power, history_c) for power in POWER_GRID]
    if not model.in_scope(rows[0]):
        return {"status": "outside_training_range"}

    predictions = model.predict(rows)
    if not np.isfinite(predictions).all():
        return {"status": "invalid_forecast"}
    fixed_prediction = float(predictions[FIXED_POWER_KW])
    checks = 0
    if observed_c <= ceiling_c:
        for power, forecast in zip(POWER_GRID, predictions, strict=True):
            if target_c <= forecast <= ceiling_c:
                checks += 1
                verified = outcome(vessel, observed_c, power, target_c, ceiling_c)
                lower_end = vessel.advance(observed_c - SENSOR_MARGIN_C, power)
                upper_start = observed_c + SENSOR_MARGIN_C
                upper_end = vessel.advance(upper_start, power)
                if (
                    verified["feasible"]
                    and lower_end >= target_c
                    and max(upper_start, upper_end) <= ceiling_c
                ):
                    return {
                        "status": "recommend",
                        "predicted_end_c": float(forecast),
                        "fixed_power_prediction_c": fixed_prediction,
                        "candidates_rechecked": checks,
                        "sensor_margin_c": SENSOR_MARGIN_C,
                        "setting": verified,
                    }
    return {
        "status": "no_feasible_setting",
        "fixed_power_prediction_c": fixed_prediction,
        "candidates_rechecked": checks,
    }


def replay(model, vessel, initial_c=36, target_c=41.5, ceiling_c=46):
    """Replay simulated readings with one delayed event and a fixed historical heater schedule."""
    temperature = initial_c
    prior_observation = None
    events = []
    for index, actual_power in enumerate((6, 8, 8, 8, 8, 8, 6, 4)):
        decision_s = index * HORIZON_S
        condition = "delayed" if index == 3 else "normal"
        observed = temperature + (0.10 if index % 2 else -0.10)
        observation_s = decision_s
        if condition == "delayed":
            observed, observation_s = prior_observation
        history = math.nan
        if prior_observation is not None and observation_s - prior_observation[1] == HORIZON_S:
            history = observed - prior_observation[0]
        decision = recommend(
            model, vessel, observed, observation_s, decision_s, history, target_c, ceiling_c,
        )
        events.append({
            "decision_s": decision_s,
            "reading_c": observed,
            "reading_s": observation_s,
            "sensor_condition": condition,
            "operator_action": "review_setting" if decision["status"] == "recommend" else "decline",
            "historical_power_kw": actual_power,
            **decision,
        })
        if observation_s == decision_s:
            prior_observation = observed, observation_s
        temperature = vessel.advance(temperature, actual_power)
    return {"source": "simulated", "target_c": target_c, "ceiling_c": ceiling_c, "events": events}


def evaluate_decisions(model, seed=9042, cases=150):
    """Compare policies on independently seeded scenarios using hidden simulator state."""
    rng = np.random.default_rng(seed)
    selection_stats = {
        "proposed": 0, "feasible": 0, "below_target": 0,
        "above_ceiling": 0, "declined": 0,
    }
    fixed_stats = {"feasible": 0, "below_target": 0, "above_ceiling": 0}
    paired_energy, paired_time, scenarios = [], [], []
    for case in range(cases):
        scenario = "delayed" if case % 15 == 0 else "noisy" if case % 7 == 0 else "normal"
        vessel = sample_vessel(rng)
        current = rng.uniform(33, 48)
        observed = current + rng.normal(0, 0.65 if scenario == "noisy" else 0.15)
        target = current + rng.uniform(1, 5)
        ceiling = target + rng.uniform(0.6, 2.8)
        observation_s = 0 if scenario == "delayed" else HORIZON_S
        decision = recommend(
            model, vessel, observed, observation_s, HORIZON_S, math.nan, target, ceiling,
        )
        selected = None
        fixed = outcome(vessel, current, FIXED_POWER_KW, target, ceiling)
        fixed_stats["feasible"] += fixed["feasible"]
        fixed_stats["below_target"] += fixed["end_c"] < target
        fixed_stats["above_ceiling"] += fixed["peak_c"] > ceiling
        if decision["status"] == "recommend":
            selection_stats["proposed"] += 1
            selected = outcome(vessel, current, decision["setting"]["power_kw"], target, ceiling)
            selection_stats["feasible"] += selected["feasible"]
            selection_stats["below_target"] += selected["end_c"] < target
            selection_stats["above_ceiling"] += selected["peak_c"] > ceiling
            if selected["feasible"] and fixed["feasible"]:
                paired_energy.append(selected["energy_kwh"] - fixed["energy_kwh"])
                paired_time.append(selected["time_to_target_s"] - fixed["time_to_target_s"])
        else:
            selection_stats["declined"] += 1
        scenarios.append({
            "id": case, "condition": scenario, "target_c": target, "ceiling_c": ceiling,
            "decision_status": decision["status"], "selected_simulator_outcome": selected,
            "fixed_simulator_outcome": fixed,
        })
    return {
        "source": "simulated",
        "seed": seed,
        "scenarios": cases,
        "surrogate_optimizer": selection_stats,
        "fixed_rule": {"power_kw": FIXED_POWER_KW, **fixed_stats},
        "sensor_margin_c": SENSOR_MARGIN_C,
        "paired_feasible_cases": len(paired_energy),
        "paired_mean_energy_delta_kwh": float(np.mean(paired_energy)) if paired_energy else None,
        "paired_mean_time_delta_s": float(np.mean(paired_time)) if paired_time else None,
        "case_details": scenarios,
    }
