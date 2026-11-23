"""Lumped heat balance and deterministic simulated sensor data."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

STEP_S = 5
HORIZON_S = 60
MAX_POWER_KW = 12
SENSOR_LIMITS_C = (0, 100)
FEATURES = (
    "temperature_c",
    "ambient_c",
    "capacity_kj_per_c",
    "loss_kw_per_c",
    "power_kw",
)
HISTORY = "recent_rise_c"


@dataclass(frozen=True)
class Vessel:
    ambient_c: float
    capacity_kj_per_c: float
    loss_kw_per_c: float

    def __post_init__(self):
        values = (self.ambient_c, self.capacity_kj_per_c, self.loss_kw_per_c)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vessel parameters must be finite")
        if self.capacity_kj_per_c <= 0 or self.loss_kw_per_c < 0:
            raise ValueError("Heat capacity must be positive and heat loss nonnegative")

    def advance(self, temperature_c, power_kw, seconds=HORIZON_S, step_s=STEP_S):
        if not math.isfinite(temperature_c) or not 0 <= power_kw <= MAX_POWER_KW:
            raise ValueError("Invalid temperature or heater power")
        if step_s <= 0 or seconds < 0 or seconds % step_s:
            raise ValueError("Duration must be a nonnegative multiple of the step")
        for _ in range(seconds // step_s):
            net_kw = power_kw - self.loss_kw_per_c * (temperature_c - self.ambient_c)
            temperature_c += step_s * net_kw / self.capacity_kj_per_c
        return temperature_c

    def features(self, temperature_c, power_kw, recent_rise_c):
        return {
            "temperature_c": temperature_c,
            "ambient_c": self.ambient_c,
            "capacity_kj_per_c": self.capacity_kj_per_c,
            "loss_kw_per_c": self.loss_kw_per_c,
            "power_kw": power_kw,
            HISTORY: recent_rise_c,
        }


def sample_vessel(rng, stress=False):
    if stress:
        return Vessel(rng.uniform(15, 28), rng.uniform(47, 76), rng.uniform(0.18, 0.26))
    return Vessel(rng.uniform(18, 26), rng.uniform(85, 150), rng.uniform(0.08, 0.17))


def generate_batches(seed=41, nominal_batches=115, stress_batches=20):
    """Generate decisions with features available when each synthetic forecast is issued."""
    rng = np.random.default_rng(seed)
    rows = []
    for batch in range(nominal_batches + stress_batches):
        stress = batch >= nominal_batches
        vessel = sample_vessel(rng, stress)
        temperature = rng.uniform(28, 46)
        previous_observation = None
        for elapsed_s in range(0, 12 * HORIZON_S, HORIZON_S):
            power = rng.uniform(0, MAX_POWER_KW)
            noise = rng.normal(0, 0.65 if stress else 0.25)
            condition = rng.choice(("normal", "missing", "delayed", "noisy"), p=(.83, .05, .05, .07))
            observed = temperature + noise
            observation_s = elapsed_s
            if condition == "missing":
                observed = np.nan
                observation_s = np.nan
            elif condition == "delayed":
                observed, observation_s = previous_observation or (np.nan, np.nan)
            elif condition == "noisy":
                observed += rng.normal(0, 2.0)

            history = np.nan
            if previous_observation is not None and observation_s == elapsed_s:
                old_value, old_time = previous_observation
                if elapsed_s - old_time == HORIZON_S:
                    history = observed - old_value

            forecast_target = vessel.advance(temperature, power)
            rows.append({
                "batch_id": batch,
                "decision_s": elapsed_s,
                "observation_s": observation_s,
                "sensor_condition": condition,
                "regime": "stress" if stress else "nominal",
                **vessel.features(observed, power, history),
                "target_c": forecast_target,
            })
            if observation_s == elapsed_s:
                previous_observation = observed, elapsed_s
            temperature = forecast_target
    return pd.DataFrame(rows)


def prepare_data(raw):
    required = {
        *FEATURES, HISTORY, "batch_id", "decision_s", "observation_s",
        "target_c", "regime", "sensor_condition",
    }
    if missing := required.difference(raw.columns):
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if raw.duplicated(["batch_id", "decision_s"]).any():
        raise ValueError("Duplicate batch decision")
    numeric = raw[[*FEATURES, "decision_s", "observation_s", "target_c"]].apply(
        pd.to_numeric, errors="coerce",
    )
    valid = np.isfinite(numeric).all(axis=1)
    valid &= numeric["observation_s"].eq(numeric["decision_s"]) & numeric["decision_s"].ge(0)
    valid &= numeric["temperature_c"].between(*SENSOR_LIMITS_C)
    valid &= numeric["capacity_kj_per_c"].gt(0) & numeric["loss_kw_per_c"].ge(0)
    valid &= numeric["power_kw"].between(0, MAX_POWER_KW)
    clean = raw.loc[valid].copy()
    clean[numeric.columns] = numeric.loc[valid]
    clean[HISTORY] = pd.to_numeric(clean[HISTORY], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )
    return clean
