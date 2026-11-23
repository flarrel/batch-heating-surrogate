"""CPU regression, grouped evaluation, and local run artifacts."""

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .simulation import FEATURES, HISTORY, HORIZON_S, Vessel, generate_batches, prepare_data


class TorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, width=32, seed=41):
        self.width = width
        self.seed = seed

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x = torch.as_tensor(x, dtype=torch.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = float(y.mean())
        self.scale_ = float(y.std()) or 1.0
        y = torch.as_tensor((y - self.mean_) / self.scale_).reshape(-1, 1)
        self.network_ = nn.Sequential(
            nn.Linear(x.shape[1], self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.width),
            nn.Tanh(),
            nn.Linear(self.width, 1),
        )
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=0.005)
        for _ in range(65):
            for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // 128)):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.network_(x[indices]), y[indices])
                loss.backward()
                optimizer.step()
        self.network_.eval()
        return self

    def predict(self, x):
        with torch.inference_mode():
            values = self.network_(torch.as_tensor(x, dtype=torch.float32))
            return values.numpy().ravel() * self.scale_ + self.mean_


@dataclass
class Surrogate:
    estimator: object
    features: tuple
    bounds: dict

    def predict(self, rows):
        x = np.asarray([[row[name] for name in self.features] for row in rows], dtype=float)
        return self.estimator.predict(x)

    def in_scope(self, row):
        return all(low <= row[name] <= high for name, (low, high) in self.bounds.items())


def split_batches(data, seed):
    nominal = data[data.regime.eq("nominal")]
    stress = data[data.regime.eq("stress")]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    remaining, test = next(outer.split(nominal, groups=nominal.batch_id))
    training_pool = nominal.iloc[remaining]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train, validation = next(inner.split(training_pool, groups=training_pool.batch_id))
    return {
        "train": training_pool.iloc[train],
        "validation": training_pool.iloc[validation],
        "test": nominal.iloc[test],
        "stress": stress,
    }


def metrics(target, predicted):
    error = np.asarray(predicted) - np.asarray(target)
    return {
        "rows": len(error),
        "mae_c": float(np.abs(error).mean()),
        "rmse_c": float(np.sqrt(np.square(error).mean())),
    }


def fit_surrogate(family, value, train, features, seed):
    models = {
        "ridge": lambda: Ridge(alpha=value),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=48, min_samples_leaf=value, random_state=seed, n_jobs=1,
        ),
        "torch": lambda: TorchRegressor(width=value, seed=seed),
    }
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        models[family](),
    )
    pipeline.fit(train[list(features)].to_numpy(dtype=float), train.target_c.to_numpy())
    # Current power is a user-chosen input with an explicit physical limit.
    bounds = {
        name: (float(train[name].min()), float(train[name].max()))
        for name in FEATURES if name != "power_kw"
    }
    return Surrogate(pipeline, features, bounds)


def predict_frame(model, frame):
    return model.predict(frame[list(model.features)].to_dict("records"))


def summarize(data):
    return {
        "rows_by_regime": {str(key): int(value) for key, value in data.regime.value_counts().items()},
        "rows_by_sensor_condition": {
            str(key): int(value) for key, value in data.sensor_condition.value_counts().items()
        },
        "missing_temperature": int(data.temperature_c.isna().sum()),
        "stale_readings": int(data.observation_s.lt(data.decision_s).sum()),
        "ranges_and_medians": {
            name: {
                "min": float(data[name].min()),
                "median": float(data[name].median()),
                "max": float(data[name].max()),
            }
            for name in ("temperature_c", "power_kw", "target_c")
        },
    }


def physics_baseline(frame):
    return [
        Vessel(row.ambient_c, row.capacity_kj_per_c, row.loss_kw_per_c).advance(
            row.temperature_c, row.power_kw,
        )
        for row in frame.itertuples(index=False)
    ]


def train_run(output: Path, seed=41):
    torch.set_num_threads(1)
    raw = generate_batches(seed)
    data = prepare_data(raw)
    splits = split_batches(data, seed)
    train, validation = splits["train"], splits["validation"]
    trials, family_winners = [], {}
    for family, values in (("ridge", (0.1, 10)), ("extra_trees", (2, 8)), ("torch", (16, 48))):
        for value in values:
            name = f"{family}_{value}"
            model = fit_surrogate(family, value, train, FEATURES, seed)
            result = metrics(validation.target_c, predict_frame(model, validation))
            trials.append({"name": name, "family": family, "parameter": value, **result})
            if family not in family_winners or result["mae_c"] < family_winners[family][0]:
                family_winners[family] = (result["mae_c"], name, model, value)

    _, base_name, base_model, value = min(family_winners.values(), key=lambda trial: trial[0])
    base_family = next(family for family, trial in family_winners.items() if trial[1] == base_name)
    validation_slices = {
        str(condition): metrics(frame.target_c, predict_frame(base_model, frame))
        for condition, frame in validation.groupby("sensor_condition", sort=True)
    }
    worst = max(validation_slices, key=lambda key: validation_slices[key]["mae_c"])
    ablation = fit_surrogate(base_family, value, train, (*FEATURES, HISTORY), seed)
    ablation_result = metrics(validation.target_c, predict_frame(ablation, validation))
    worst_frame = validation[validation.sensor_condition.eq(worst)]
    ablation_name = f"{base_name}_with_history"
    selected_name, selected = base_name, base_model
    if ablation_result["mae_c"] < metrics(validation.target_c, predict_frame(base_model, validation))["mae_c"]:
        selected_name, selected = ablation_name, ablation

    # Model selection is complete before test or stress results are computed.
    compared = {trial[1]: trial[2] for trial in family_winners.values()}
    compared[ablation_name] = ablation
    results = {}
    for partition in ("test", "stress"):
        frame = splits[partition]
        results[partition] = {
            "persistence": metrics(frame.target_c, frame.temperature_c),
            "physics_from_sensor": metrics(frame.target_c, physics_baseline(frame)),
            **{
                name: metrics(frame.target_c, predict_frame(model, frame))
                for name, model in compared.items()
            },
        }
    test_slices = {
        str(condition): metrics(frame.target_c, predict_frame(selected, frame))
        for condition, frame in splits["test"].groupby("sensor_condition", sort=True)
    }
    sample = splits["test"].iloc[[0]][list(selected.features)].to_dict("records")
    selected.predict(sample)
    durations = []
    for _ in range(100):
        start = perf_counter()
        selected.predict(sample)
        durations.append((perf_counter() - start) * 1_000)

    report = {
        "artifact_version": 1,
        "source": "simulated",
        "seed": seed,
        "horizon_s": HORIZON_S,
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch")},
        "exploration": summarize(raw),
        "rejected_rows": len(raw) - len(data),
        "split_batch_ids": {
            name: sorted(map(int, frame.batch_id.unique())) for name, frame in splits.items()
        },
        "validation_trials": trials,
        "ablation": {
            "feature": HISTORY,
            "rationale": "Prior sensor movement may help identify an anomalous current reading.",
            "targeted_validation_slice": worst,
            "base_validation_slices": validation_slices,
            "candidate": ablation_name,
            "overall_validation": ablation_result,
            "targeted_validation_result": metrics(worst_frame.target_c, predict_frame(ablation, worst_frame)),
        },
        "selected_model": selected_name,
        "selected_features": list(selected.features),
        "in_scope_training_bounds": selected.bounds,
        "results": results,
        "selected_test_slices": test_slices,
        "inference_ms": {
            "samples": len(durations),
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
        },
    }
    raw.to_csv(output / "synthetic_batches.csv", index=False)
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    joblib.dump(selected, output / "surrogate.joblib")
    return report
