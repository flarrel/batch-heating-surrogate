# Batch heating decision support

Forecast temperature in a simulated stirred vessel and compare bounded heater settings with a fixed operating rule. The tool is intended for local process-modeling experiments on a CPU.

## Run locally

Use Python 3.12. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m batch_heat train --out runs/seed-41
python -m batch_heat replay --run runs/seed-41
python -m batch_heat evaluate --run runs/seed-41
```

Use a new output directory for each training run. Training creates simulated sensor data, a benchmark report, and a saved surrogate. Load model artifacts only from a run you trust. Replay returns JSON with each reading, forecast, recommendation or decline reason, and simulated operator action. The historical heater schedule stays fixed during replay. To exercise an infeasible target, add `--target 95 --ceiling 98` to the replay command.

## Model card

- **Forecast.** Predict temperature 60 seconds ahead from the current reading, planned heater power, ambient temperature, estimated heat capacity, and heat loss. The heat balance integrates net power in 5-second steps. Sensor readings with missing, invalid, or stale values are rejected.
- **Training and selection.** Compare ridge, extra trees, and a small PyTorch network. Split nominal data by batch before fitting. Use validation MAE to select the model. Report results on held-out nominal batches and a separate stress set with lower thermal capacity and higher heat loss.
- **Decision rule.** Screen settings from 0 to 12 kW with the learned forecast. Check surviving settings in increasing power order with the heat balance, allowing a hypothetical ±0.5 °C reading error. A setting must reach the target by 60 seconds and stay below the ceiling across that interval. Decline if the reading is invalid or stale, the inputs fall outside the trained range, or no setting passes.
- **Operating limits.** The simulator assumes a well-mixed vessel and known thermal parameters. The stress results show sensitivity to extrapolation. The sensor margin is an assumption, and larger errors can still make a recommendation infeasible. The tool makes recommendations for simulated experiments and has no connection to plant equipment.

## Reference run

Seed 41 generated 1,620 readings across 135 simulated batches. Preprocessing rejected 146 rows. The 60-second forecast used 69 training, 23 validation, 23 nominal test, and 20 stress batches with disjoint IDs. Two runs produced the same data and forecast results.

- **Nominal test MAE, 254 rows.** Physics estimate 0.307 °C, selected ridge 0.392 °C, PyTorch 0.393 °C, extra trees 0.446 °C, and persistence 1.903 °C. The physics estimate uses the same heat-balance equation as the simulator.
- **Stress test MAE, 221 rows.** Ridge 1.552 °C and the physics estimate 0.500 °C. The recent-temperature-change feature improved the noisy validation slice from 1.776 to 1.670 °C MAE, but worsened overall validation MAE, so it was left out of the selected model.
- **Setting comparison, 150 fresh simulated cases at seed 9042.** The optimizer proposed 74 settings and declined 76. Of the proposals, 72 met both constraints, two missed the target, and zero exceeded the ceiling when evaluated against the underlying simulated temperature. The fixed 10 kW rule met both constraints in 46 cases and exceeded the ceiling in 52. In the 37 cases where both policies were feasible, the optimizer used 0.0050 kWh less on average and reached the target 2.3 seconds later.
