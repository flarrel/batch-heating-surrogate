# Batch heating decision support

A local command-line tool for testing heater settings in a simulated stirred vessel. At each step of a batch replay, it uses the latest temperature reading to forecast the next 60 seconds and looks for a heater setting that reaches the target without exceeding the temperature ceiling. If the reading is stale or no setting passes the checks, it reports why it declined. The tool runs on a CPU and uses simulated readings throughout.

## Run a simulation

Use Python 3.12. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m batch_heat train --out runs/seed-41
python -m batch_heat replay --run runs/seed-41
python -m batch_heat evaluate --run runs/seed-41
```

- `train` generates simulated sensor data, compares forecasting models, and saves the selected model and benchmark report. Choose a new output directory for each run. Only load model artifacts from a source you trust.
- `replay` emits JSON for each reading in a recorded simulation. It shows the forecast, recommended setting or reason for declining, and the simulated operator action. The replay follows a fixed historical heater schedule. Set `--target` and `--ceiling` in °C to compare different operating limits.
- `evaluate` compares the recommendation policy with a fixed 10 kW rule on the same simulated scenarios. It reports how often each policy meets the target and ceiling, and compares energy use and time-to-target when both meet the limits.

Run the tests with `python -m unittest discover -s tests -v`.

## How it works

A heat balance advances the temperature of a well-mixed vessel in five-second steps. The forecast uses the current reading, planned heater power, ambient temperature, heat capacity, and heat loss. Training compares ridge regression, extra trees, and a small PyTorch network. Validation selects a model by mean absolute error, using batches separate from training and testing.

For a recommendation, the forecast screens settings from 0 to 12 kW. The heat balance checks the remaining settings in increasing power order, allowing for a hypothetical ±0.5 °C error in the starting reading. A setting must reach the target by 60 seconds and stay at or below the ceiling throughout. The tool declines invalid or stale readings, inputs outside the training range, and cases where no setting passes. Actual equipment would require independently validated thermal parameters and sensor error bounds.

## Reference results

The default training run generated 1,620 simulated readings from 135 batches. Across 254 held-out readings from the nominal operating range, mean absolute error was 0.392 °C for the selected ridge model, 0.393 °C for PyTorch, 0.446 °C for extra trees, and 1.903 °C for persistence. The heat-balance baseline scored 0.307 °C using the same equation as the data generator. In a separate stress test with lower heat capacity and higher heat loss, ridge MAE rose to 1.552 °C. The heat-balance baseline scored 0.500 °C.

In the default evaluation of 150 simulated cases, the tool recommended 74 settings. Of those, 72 met both limits and two missed the target. The fixed 10 kW rule met both limits in 46 cases and exceeded the ceiling in 52. In the 37 cases where both policies met the limits, the recommended settings used 0.0050 kWh less on average and took 2.3 seconds longer to reach the target.
