"""Command-line access to simulation, training, and local batch replay."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import torch

from .decision import evaluate_decisions, replay
from .model import train_run
from .simulation import Vessel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run a fresh benchmark and save local artifacts")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=41)
    demo = commands.add_parser("replay", help="Replay a simulated batch")
    demo.add_argument("--run", type=Path, required=True)
    demo.add_argument("--target", type=float, default=41.5)
    demo.add_argument("--ceiling", type=float, default=46)
    evaluation = commands.add_parser("evaluate", help="Compare the saved model against a fixed policy")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, default=9042)
    args = parser.parse_args(argv)

    if args.command == "train":
        if args.out.exists():
            parser.error("Choose a new output directory for each run")
        if not 0 <= args.seed < 2**32 - 1:
            parser.error("Seed must be between 0 and 4294967294")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".batch-heat-", dir=args.out.parent) as staging:
            artifacts = Path(staging) / "artifacts"
            artifacts.mkdir()
            report = train_run(artifacts, args.seed)
            artifacts.rename(args.out)
        result = {"output": str(args.out), "selected_model": report["selected_model"]}
    else:
        torch.set_num_threads(1)
        try:
            model = joblib.load(args.run / "surrogate.joblib")
            if args.command == "replay":
                result = replay(model, Vessel(22, 100, 0.12), target_c=args.target, ceiling_c=args.ceiling)
            else:
                result = evaluate_decisions(model, seed=args.seed)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
