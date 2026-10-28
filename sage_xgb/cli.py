"""Command line entry point: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from sage_xgb import __version__, pipeline
from sage_xgb.artifacts import PIPELINE_STAGES
from sage_xgb.config import ConfigError, load_config


STAGE_COMMANDS: dict[str, str] = {
    "download": "Download the raw AMLworld files from Kaggle",
    "prepare": "Sort transactions and write the split table",
    "features": "Build the account window feature table",
    "gnn": "Fit the cross-fit GNNs and write out-of-fold scores",
    "train-xgb": "Train XGBoost and write the model bundle",
    "report": "Score test and Tail once and write the report",
}


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, for example gnn.epochs=2",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per stage."""
    parser = argparse.ArgumentParser(
        prog="sage-xgb", description="GNN output into XGBoost for AMLworld."
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in STAGE_COMMANDS.items():
        _add_config_arguments(subcommands.add_parser(name, help=help_text))
    run = subcommands.add_parser(
        "run", help="Run prepare, features, gnn and train-xgb with skipping"
    )
    _add_config_arguments(run)
    run.add_argument("--force", action="store_true", help="rerun every stage")
    run.add_argument(
        "--from-stage",
        choices=PIPELINE_STAGES,
        help="rerun this stage and every stage after it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config, args.overrides)
    except ConfigError as error:
        sys.stderr.write(f"config error: {error}\n")
        return 2
    if args.command == "run":
        pipeline.run(config, force=args.force, from_stage=args.from_stage)
    else:
        pipeline.execute_stage(config, args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
