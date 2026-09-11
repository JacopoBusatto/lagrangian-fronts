"""The only production CLI option is the YAML configuration path."""

import argparse
import logging

from .analysis import _stage
from .config import load_config
from .workflow import run


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory-to-matrix Lagrangian flux and directional front analysis"
    )
    parser.add_argument(
        "--config", required=True, help="Analysis YAML (config_version: 1)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = logging.getLogger("lagrangian_fronts")
    handler = logging.StreamHandler()  # stderr; stdout remains the output path.
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        print(_stage("Run", run, config))
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
