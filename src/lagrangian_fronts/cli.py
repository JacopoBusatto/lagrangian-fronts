"""The only production CLI option is the YAML configuration path."""

import argparse

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
    print(run(load_config(args.config)))
