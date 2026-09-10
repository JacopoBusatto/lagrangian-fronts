"""Numerical references captured from scientific baseline 10e63b4."""

import json
from dataclasses import fields
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from lagrangian_fronts.config import load_config


# Captures retain their original names. Only these reviewed names may change;
# an unexpected column, label, number, dtype or ordering still fails comparison.
REFERENCE_GROUPS = {
    "flux_cores": "transport_cores",
    "flux_fronts": "transport_fronts",
}
REFERENCE_NAMES = {
    "S_transport": "S_flux",
    "S_transport_rate": "S_flux_rate",
    "absolute_transport_loss": "absolute_flux_loss",
    "relative_transport_loss": "relative_flux_loss",
    "transport_loss_rate": "flux_loss_rate",
    "integrated_transport_area_rate": "integrated_ridge_strength_area_rate",
    "rank_integrated_transport": "rank_integrated_ridge_strength",
    "rank_median_transport": "rank_median_flux",
    "opposing_outer_transport": "opposing_outer_flux",
    "transport_component_id": "flux_component_id",
    "transport_core": "flux_core",
    "transport_and_directional": "flux_and_directional",
    "transport_only": "flux_only",
    "transport_and_directional_fraction": "flux_and_directional_fraction",
    "transport_only_fraction": "flux_only_fraction",
    "global_transport_defined_cells": "global_flux_defined_cells",
    "spearman_transport_loss_vs_abs_G_perp": "spearman_flux_loss_vs_abs_G_perp",
    "transport_cores.threshold_rate": "flux_cores.threshold_rate",
    "transport_cores.selection_label": "flux_cores.selection_label",
    "transport_fronts.experiment_id": "flux_fronts.experiment_id",
}
REFERENCE_LABELS = {
    "transport": "flux",
    "transport_and_directional": "flux_and_directional",
    "transport_only": "flux_only",
    "probable_transport_front": "probable_flux_front",
    "no_transport_core_sections": "no_flux_core_sections",
}
REFERENCE_FLAGS = {
    "opposing_outer_transport": "opposing_outer_flux",
    "transport_undefined": "flux_undefined",
}


def _reference_metadata(value):
    if isinstance(value, dict):
        return {
            REFERENCE_NAMES.get(key, key): _reference_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_reference_metadata(item) for item in value]
    if isinstance(value, str):
        return REFERENCE_LABELS.get(value, value)
    return value


def _reference_table(table):
    table = table.rename(columns=REFERENCE_NAMES)
    for column in table.select_dtypes(include=["object", "string", "category"]):
        if column in {"quality_flags", "gradient_quality_flags"}:
            mapping = {
                value: ";".join(REFERENCE_FLAGS.get(flag, flag) for flag in value.split(";"))
                for value in table[column] if isinstance(value, str)
            }
        else:
            mapping = REFERENCE_LABELS
        # Avoid touching unrelated object columns (including baseline bools).
        if any(isinstance(value, str) and mapping.get(value, value) != value
               for value in table[column]):
            table[column] = table[column].map(
                lambda value: mapping.get(value, value) if isinstance(value, str) else value
            )
    return table


def analyze(table, config):
    from lagrangian_fronts.analysis import analyze_transition_matrix

    result = analyze_transition_matrix(table, config)
    return {item.name: getattr(result, item.name) for item in fields(result)}


def assert_reference(directory, results):
    metadata = {}
    for group, result in results.items():
        for item in fields(result):
            value = getattr(result, item.name)
            key = group + "." + item.name
            if isinstance(value, pd.DataFrame):
                reference_key = REFERENCE_GROUPS.get(group, group) + "." + item.name
                expected = _reference_table(
                    pd.read_parquet(directory / (reference_key + ".parquet"))
                )
                # Compare at the same serialization boundary as the captured reference.
                # Parquet infers bool dtype for some baseline object-of-bool columns.
                buffer = BytesIO()
                value.to_parquet(buffer, index=True)
                buffer.seek(0)
                pd.testing.assert_frame_equal(
                    pd.read_parquet(buffer), expected, check_exact=True, obj=key
                )
            else:
                metadata[key] = value
    expected = _reference_metadata(
        json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    )
    assert json.dumps(
        metadata, sort_keys=True, default=lambda v: v.item()
    ) == json.dumps(expected, sort_keys=True)


@pytest.mark.parametrize("mode", ["cartesian", "geographic"])
def test_complete_scientific_reference(mode, tmp_path):
    directory = Path(__file__).parent / "fixtures" / mode
    raw = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    raw["flux"] = raw.pop("transport")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    results = analyze(pd.read_parquet(directory / "matrix.parquet"), config)
    assert_reference(directory, results)
    assert not results["flux_cores"].cores.empty
    assert results["flux_fronts"].fronts.front_detected.any()
    assert not results["directional_cores"].cores.empty
    assert results["directional_fronts"].fronts.front_detected.any()
