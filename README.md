# Lagrangian fronts

Turn prepared trajectories into a sparse transition matrix, then identify **Lagrangian displacement flux** cores/fronts and complementary **directional** cores/fronts. You can also start with an existing transition matrix. One YAML file controls the complete workflow.

```text
trajectories → transition matrix → cell statistics
                                    ├─ flux ridge → flux cores → flux fronts
                                    └─ directional candidates → ridge → plateau thinning
                                                                  → cores → fronts
```

The package supports geographic longitude/latitude and Cartesian x/y coordinates. Trajectory readers accept CSV, Parquet, NetCDF and Zarr, including multiple files and globs. Matrices and analysis tables use Parquet. Scientific trajectory preparation and QC must happen upstream.

## Install locally

Use Python 3.10 or newer. From the downloaded or copied `lagrangian-fronts` project directory, create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install .
```

This installs the required scientific libraries and the `lagrangian-fronts` command. No kinematicParcels installation, Git checkout, or `PYTHONPATH` adjustment is needed to run it. Version 0.1.0 supports the validated NumPy 1.x, pandas 2.x and Zarr 2.x interfaces; their major-version bounds are declared in `pyproject.toml`. Other dependencies use minimum versions, not exact pins. Manifests record the versions actually used.

The repository URL is a maintainer placeholder (`<repository-url>`); no public remote is configured. This software has not been published to PyPI. Install from the source directory or the supplied wheel rather than assuming an online release exists.

## Run the included examples

From the project directory after installation:

```bash
lagrangian-fronts --config examples/geographic/config.yaml
```
or

```bash
python -m lagrangian_fronts --config examples/cartesian/config.yaml
```

These small, offline examples include CSV data and exercise actual cores, fronts, validation and seven figures. They use synthetic stay/move profiles, not observed currents. Outputs appear under `outputs/geographic/` and `outputs/cartesian/`.

Reuse the generated matrices with the included load configurations:

```bash
lagrangian-fronts --config examples/geographic/load.yaml
python -m lagrangian_fronts --config examples/cartesian/load.yaml
```

The entry points are equivalent. Both accept only `--config`; scientific options belong in YAML. **Existing run directories are rejected.** Choose a fresh `output.run_name` when rerunning an analysis.

From any other working directory, pass an absolute or appropriate relative path to the YAML. Data and output paths inside it resolve relative to the YAML's location. Keep the example CSV and YAML together when copying an example elsewhere. Example figures disable coastline downloads; enabling coastlines may require Cartopy's external map data.

`examples/production/argo.yaml` and `drifters.yaml` preserve the source implementation's production thresholds. They are templates for your own local matrices: supply the indicated Parquet input or edit `matrix.path` and `output`. No external ARGO/drifter data are distributed.

## Configure your analysis

Keep `config_version: 1`. Use `matrix.compute: true` with an `input` section for trajectories, or `matrix.compute: false` with `matrix.path` and explicit timing to load a matrix. Set the coordinate system and grid explicitly. The selection settings use `flux:` and `directional:`; the earlier `transport:` section is rejected with a migration explanation.

Lagrangian displacement flux is a probability-weighted displacement rate with units of length/time. Directional strength is the dimensionless product `P_move * R1`. There is no directional percentile. The [methodology](docs/METHODOLOGY.md) explains both pathways, interpolation, row normalization and integrated ridge strength.

Each successful run contains `matrix/`, `analysis/`, `resolved_config.yaml` and `manifest.json`. Analysis includes `flux_cores.parquet`, `flux_fronts.parquet`, directional tables and comparisons; figures are in `analysis/figures/`. Validation and detailed component/section diagnostics retain their YAML switches. Load mode references the original matrix instead of copying it. Git metadata is optional.

The CLI automatically reports stages and elapsed times on stderr, with a live tqdm bar inside slow front-analysis and validation loops. Each bar shows completed items and a local ETA; flux profile composites count segment groups, while other front loops count sections. Redirected output keeps plain stage messages, and ordinary Python calls remain silent. Stdout contains the final output path. No progress setting is needed. After updating, run `python -m pip install -e .` to install the tqdm dependency.

## Python API

```python
from lagrangian_fronts import (
    load_config, read_trajectories, compute_transition_matrix,
    analyze_transition_matrix, run,
)

config = load_config("examples/cartesian/config.yaml")
trajectories = read_trajectories(config)
matrix = compute_transition_matrix(trajectories, config)
analysis = analyze_transition_matrix(matrix.transition_table, config)
flux_cores = analysis.flux_cores.cores
flux_fronts = analysis.flux_fronts.fronts

# Execute the complete configured workflow and write a fresh run directory:
run_directory = run(config)
```

`read_transition_matrix(path, config)` supplies the dataframe interface for existing matrices. `analyze_transition_matrix` performs no file reads, writes or plotting. Compute mode passes the matrix directly to analysis in memory.

## Documentation and development

- [Input formats and canonical trajectories](docs/INPUT_DATA.md)
- [Scientific methodology](docs/METHODOLOGY.md)
- [Every YAML parameter and the output contract](docs/CONFIGURATION.md)
- [Extraction and verification report](docs/EXTRACTION_REPORT.md)

For development, install the package in editable mode with the test tools:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
python -m build
```

The source archive includes examples, documentation, tests and scientific reference fixtures. The wheel installs the analysis package and CLI; use your own YAML/data or examples from the source archive. The auxiliary `python -m lagrangian_fronts.plot_flux_cores_hydrography` remains a separate visualization tool requiring user-supplied hydrographic and reference-front data.

## Origin, license and citation

This is the first standalone packaging of the validated Lagrangian fronts implementation developed in kinematicParcels, including its flux terminology and directional ridge/plateau analysis. Packaging does not change the scientific method. The historical scientific reference is commit `10e63b4675a6b6d3f9eb1746bc9ecb6b04bf0ed5`; extraction is checked against the subsequent validated working implementation.

The [MIT license](LICENSE) and Jacopo Busatto copyright notice are preserved unchanged. [CITATION.cff](CITATION.cff) records supported software metadata. A paper citation, DOI and repository URL can be added by the maintainer when available.
