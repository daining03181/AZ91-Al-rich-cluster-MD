# Supplementary Data

This folder contains the scripts and processed CSV data used for the supplementary sensitivity analyses.

The supplementary analyses are used to test whether the shear-rate-dependent trends reported in the manuscript are robust against changes in the key identification parameters for Al-rich local environments and Al-rich core clusters.

## Requirements

The supplementary scripts require:

```text
numpy
pandas
ovito
```

The scripts should be run in an OVITO-compatible Python or conda environment because they use:

```python
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder
```

## Files in this folder

| File | Type | Description |
|---|---|---|
| `figS1.py` | Script | Extracts the sensitivity-analysis data for Al-rich local-environment identification. |
| `figS2.py` | Script | Extracts the sensitivity-analysis data for Al-rich core-cluster statistics. |
| `FigS1_raw_frame_results.csv` | Data | Per-frame raw data for the Fig. S1 sensitivity analysis. |
| `FigS1_summary.csv` | Data | Averaged summary data directly used for Supplementary Fig. S1. |
| `FigS2_raw_frame_results.csv` | Data | Per-frame raw data for the Fig. S2 sensitivity analysis. |
| `FigS2_summary.csv` | Data | Averaged summary data directly used for Supplementary Fig. S2. |

## `figS1.py`

`figS1.py` extracts the sensitivity-analysis data for identifying Al-rich local environments.

Input:

```text
dump.production.Alrich*.lammpstrj
```

Analysis:

- Uses the `fs = 40%` cases.
- Reads five shear-rate cases: `0`, `0.001`, `0.005`, `0.01`, and `0.02 ps^-1`.
- Calculates the local Al enrichment factor `eta_Al` for Al atoms using OVITO neighbor searching.
- Tests the sensitivity of the Al-rich local environment fraction to different local-neighbor radii and Al-enrichment thresholds.
- Uses the last 20 frames by default.

Parameter settings:

```text
r_loc sensitivity: 4.5, 5.0, 5.5 Å, with eta_Al >= 2.0
eta_Al sensitivity: 1.8, 2.0, 2.2, with r_loc = 5.0 Å
minimum neighbor number: 6
```

Output:

```text
FigS1_raw_frame_results.csv
FigS1_summary.csv
```

## `figS2.py`

`figS2.py` extracts the sensitivity-analysis data for Al-rich core-cluster statistics.

Input:

```text
dump.production.Alrich*.lammpstrj
```

Analysis:

- Uses the `fs = 40%` cases.
- Reads five shear-rate cases: `0`, `0.001`, `0.005`, `0.01`, and `0.02 ps^-1`.
- Identifies Al-rich core atoms using the same local-environment criterion as the main analysis.
- Performs connectivity clustering of Al-rich core atoms using OVITO neighbor searching and a union-find clustering algorithm.
- Tests the sensitivity of cluster number and maximum cluster size to different connection cutoffs and minimum cluster sizes.
- Uses the last 20 frames by default.

Parameter settings:

```text
Al-rich local environment: eta_Al >= 2.0
local-neighbor radius: r_loc = 5.0 Å
minimum neighbor number: 6

r_conn sensitivity: 3.3, 3.5, 3.7 Å, with N_min = 5
N_min sensitivity: 4, 5, 6, with r_conn = 3.5 Å
```

Output:

```text
FigS2_raw_frame_results.csv
FigS2_summary.csv
```

## CSV file descriptions

### `FigS1_raw_frame_results.csv`

This file contains per-frame raw statistics for the Fig. S1 sensitivity analysis.

Main columns:

| Column | Description |
|---|---|
| `frame` | Frame index used in the production dump |
| `gdot` | Shear rate in `ps^-1` |
| `panel` | Sensitivity type, either `rloc_sensitivity` or `eta_sensitivity` |
| `parameter` | Parameter label used for the sensitivity analysis |
| `rloc` | Local-neighbor radius in Å |
| `eta_cut` | Al enrichment threshold |
| `n_al_total` | Total number of Al atoms in the frame |
| `n_rich_al` | Number of Al atoms satisfying the Al-rich local environment criterion |
| `fraction_percent` | Fraction of Al atoms in Al-rich local environments |

### `FigS1_summary.csv`

This file contains the averaged summary statistics for Supplementary Fig. S1.

Main columns:

| Column | Description |
|---|---|
| `panel` | Sensitivity type |
| `gdot` | Shear rate in `ps^-1` |
| `parameter` | Parameter label |
| `rloc` | Local-neighbor radius in Å |
| `eta_cut` | Al enrichment threshold |
| `mean_fraction_percent` | Mean fraction of Al-rich local environments |
| `std_fraction_percent` | Standard deviation of the fraction |
| `n_frames` | Number of frames used for averaging |

### `FigS2_raw_frame_results.csv`

This file contains per-frame raw cluster statistics for the Fig. S2 sensitivity analysis.

Main columns:

| Column | Description |
|---|---|
| `frame` | Frame index used in the production dump |
| `gdot` | Shear rate in `ps^-1` |
| `panel` | Sensitivity type, either `rconn_sensitivity` or `nmin_sensitivity` |
| `parameter` | Parameter label used for the sensitivity analysis |
| `r_conn` | Cluster connection cutoff in Å |
| `nmin` | Minimum cluster size threshold |
| `n_al_total` | Total number of Al atoms in the frame |
| `n_rich_al_total` | Total number of Al-rich core atoms in the frame |
| `n_clusters` | Number of Al-rich core clusters satisfying the size threshold |
| `max_cluster_size` | Maximum size of Al-rich core clusters satisfying the size threshold |

### `FigS2_summary.csv`

This file contains the averaged summary statistics for Supplementary Fig. S2.

Main columns:

| Column | Description |
|---|---|
| `panel` | Sensitivity type |
| `gdot` | Shear rate in `ps^-1` |
| `parameter` | Parameter label |
| `r_conn` | Cluster connection cutoff in Å |
| `nmin` | Minimum cluster size threshold |
| `mean_n_clusters` | Mean number of Al-rich core clusters |
| `std_n_clusters` | Standard deviation of cluster number |
| `mean_max_cluster_size` | Mean maximum cluster size |
| `std_max_cluster_size` | Standard deviation of maximum cluster size |
| `n_frames` | Number of frames used for averaging |

## Running commands

Run the scripts from the project root directory containing the simulation folders.

```bat
conda activate your_ovito_environment

python Supplementary_Data\figS1.py --root . --outdir Supplementary_Data --workers 5 --last-n-frames 20
python Supplementary_Data\figS2.py --root . --outdir Supplementary_Data --workers 5 --last-n-frames 20
```

Replace `your_ovito_environment` with the actual OVITO-compatible conda environment name.

## Notes

The default supplementary analysis settings are:

```text
solid fraction: fs = 40%
temperature: T = 852.0 K
shear rates: 0, 0.001, 0.005, 0.01, 0.02 ps^-1
number of frames used: last 20 frames
```

These supplementary data support the robustness checks of the identification parameters used in the manuscript:

- Supplementary Fig. S1 tests the sensitivity to `r_loc` and `eta_Al`.
- Supplementary Fig. S2 tests the sensitivity to `r_conn` and `N_min`.

The intended workflow is:

```text
Raw LAMMPS dump files -> Supplementary_Data/*.py -> Supplementary_Data/*.csv -> Supplementary figures
```
