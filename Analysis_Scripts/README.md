# Analysis Scripts

This folder contains the Python scripts used to extract the processed CSV data for the manuscript figures.  
The scripts are used for data extraction and tabulation only. They do not regenerate the final manuscript figures.

## Requirements

Most scripts require:

```text
numpy
pandas
```

## Script descriptions

### `fig4.py`

Extracts no-shear Al-rich statistics for different solid fractions.

Input:

```text
dump.production.Alrich*.lammpstrj
```

Analysis:

- Reads the no-shear cases with `gdot = 0`.
- Uses the last 20% of production frames.
- Counts Al-rich core atoms, Al-rich local environments, strong Al-rich local environments, Al-rich core clusters, and maximum cluster size.

Output:

```text
Fig4_noshear_statistics_last20_summary.csv
Fig4_noshear_statistics_last20_frames.csv
```

### `fig6.py`

Extracts shear-rate-dependent Al-rich statistics for all solid fractions and shear rates.

Input:

```text
dump.production.Alrich*.lammpstrj
```

Analysis:

- Reads all 20 cases, including four solid fractions and five shear rates.
- Uses the last 20% of production frames.
- Calculates the mean and standard deviation of Al-rich core atoms, `eta_Al >= 2.0`, `eta_Al >= 3.0`, cluster number, and maximum cluster size.

Output:

```text
Fig6_shear_statistics_last20_summary.csv
Fig6_shear_statistics_last20_frames.csv
```

### `fig7.py`

Calculates relative shear response and response-correlation data from Fig. 6 data.

Input:

```text
Fig6_shear_statistics_last20_summary.csv
```

Analysis:

- Uses the no-shear case at the same solid fraction as the baseline.
- Calculates the relative change of Al-rich core atoms, strong Al-rich local environments, cluster number, and maximum cluster size.
- Generates the reduced dataset used for the Fig. 7 response-correlation plots.

Output:

```text
Fig7_relative_shear_response_data.csv
Fig7_response_correlation.csv
```

### `fig8.py`

Extracts structural characteristics of Al-rich local environments.

Input:

```text
stat.production*.dat
```

Analysis:

- Uses the last 20% of production records.
- Calculates the fraction of disordered or PTM-unknown atoms in Al-rich local environments.
- Calculates the enrichment ratio of local disorder relative to the whole system.

Output:

```text
Fig8_structural_characteristics.csv
```

### `fig9.py`

Extracts energy characteristics of Al-rich and matrix regions.

Input:

```text
stat.production*.dat
```

Analysis:

- Uses the last 20% of production records.
- Extracts the average potential energy of Al-rich regions and matrix regions.
- Calculates the energy difference between Al-rich and matrix regions.
- Stores the relative energy response with respect to the no-shear baseline.

Output:

```text
Fig9_energy_characteristics.csv
```

### `fig10.py`

Extracts local Al-Al correlation data from RDF files.

Input:

```text
rdf.production*.dat
```

Analysis:

- Reads the last RDF block of each case.
- Extracts Al-Al and Mg-Al coordination numbers at `r = 3.8 Å`.
- Extracts the Al-Al RDF first peak intensity in the range `2.5–3.6 Å`.
- Combines the coordination and peak information into the final Fig. 10 dataset.

Output:

```text
Fig10_local_correlation.csv
Fig10_rdf_coordination_summary.csv
Fig10_rdf_peak_summary.csv
```

### `fig11.py`

Extracts Al diffusion coefficients under shear.

Input:

```text
msd.production.components*.dat
dump.production.Alrich*.lammpstrj
```

Analysis:

- Calculates the transverse Al diffusion coefficient from `MSD_x + MSD_z`.
- Calculates the desheared three-dimensional Al diffusion coefficient after removing the affine shear contribution.
- Calculates diffusion enhancement factors using the no-shear case at the same solid fraction as the baseline.

Output:

```text
Fig11_al_diffusion_coefficients.csv
```

## Running order

Run the scripts from the project root directory containing the simulation folders.

```bat
python Analysis_Scripts\fig4.py
python Analysis_Scripts\fig6.py
python Analysis_Scripts\fig7.py
python Analysis_Scripts\fig8.py

python Analysis_Scripts\fig9.py --root . --out Processed_Data\Fig9_energy_characteristics.csv
python Analysis_Scripts\fig10.py --root . --out Processed_Data\Fig10_local_correlation.csv --save-intermediate
python Analysis_Scripts\fig11.py --root . --out Processed_Data\Fig11_al_diffusion_coefficients.csv
```

`fig7.py` should be run after `fig6.py`, because it uses `Fig6_shear_statistics_last20_summary.csv` as input.

## Notes

The default analysis settings are consistent with the manuscript:

- Main statistics are averaged over the last 20% of production frames.
- Al-rich local environment: `eta_Al >= 2.0`.
- Strong Al-rich local environment: `eta_Al >= 3.0`.
- Minimum neighbor number: `MIN_NEIGH = 6`.
- Main Al-rich core-cluster threshold: `N_min = 5`.
- Fig. 10 uses RDF data to quantify local Al-Al correlations.
- Fig. 11 reports `D_perp_Al` and `D3D_deshear_Al`.

The intended workflow is:

```text
Raw LAMMPS output -> Analysis_Scripts/*.py -> Processed_Data/*.csv -> Manuscript figures
```
