# Processed Data

This folder contains the processed CSV data used for the main manuscript figures.  
These files were generated from the raw LAMMPS output files using the scripts in `Analysis_Scripts`.

## File descriptions

| File | Description | Related figure |
|---|---|---|
| `Fig4_noshear_statistics_last20_summary.csv` | Summary statistics for the no-shear cases at different solid fractions. The data are averaged over the last 20% of production frames. | Fig. 4 |
| `Fig4_noshear_statistics_last20_frames.csv` | Per-frame statistics used to calculate `Fig4_noshear_statistics_last20_summary.csv`. | Fig. 4 |
| `Fig6_shear_statistics_last20_summary.csv` | Summary statistics for all shear cases, including Al-rich core atoms, Al-rich local environments, strong Al-rich local environments, cluster number, and maximum cluster size. The data are averaged over the last 20% of production frames. | Fig. 6 |
| `Fig6_shear_statistics_last20_frames.csv` | Per-frame statistics used to calculate `Fig6_shear_statistics_last20_summary.csv`. | Fig. 6 |
| `Fig7_relative_shear_response_data.csv` | Relative shear-response data calculated from `Fig6_shear_statistics_last20_summary.csv`, using the no-shear case at the same solid fraction as the baseline. | Fig. 7 |
| `Fig7_response_correlation.csv` | Reduced response-correlation data directly used for Fig. 7. | Fig. 7 |
| `Fig8_structural_characteristics.csv` | Structural characteristics of Al-rich local environments, including the fraction and enrichment ratio of locally disordered atoms. | Fig. 8 |
| `Fig9_energy_characteristics.csv` | Energy characteristics of Al-rich and matrix regions, including average potential energies and the energy difference between these regions. | Fig. 9 |
| `Fig10_local_correlation.csv` | Combined local Al-Al correlation data, including Al-Al coordination, Mg-Al coordination, their ratio, and Al-Al RDF peak intensity. | Fig. 10 |
| `Fig10_rdf_coordination_summary.csv` | Intermediate RDF coordination-number data extracted at `r = 3.8 Å`. | Fig. 10 |
| `Fig10_rdf_peak_summary.csv` | Intermediate RDF first-peak data for Al-Al correlations in the range `2.5–3.6 Å`. | Fig. 10 |
| `Fig11_al_diffusion_coefficients.csv` | Al diffusion coefficients and diffusion enhancement factors under different solid fractions and shear rates. | Fig. 11 |

## Notes

The main statistical files are based on the same analysis criteria used in the manuscript:

- Al-rich local environment: `eta_Al >= 2.0`
- Strong Al-rich local environment: `eta_Al >= 3.0`
- Main Al-rich core-cluster threshold: `N_min = 5`
- Main statistics are averaged over the last 20% of production frames
- `D_perp_Al` is the transverse Al diffusion coefficient
- `D3D_deshear_Al` is the desheared three-dimensional Al diffusion coefficient

## Data workflow

```text
Raw LAMMPS output
    -> Analysis_Scripts/*.py
    -> Processed_Data/*.csv
    -> Manuscript figures
