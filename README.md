# AZ91 Al-rich cluster MD

This repository contains the processed data, LAMMPS input files, interatomic potential files, and analysis scripts used in the manuscript:

"Shear-Induced Evolution of Al-rich Clusters in Semisolid AZ91 Magnesium Alloy: A Molecular Dynamics Study"

## Repository structure

- `LAMMPS_Input/`
  - LAMMPS input script and initial atomic structure used for the semisolid AZ91 shear simulations.

- `Potential/`
  - MEAM potential files used in the simulations.

- `Analysis_Scripts/`
  - Post-processing scripts for local Al enrichment, Al-rich core cluster statistics, potential energy analysis, RDF and coordination analysis, and diffusion analysis.

- `Processed_Data/`
  - Processed data used to reproduce the figures and statistical results in the manuscript.

- `Supplementary_Data/`
  - Data related to supplementary tables and parameter sensitivity analyses.

## Simulation details

The simulations were performed using LAMMPS with the modified embedded atom method (MEAM) potential for Mg-Al-Zn alloys developed by Dickel et al. The main simulation matrix includes four semisolid states and five shear rates:

- Solid fractions: 20%, 40%, 60%, and 80%
- Temperatures: 865 K, 852 K, 833 K, and 813 K
- Shear rates: 0, 0.001, 0.005, 0.01, and 0.02 ps^-1
- Shear mode: yz shear, with flow along the y direction and velocity gradient along the z direction

The main LAMMPS input file is:

`LAMMPS_Input/cluster_shear.in`

## Potential files

The MEAM potential files used in this work are included in the `Potential/` folder:

- `library.meam`
- `Mg-Al-Zn.meam`

These files correspond to the Mg-Al-Zn MEAM potential reported by:

Dickel D. E., Baskes M. I., Aslam I., Barrett C. D.  
New interatomic potential for Mg-Al-Zn alloys with specific application to dilute Mg-based alloys.  
Modelling and Simulation in Materials Science and Engineering, 2018, 26, 045010.

## Data availability

Due to the large size of raw molecular dynamics trajectory files, this repository provides the processed data required to reproduce the figures and statistical results reported in the manuscript. Additional raw trajectory files are available from the corresponding author upon reasonable request.

## Notes

All statistical quantities reported in the manuscript were obtained from the production stage of each simulation case. Unless otherwise specified, the reported values were averaged over the last 20% of each production trajectory.
