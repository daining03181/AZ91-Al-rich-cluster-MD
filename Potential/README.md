# MEAM potential files

This folder contains the MEAM potential files used for the Mg-Al-Zn molecular dynamics simulations in this work.

Files included:

- `library.meam`
- `Mg-Al-Zn.meam`

The potential corresponds to the Mg-Al-Zn MEAM potential developed by Dickel et al.:

Dickel D. E., Baskes M. I., Aslam I., Barrett C. D.  
New interatomic potential for Mg-Al-Zn alloys with specific application to dilute Mg-based alloys.  
Modelling and Simulation in Materials Science and Engineering, 2018, 26, 045010.

These potential files were used together with the following LAMMPS command:

pair_style meam
pair_coeff * * library.meam Mg Al Zn Mg-Al-Zn.meam Mg Al Zn
