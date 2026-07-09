from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# RDF column mapping
# =========================
R_COL = 1
G_MG_AL = 4
COORD_MG_AL = 5
G_AL_AL = 8
COORD_AL_AL = 9


# =========================
# Case order
# =========================
FS_ORDER = ["fs20", "fs40", "fs60", "fs80"]
GDOT_ORDER = [0.0, 0.001, 0.005, 0.01, 0.02]


# =========================
# Helper functions
# =========================
def parse_case_from_folder(folder_name: str) -> Dict[str, Any]:
    """
    Example:
        AZ91_fs60_T833.0_shearYZ_gdot0.01
    """
    out: Dict[str, Any] = {
        "fs_label": "",
        "fs": np.nan,
        "T": np.nan,
        "gdot": np.nan,
    }

    parts = folder_name.split("_")

    for p in parts:
        if p.startswith("fs"):
            out["fs_label"] = p

            m = re.search(r"fs(\d+)", p)
            if m:
                out["fs"] = int(m.group(1))

        elif p.startswith("T"):
            try:
                out["T"] = float(p[1:])
            except Exception:
                pass

        elif p.startswith("gdot"):
            try:
                out["gdot"] = float(p.replace("gdot", ""))
            except Exception:
                pass

    return out


def find_rdf_file(root: Path, fs_label: str, gdot: float) -> Path:
    """
    Find rdf.production*.dat for a given solid fraction and shear rate.
    """
    candidates = sorted(
        root.glob(f"AZ91_{fs_label}_T*_shearYZ_gdot{gdot}*/rdf.production*.dat")
    )

    if not candidates:
        all_files = sorted(
            root.glob(f"AZ91_{fs_label}_T*_shearYZ_gdot*/rdf.production*.dat")
        )

        matched = []

        for f in all_files:
            info = parse_case_from_folder(f.parent.name)

            if info["fs_label"] == fs_label and np.isclose(info["gdot"], gdot):
                matched.append(f)

        candidates = matched

    if not candidates:
        raise FileNotFoundError(f"Cannot find RDF file for {fs_label}, gdot={gdot}")

    if len(candidates) > 1:
        print(f"Warning: multiple RDF files found for {fs_label}, gdot={gdot}. Use {candidates[0]}")

    return candidates[0]


def read_last_rdf_block(rdf_file: Path) -> Tuple[int, np.ndarray]:
    """
    Read the last timestep block from a LAMMPS fix ave/time RDF file.
    """
    rdf_file = Path(rdf_file)
    lines = rdf_file.read_text(encoding="utf-8", errors="ignore").splitlines()

    blocks: List[Tuple[int, np.ndarray]] = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if (not line) or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        if len(parts) == 2:
            try:
                timestep = int(float(parts[0]))
                nrows = int(float(parts[1]))

                rows = []
                ok = True

                for j in range(i + 1, min(i + 1 + nrows, n)):
                    row_parts = lines[j].split()

                    if len(row_parts) < 14:
                        ok = False
                        break

                    rows.append([float(x) for x in row_parts[:14]])

                if ok and len(rows) == nrows:
                    blocks.append((timestep, np.array(rows, dtype=float)))
                    i += 1 + nrows
                    continue

            except Exception:
                pass

        i += 1

    if not blocks:
        raise RuntimeError(f"No RDF block found in {rdf_file}")

    return blocks[-1]


def value_at_rcut_nearest(r: np.ndarray, y: np.ndarray, rcut: float) -> float:
    """
    Take the coordination number at the grid point nearest to rcut.

    This follows the original Fig7_RDF_coordination_number.py.
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y)

    if not np.any(mask):
        return np.nan

    rr = r[mask]
    yy = y[mask]

    idx = int(np.argmin(np.abs(rr - rcut)))

    return float(yy[idx])


def first_peak_intensity(r: np.ndarray, g: np.ndarray, rmin: float, rmax: float) -> Tuple[float, float]:
    """
    Extract the first peak intensity and corresponding peak position.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)

    mask = np.isfinite(r) & np.isfinite(g) & (r >= rmin) & (r <= rmax)

    if not np.any(mask):
        return np.nan, np.nan

    rr = r[mask]
    gg = g[mask]

    idx = int(np.nanargmax(gg))

    return float(gg[idx]), float(rr[idx])


def extract_one_case(root: Path, fs_label: str, gdot: float, rcut: float, peak_rmin: float, peak_rmax: float) -> Dict[str, Any]:
    rdf_file = find_rdf_file(root, fs_label, gdot)

    folder_info = parse_case_from_folder(rdf_file.parent.name)
    timestep, arr = read_last_rdf_block(rdf_file)

    r = arr[:, R_COL]
    g_al_al = arr[:, G_AL_AL]
    coord_al_al = arr[:, COORD_AL_AL]
    coord_mg_al = arr[:, COORD_MG_AL]

    n_alal = value_at_rcut_nearest(r, coord_al_al, rcut)
    n_mgal = value_at_rcut_nearest(r, coord_mg_al, rcut)

    if np.isfinite(n_mgal) and abs(n_mgal) > 1e-12:
        ratio = n_alal / n_mgal
    else:
        ratio = np.nan

    alal_peak, r_alal_peak = first_peak_intensity(
        r,
        g_al_al,
        rmin=peak_rmin,
        rmax=peak_rmax,
    )

    return {
        "fs": int(folder_info["fs"]),
        "fs_label": fs_label,
        "T": float(folder_info["T"]),
        "gdot": float(gdot),

        "rdf_timestep": int(timestep),
        "r_cut_A": float(rcut),
        "peak_rmin_A": float(peak_rmin),
        "peak_rmax_A": float(peak_rmax),

        # Direct Fig. 10 data
        "N_AlAl_rcut": float(n_alal),
        "N_MgAl_rcut": float(n_mgal),
        "ratio_AlAl_MgAl_rcut": float(ratio),
        "AlAl_peak": float(alal_peak),

        # Traceability
        "r_AlAl_peak_A": float(r_alal_peak),
        "rdf_file": str(rdf_file.relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Fig. 10 local Al-Al correlation data from RDF production files."
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Root folder containing AZ91_fs*_T*_shearYZ_gdot* directories.",
    )

    parser.add_argument(
        "--rcut",
        type=float,
        default=3.8,
        help="Cutoff radius for first-shell coordination numbers in Å. Default: 3.8.",
    )

    parser.add_argument(
        "--peak-rmin",
        type=float,
        default=2.5,
        help="Lower bound for Al-Al first peak search in Å. Default: 2.5.",
    )

    parser.add_argument(
        "--peak-rmax",
        type=float,
        default=3.6,
        help="Upper bound for Al-Al first peak search in Å. Default: 3.6.",
    )

    parser.add_argument(
        "--out",
        default="Processed_Data/Fig10_local_correlation.csv",
        help="Output CSV path. Default: Processed_Data/Fig10_local_correlation.csv",
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also save rdf_coordination_summary.csv and rdf_peak_summary.csv in Processed_Data/.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    out_csv = Path(args.out)

    if not out_csv.is_absolute():
        out_csv = root / out_csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    print(f"Root folder: {root}")
    print(f"r_cut = {args.rcut} Å")
    print(f"Al-Al RDF first peak search range = {args.peak_rmin}–{args.peak_rmax} Å")

    for fs_label in FS_ORDER:
        for gdot in GDOT_ORDER:
            print(f"Reading {fs_label}, gdot={gdot}")

            row = extract_one_case(
                root=root,
                fs_label=fs_label,
                gdot=gdot,
                rcut=args.rcut,
                peak_rmin=args.peak_rmin,
                peak_rmax=args.peak_rmax,
            )

            rows.append(row)

    fig10_df = pd.DataFrame(rows)
    fig10_df = fig10_df.sort_values(["fs", "gdot"]).reset_index(drop=True)

    # Put the columns used by Fig. 10 in a clear order.
    preferred_cols = [
        "fs", "T", "gdot",
        "N_AlAl_rcut",
        "N_MgAl_rcut",
        "ratio_AlAl_MgAl_rcut",
        "AlAl_peak",
        "r_AlAl_peak_A",
        "r_cut_A",
        "peak_rmin_A",
        "peak_rmax_A",
        "rdf_timestep",
        "rdf_file",
        "fs_label",
    ]

    fig10_df = fig10_df[[c for c in preferred_cols if c in fig10_df.columns]]

    fig10_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if args.save_intermediate:
        coord_csv = out_csv.parent / "Fig10_rdf_coordination_summary.csv"
        peak_csv = out_csv.parent / "Fig10_rdf_peak_summary.csv"

        coord_df = fig10_df[
            [
                "fs",
                "T",
                "gdot",
                "rdf_timestep",
                "r_cut_A",
                "N_AlAl_rcut",
                "N_MgAl_rcut",
                "ratio_AlAl_MgAl_rcut",
                "rdf_file",
            ]
        ].copy()

        peak_df = fig10_df[
            [
                "fs",
                "T",
                "gdot",
                "rdf_timestep",
                "AlAl_peak",
                "r_AlAl_peak_A",
                "peak_rmin_A",
                "peak_rmax_A",
                "rdf_file",
            ]
        ].copy()

        coord_df.to_csv(coord_csv, index=False, encoding="utf-8-sig")
        peak_df.to_csv(peak_csv, index=False, encoding="utf-8-sig")

        print(f"Saved intermediate coordination summary to: {coord_csv}")
        print(f"Saved intermediate RDF peak summary to: {peak_csv}")

    print("\n============================================================")
    print(f"Saved Fig. 10 local-correlation data to: {out_csv}")
    print("Direct Fig. 10 columns:")
    print("  N_AlAl_rcut")
    print("  N_MgAl_rcut")
    print("  ratio_AlAl_MgAl_rcut")
    print("  AlAl_peak")
    print("============================================================")

    with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 220):
        print(
            fig10_df[
                [
                    "fs",
                    "T",
                    "gdot",
                    "N_AlAl_rcut",
                    "N_MgAl_rcut",
                    "ratio_AlAl_MgAl_rcut",
                    "AlAl_peak",
                    "r_AlAl_peak_A",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
