from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Default parameters
# ============================================================
DT_PS = 0.001
FIT_START_FRAC = 0.30
AL_TYPE = 2

CASE_PATTERN = re.compile(
    r"AZ91_fs(?P<fs>\d+)_T(?P<T>[\d.]+)_shearYZ_gdot(?P<gdot>[\d.]+)"
)


# ============================================================
# Case folder parsing
# ============================================================
def parse_case_folder(folder: Path) -> Dict[str, Any] | None:
    m = CASE_PATTERN.search(folder.name)

    if not m:
        return None

    return {
        "folder": str(folder),
        "folder_name": folder.name,
        "fs": int(m.group("fs")),
        "T": float(m.group("T")),
        "gdot": float(m.group("gdot")),
    }


def scan_case_folders(root: Path) -> List[Dict[str, Any]]:
    case_folders: List[Dict[str, Any]] = []

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        info = parse_case_folder(folder)

        if info is not None:
            case_folders.append(info)

    case_folders = sorted(case_folders, key=lambda x: (x["fs"], x["gdot"]))

    if not case_folders:
        raise RuntimeError(
            f"No case folders found under: {root}\n"
            "Expected folder pattern: AZ91_fs20_T865.0_shearYZ_gdot0.001"
        )

    return case_folders


# ============================================================
# Read fix ave/time MSD output
# ============================================================
def read_fix_ave_time(file_path: Path) -> np.ndarray:
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue

    arr = np.asarray(rows, dtype=float)

    if arr.ndim != 2 or arr.shape[0] < 5:
        raise RuntimeError(f"Too few valid rows for diffusion fitting: {file_path}")

    return arr


# ============================================================
# D_perp_Al from msd.production.components
# ============================================================
def calc_Dperp_Al_from_msd(msd_file: Path) -> float:
    arr = read_fix_ave_time(msd_file)

    # Column mapping:
    # 0  Step
    # 7  Al_MSDx
    # 9  Al_MSDz
    if arr.shape[1] < 19:
        raise RuntimeError(
            f"MSD file has insufficient columns. "
            f"Expected at least 19, got {arr.shape[1]}: {msd_file}"
        )

    step = arr[:, 0]
    time_ps = step * DT_PS

    msd_x_Al = arr[:, 7]
    msd_z_Al = arr[:, 9]

    # yz shear: y is the flow direction, x and z are transverse directions.
    msd_perp_Al = msd_x_Al + msd_z_Al

    mask = np.isfinite(time_ps) & np.isfinite(msd_perp_Al) & (time_ps > 0)
    time_ps = time_ps[mask]
    msd_perp_Al = msd_perp_Al[mask]

    if len(time_ps) < 5:
        raise RuntimeError(f"Too few valid MSD points for fitting: {msd_file}")

    i0 = int(len(time_ps) * FIT_START_FRAC)

    slope, _ = np.polyfit(time_ps[i0:], msd_perp_Al[i0:], 1)

    # MSD_x + MSD_z = 4 D t
    return float(slope / 4.0)


# ============================================================
# Dump reader
# ============================================================
def iter_lammpstrj_frames(dump_file: Path):
    with open(dump_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()

            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(f"Missing NUMBER OF ATOMS section: {dump_file}")

            natoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"Missing BOX BOUNDS section: {dump_file}")

            # Box lines are not used in this original deshear implementation.
            f.readline()
            f.readline()
            f.readline()

            line = f.readline().strip()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError(f"Missing ATOMS section: {dump_file}")

            cols = line.split()[2:]
            col_index = {name: i for i, name in enumerate(cols)}

            required_cols = ["id", "type", "xu", "yu", "zu"]
            for col in required_cols:
                if col not in col_index:
                    raise RuntimeError(f"Dump file missing column {col}: {dump_file}")

            data = np.empty((natoms, len(cols)), dtype=float)

            for i in range(natoms):
                data[i, :] = np.fromstring(f.readline(), sep=" ")

            yield timestep, cols, data


# ============================================================
# D_deshear_Al from dump.production.Alrich
# ============================================================
def calc_D_deshear_Al_from_dump(dump_file: Path, gdot: float) -> float:
    t_list = []
    msd_list = []

    ids0 = None
    r0 = None

    for iframe, (step, cols, data) in enumerate(iter_lammpstrj_frames(dump_file)):
        ci = {name: i for i, name in enumerate(cols)}

        ids = data[:, ci["id"]].astype(np.int64)
        types = data[:, ci["type"]].astype(np.int64)

        x = data[:, ci["xu"]]
        y = data[:, ci["yu"]]
        z = data[:, ci["zu"]]

        time_ps = step * DT_PS
        gamma = gdot * time_ps

        # yz shear:
        # flow direction = y
        # gradient direction = z
        # affine contribution is removed as y' = y - gamma z
        y_deshear = y - gamma * z

        r = np.column_stack([x, y_deshear, z])

        mask_Al = types == AL_TYPE
        ids_Al = ids[mask_Al]
        r_Al = r[mask_Al]

        order = np.argsort(ids_Al)
        ids_Al = ids_Al[order]
        r_Al = r_Al[order]

        if iframe == 0:
            ids0 = ids_Al.copy()
            r0 = r_Al.copy()
            continue

        if not np.array_equal(ids_Al, ids0):
            raise RuntimeError(
                f"Al atom IDs changed during trajectory, cannot calculate MSD: {dump_file}"
            )

        dr = r_Al - r0

        # Remove center-of-mass drift of the Al group, similar to compute msd com yes.
        dr = dr - dr.mean(axis=0)

        msd_3d = np.mean(np.sum(dr * dr, axis=1))

        t_list.append(time_ps)
        msd_list.append(msd_3d)

    time_ps = np.asarray(t_list, dtype=float)
    msd_3d = np.asarray(msd_list, dtype=float)

    mask = np.isfinite(time_ps) & np.isfinite(msd_3d) & (time_ps > 0)
    time_ps = time_ps[mask]
    msd_3d = msd_3d[mask]

    if len(time_ps) < 5:
        raise RuntimeError(f"Too few dump frames for D_deshear fitting: {dump_file}")

    i0 = int(len(time_ps) * FIT_START_FRAC)

    slope, _ = np.polyfit(time_ps[i0:], msd_3d[i0:], 1)

    # MSD_3D_deshear = 6 D t
    return float(slope / 6.0)


# ============================================================
# One case
# ============================================================
def process_one_case(case: Dict[str, Any]) -> Dict[str, Any]:
    folder = Path(case["folder"])
    fs = case["fs"]
    T = case["T"]
    gdot = case["gdot"]

    msd_files = sorted(folder.glob("msd.production.components.*.dat"))
    dump_files = sorted(folder.glob("dump.production.Alrich.*.lammpstrj"))

    if not msd_files:
        raise RuntimeError(f"No msd.production.components file found: {folder}")

    if not dump_files:
        raise RuntimeError(f"No dump.production.Alrich file found: {folder}")

    msd_file = msd_files[0]
    dump_file = dump_files[0]

    D_perp_Al = calc_Dperp_Al_from_msd(msd_file)
    D_deshear_Al = calc_D_deshear_Al_from_dump(dump_file, gdot)

    return {
        "fs": fs,
        "T": T,
        "gdot": gdot,

        # Direct Fig. 11 data
        "D_perp_Al": D_perp_Al,
        "D_deshear_Al": D_deshear_Al,

        # Alias retained for consistency with older scripts
        "D3D_deshear_Al": D_deshear_Al,

        "folder": folder.name,
        "msd_file": msd_file.name,
        "dump_file": dump_file.name,
    }


# ============================================================
# Enhancement ratios
# ============================================================
def add_enhancement_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["K_perp_Al"] = np.nan
    df["K_deshear_Al"] = np.nan
    df["K3D_deshear_Al"] = np.nan

    for fs, sub in df.groupby("fs"):
        base = sub[np.isclose(sub["gdot"], 0.0)]

        if base.empty:
            raise RuntimeError(f"fs={fs}% has no gdot=0 baseline.")

        D0_perp = float(base["D_perp_Al"].iloc[0])
        D0_deshear = float(base["D_deshear_Al"].iloc[0])

        if abs(D0_perp) < 1e-12 or abs(D0_deshear) < 1e-12:
            raise RuntimeError(f"fs={fs}% baseline diffusion coefficient is zero.")

        idx = df["fs"] == fs

        df.loc[idx, "K_perp_Al"] = df.loc[idx, "D_perp_Al"] / D0_perp
        df.loc[idx, "K_deshear_Al"] = df.loc[idx, "D_deshear_Al"] / D0_deshear
        df.loc[idx, "K3D_deshear_Al"] = df.loc[idx, "K_deshear_Al"]

    return df


# ============================================================
# Main extraction
# ============================================================
def extract_diffusion_data(root: Path, n_workers: int) -> pd.DataFrame:
    case_folders = scan_case_folders(root)

    n_workers = min(n_workers, os.cpu_count() or 1, len(case_folders))

    print(f"Found {len(case_folders)} cases.")
    print(f"Using {n_workers} parallel workers.")

    records = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_one_case, case): case
            for case in case_folders
        }

        for future in as_completed(futures):
            case = futures[future]

            try:
                rec = future.result()
                records.append(rec)

                print(
                    f"Finished fs={rec['fs']}%, gdot={rec['gdot']}: "
                    f"D_perp_Al={rec['D_perp_Al']:.6e}, "
                    f"D_deshear_Al={rec['D_deshear_Al']:.6e}"
                )

            except Exception as exc:
                print(
                    f"\nError in fs={case['fs']}%, "
                    f"gdot={case['gdot']}: {exc}"
                )
                raise

    df = pd.DataFrame(records)
    df = df.sort_values(["fs", "gdot"]).reset_index(drop=True)

    df = add_enhancement_ratios(df)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Fig. 11 Al diffusion coefficients from MSD and dump files."
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Root folder containing AZ91_fs*_T*_shearYZ_gdot* directories.",
    )

    parser.add_argument(
        "--out",
        default="Processed_Data/Fig11_al_diffusion_coefficients.csv",
        help="Output CSV path. Default: Processed_Data/Fig11_al_diffusion_coefficients.csv",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel workers. Default: 20.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = root / out_csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = extract_diffusion_data(root=root, n_workers=args.workers)

    # Put the direct Fig. 11 columns first.
    preferred_cols = [
        "fs",
        "T",
        "gdot",
        "D_perp_Al",
        "D_deshear_Al",
        "D3D_deshear_Al",
        "K_perp_Al",
        "K_deshear_Al",
        "K3D_deshear_Al",
        "folder",
        "msd_file",
        "dump_file",
    ]

    ordered_cols = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n============================================================")
    print(f"Saved Fig. 11 Al diffusion coefficients to: {out_csv}")
    print("Direct Fig. 11 columns:")
    print("  D_perp_Al")
    print("  D_deshear_Al")
    print("============================================================")

    with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 220):
        print(
            df[
                [
                    "fs",
                    "T",
                    "gdot",
                    "D_perp_Al",
                    "D_deshear_Al",
                    "K_perp_Al",
                    "K_deshear_Al",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
