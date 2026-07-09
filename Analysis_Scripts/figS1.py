from __future__ import annotations

# ============================================================
# 0. Limit internal threading before importing numpy / ovito
# ============================================================
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
# 1. Imports
# ============================================================
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ovito.io import import_file
from ovito.data import CutoffNeighborFinder


# ============================================================
# 2. Fixed paths for your local computer
# ============================================================
DATA_ROOT = Path(r"C:\Users\Donkey\Desktop\数据\团簇")
OUT_DIR = DATA_ROOT / "github" / "Processed_Data"

RAW_CSV_NAME = "FigS1_raw_frame_results.csv"
SUMMARY_CSV_NAME = "FigS1_summary.csv"


# ============================================================
# 3. Default settings
# ============================================================
FS_LABEL = "fs40"
T_LABEL = "852.0"

GDOT_CASES = [
    ("0.0", 0.0),
    ("0.001", 0.001),
    ("0.005", 0.005),
    ("0.01", 0.01),
    ("0.02", 0.02),
]

# Parallel workers. You can reduce this to 2 or 3 if memory is high.
N_WORKERS = 5

# Number of last frames used for statistics.
LAST_N_FRAMES = 20

# Atom type in the LAMMPS model:
# 1 = Mg, 2 = Al, 3 = Zn
AL_TYPE = 2

# Minimum neighbor number, consistent with the LAMMPS input.
MIN_NEIGH = 6

# Fig. S1(a): r_loc sensitivity at eta_Al >= 2.0
RLOC_LIST = [4.5, 5.0, 5.5]
ETA_FIXED = 2.0

# Fig. S1(b): eta_Al sensitivity at r_loc = 5.0 A
RLOC_FIXED = 5.0
ETA_LIST = [1.8, 2.0, 2.2]


# ============================================================
# 4. Build input file list
# ============================================================
def build_case_files(root_dir: Path) -> List[Dict[str, Any]]:
    """
    Generate the five fs40 production dump paths according to the
    original directory naming rule.
    """
    case_files: List[Dict[str, Any]] = []

    for gdot_str, gdot_value in GDOT_CASES:
        folder = root_dir / f"AZ91_{FS_LABEL}_T{T_LABEL}_shearYZ_gdot{gdot_str}"
        dump_name = f"dump.production.Alrich.AZ91.{FS_LABEL}.T{T_LABEL}.gdot{gdot_str}.lammpstrj"
        dump_path = folder / dump_name

        case_files.append({
            "gdot_str": gdot_str,
            "gdot": gdot_value,
            "folder": str(folder),
            "dump_path": str(dump_path),
        })

    return case_files


def check_input_files(case_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check whether the standard dump path exists. If it does not exist,
    automatically search for dump.production*.lammpstrj in the case folder.
    """
    checked_cases: List[Dict[str, Any]] = []

    for case in case_files:
        folder = Path(case["folder"])
        dump_path = Path(case["dump_path"])
        gdot_str = case["gdot_str"]

        if dump_path.exists():
            checked_cases.append(case)
            continue

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder for gdot = {gdot_str}: {folder}")

        candidates = sorted(folder.glob("dump.production*.lammpstrj"))

        if len(candidates) == 1:
            case["dump_path"] = str(candidates[0])
            checked_cases.append(case)
            print(f"[Auto found] gdot = {gdot_str}: {candidates[0]}")

        elif len(candidates) > 1:
            print(f"\nMultiple production dump files found in: {folder}")
            for c in candidates:
                print(f"  {c.name}")

            raise FileNotFoundError(
                "Multiple candidate dump files found. Please check file names."
            )

        else:
            raise FileNotFoundError(
                f"No dump.production*.lammpstrj found in {folder} for gdot = {gdot_str}"
            )

    return checked_cases


# ============================================================
# 5. Core calculation
# ============================================================
def calculate_eta_for_al_atoms(data, rloc: float, al_type: int = 2):
    """
    For all Al atoms, calculate:
        coord_all
        eta_Al = x_Al_local / x_Al_bulk

    where:
        x_Al_local = (Al neighbors + central Al atom) / (all neighbors + central atom)
        x_Al_bulk  = N_Al / N_total

    Return:
        coord_all_arr, eta_arr, n_al_total
    """
    particle_types = np.asarray(data.particles["Particle Type"])
    n_total = data.particles.count

    al_indices = np.where(particle_types == al_type)[0]
    n_al_total = len(al_indices)

    if n_al_total == 0:
        raise RuntimeError("No Al atoms were found in this frame.")

    x_al_bulk = n_al_total / n_total

    finder = CutoffNeighborFinder(rloc, data)

    coord_all_list = []
    eta_list = []

    for i in al_indices:
        coord_all = 0
        coord_al = 0

        for neigh in finder.find(i):
            j = neigh.index
            coord_all += 1

            if particle_types[j] == al_type:
                coord_al += 1

        # The central atom is Al, so both numerator and denominator add 1.
        x_al_local = (coord_al + 1.0) / (coord_all + 1.0)
        eta_al = x_al_local / x_al_bulk

        coord_all_list.append(coord_all)
        eta_list.append(eta_al)

    return np.asarray(coord_all_list), np.asarray(eta_list), n_al_total


def analyze_one_frame(data, frame_index: int, gdot: float) -> List[Dict[str, Any]]:
    """
    Analyze one frame and return raw statistics for Fig. S1.
    """
    rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Fig. S1(a): r_loc sensitivity, eta_Al >= 2.0
    # --------------------------------------------------------
    for rloc in RLOC_LIST:
        coord_all_arr, eta_arr, n_al_total = calculate_eta_for_al_atoms(
            data=data,
            rloc=rloc,
            al_type=AL_TYPE,
        )

        rich_mask = (coord_all_arr >= MIN_NEIGH) & (eta_arr >= ETA_FIXED)
        n_rich_al = int(np.sum(rich_mask))
        fraction_percent = 100.0 * n_rich_al / n_al_total

        rows.append({
            "frame": frame_index,
            "gdot": gdot,
            "panel": "rloc_sensitivity",
            "parameter": f"r_loc = {rloc:.1f} A",
            "rloc": rloc,
            "eta_cut": ETA_FIXED,
            "n_al_total": n_al_total,
            "n_rich_al": n_rich_al,
            "fraction_percent": fraction_percent,
        })

    # --------------------------------------------------------
    # Fig. S1(b): eta_Al sensitivity, r_loc = 5.0 A
    # --------------------------------------------------------
    coord_all_arr, eta_arr, n_al_total = calculate_eta_for_al_atoms(
        data=data,
        rloc=RLOC_FIXED,
        al_type=AL_TYPE,
    )

    for eta_cut in ETA_LIST:
        rich_mask = (coord_all_arr >= MIN_NEIGH) & (eta_arr >= eta_cut)
        n_rich_al = int(np.sum(rich_mask))
        fraction_percent = 100.0 * n_rich_al / n_al_total

        rows.append({
            "frame": frame_index,
            "gdot": gdot,
            "panel": "eta_sensitivity",
            "parameter": f"eta_Al >= {eta_cut:.1f}",
            "rloc": RLOC_FIXED,
            "eta_cut": eta_cut,
            "n_al_total": n_al_total,
            "n_rich_al": n_rich_al,
            "fraction_percent": fraction_percent,
        })

    return rows


def analyze_one_dump_worker(case: Dict[str, Any], last_n_frames: int) -> pd.DataFrame:
    """
    Worker function for one gdot case.
    """
    gdot = case["gdot"]
    gdot_str = case["gdot_str"]
    dump_path = Path(case["dump_path"])

    print("=" * 80, flush=True)
    print(f"Start processing gdot = {gdot_str} ps^-1", flush=True)
    print(f"File: {dump_path}", flush=True)

    pipeline = import_file(str(dump_path), multiple_frames=True)
    n_frames = pipeline.source.num_frames

    if n_frames <= 0:
        raise RuntimeError(f"No frame found in file: {dump_path}")

    start_frame = max(0, n_frames - last_n_frames)
    frame_indices = list(range(start_frame, n_frames))

    print(
        f"gdot = {gdot_str}: total frames = {n_frames}, using last frames = {frame_indices}",
        flush=True,
    )

    all_rows: List[Dict[str, Any]] = []

    for frame in frame_indices:
        print(f"gdot = {gdot_str}: analyzing frame {frame}", flush=True)
        data = pipeline.compute(frame)
        rows = analyze_one_frame(data=data, frame_index=frame, gdot=gdot)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    print(f"Finished gdot = {gdot_str} ps^-1", flush=True)
    return df


# ============================================================
# 6. Main extraction
# ============================================================
def main() -> None:
    root_dir = DATA_ROOT.resolve()
    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / RAW_CSV_NAME
    summary_csv = out_dir / SUMMARY_CSV_NAME

    print("=" * 80)
    print("Fig. S1 fs40 Al-rich local environment sensitivity analysis")
    print(f"Root directory : {root_dir}")
    print(f"Output folder  : {out_dir}")
    print(f"Raw CSV        : {raw_csv.name}")
    print(f"Summary CSV    : {summary_csv.name}")
    print(f"Using workers  : {N_WORKERS}")
    print(f"Last N frames  : {LAST_N_FRAMES}")
    print("=" * 80)

    case_files = build_case_files(root_dir)
    case_files = check_input_files(case_files)

    print("\nInput dump files:")
    for case in case_files:
        print(f"gdot = {case['gdot_str']}: {case['dump_path']}")

    print("\nStart parallel analysis...")

    n_workers = min(N_WORKERS, len(case_files), os.cpu_count() or 1)
    all_dfs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(analyze_one_dump_worker, case, LAST_N_FRAMES): case
            for case in case_files
        }

        for future in as_completed(futures):
            case = futures[future]
            gdot_str = case["gdot_str"]

            try:
                df_case = future.result()
                all_dfs.append(df_case)
                print(f"[Completed] gdot = {gdot_str}")

            except Exception as exc:
                print(f"[Error] gdot = {gdot_str}: {exc}")
                raise

    raw_df = pd.concat(all_dfs, ignore_index=True)

    raw_df = raw_df.sort_values(
        by=["panel", "gdot", "rloc", "eta_cut", "frame"]
    ).reset_index(drop=True)

    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved raw frame results: {raw_csv}")

    summary_df = (
        raw_df
        .groupby(["panel", "gdot", "parameter", "rloc", "eta_cut"], as_index=False)
        .agg(
            mean_fraction_percent=("fraction_percent", "mean"),
            std_fraction_percent=("fraction_percent", "std"),
            n_frames=("fraction_percent", "count"),
        )
    )

    summary_df = summary_df.sort_values(
        by=["panel", "gdot", "rloc", "eta_cut"]
    ).reset_index(drop=True)

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary results: {summary_csv}")

    print("\nSummary preview:")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 20, "display.width", 180):
        print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
