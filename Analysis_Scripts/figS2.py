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
from collections import Counter
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

RAW_CSV_NAME = "FigS2_raw_frame_results.csv"
SUMMARY_CSV_NAME = "FigS2_summary.csv"


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

# Al-rich local environment identification parameters
RLOC_RICH = 5.0
ETA_CUT_RICH = 2.0
MIN_NEIGH_RICH = 6

# Fig. S2(a): r_conn sensitivity, fixed N_min = 5
RCONN_LIST = [3.3, 3.5, 3.7]
NMIN_FIXED = 5

# Fig. S2(b): N_min sensitivity, fixed r_conn = 3.5 A
RCONN_FIXED = 3.5
NMIN_LIST = [4, 5, 6]


# ============================================================
# 4. Build input file list
# ============================================================
def build_case_files(root_dir: Path) -> List[Dict[str, Any]]:
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
    checked_cases: List[Dict[str, Any]] = []

    for case in case_files:
        gdot_str = case["gdot_str"]
        folder = Path(case["folder"])
        dump_path = Path(case["dump_path"])

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
# 5. Union-Find
# ============================================================
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# 6. Core calculation
# ============================================================
def identify_rich_al_atoms(
    data,
    rloc: float = 5.0,
    eta_cut: float = 2.0,
    min_neigh: int = 6,
    al_type: int = 2,
):
    """
    Identify Al atoms in Al-rich local environments:
        type == 2
        coord_all >= min_neigh
        eta_Al >= eta_cut

    Return:
        rich_al_indices : ndarray of global particle indices
        n_al_total      : total number of Al atoms
    """
    particle_types = np.asarray(data.particles["Particle Type"])
    n_total = data.particles.count

    al_indices = np.where(particle_types == al_type)[0]
    n_al_total = len(al_indices)

    if n_al_total == 0:
        raise RuntimeError("No Al atoms were found in this frame.")

    x_al_bulk = n_al_total / n_total
    finder = CutoffNeighborFinder(rloc, data)

    rich_al_list = []

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

        if (coord_all >= min_neigh) and (eta_al >= eta_cut):
            rich_al_list.append(i)

    return np.asarray(rich_al_list, dtype=int), n_al_total


def get_cluster_sizes(data, selected_indices: np.ndarray, r_conn: float) -> List[int]:
    """
    Cluster selected atoms using the connection cutoff r_conn and return all cluster sizes.
    """
    n_sel = len(selected_indices)

    if n_sel == 0:
        return []

    if n_sel == 1:
        return [1]

    index_to_local = {gidx: lid for lid, gidx in enumerate(selected_indices)}
    selected_set = set(selected_indices.tolist())

    uf = UnionFind(n_sel)
    finder = CutoffNeighborFinder(r_conn, data)

    for gidx in selected_indices:
        i_local = index_to_local[gidx]

        for neigh in finder.find(gidx):
            j = neigh.index

            if j not in selected_set:
                continue

            j_local = index_to_local[j]
            uf.union(i_local, j_local)

    roots = [uf.find(i) for i in range(n_sel)]
    count = Counter(roots)
    cluster_sizes = sorted(count.values(), reverse=True)

    return cluster_sizes


def summarize_clusters(cluster_sizes: List[int], nmin: int):
    """
    For a given list of cluster sizes, calculate:
        n_clusters
        max_cluster_size
    using the minimum cluster size threshold nmin.
    """
    valid_sizes = [s for s in cluster_sizes if s >= nmin]

    n_clusters = len(valid_sizes)
    max_cluster_size = max(valid_sizes) if len(valid_sizes) > 0 else 0

    return n_clusters, max_cluster_size


def analyze_one_frame(data, frame_index: int, gdot: float) -> List[Dict[str, Any]]:
    """
    Analyze one frame:
        (a) r_conn sensitivity, fixed N_min
        (b) N_min sensitivity, fixed r_conn
    """
    rows: List[Dict[str, Any]] = []

    rich_al_indices, n_al_total = identify_rich_al_atoms(
        data=data,
        rloc=RLOC_RICH,
        eta_cut=ETA_CUT_RICH,
        min_neigh=MIN_NEIGH_RICH,
        al_type=AL_TYPE,
    )

    n_rich_al_total = len(rich_al_indices)

    # --------------------------------------------------------
    # Fig. S2(a): r_conn sensitivity, fixed N_min
    # --------------------------------------------------------
    for r_conn in RCONN_LIST:
        cluster_sizes = get_cluster_sizes(data, rich_al_indices, r_conn)
        n_clusters, max_cluster_size = summarize_clusters(cluster_sizes, NMIN_FIXED)

        rows.append({
            "frame": frame_index,
            "gdot": gdot,
            "panel": "rconn_sensitivity",
            "parameter": f"r_conn = {r_conn:.1f} A",
            "r_conn": r_conn,
            "nmin": NMIN_FIXED,
            "n_al_total": n_al_total,
            "n_rich_al_total": n_rich_al_total,
            "n_clusters": n_clusters,
            "max_cluster_size": max_cluster_size,
        })

    # --------------------------------------------------------
    # Fig. S2(b): N_min sensitivity, fixed r_conn
    # --------------------------------------------------------
    cluster_sizes_fixed = get_cluster_sizes(data, rich_al_indices, RCONN_FIXED)

    for nmin in NMIN_LIST:
        n_clusters, max_cluster_size = summarize_clusters(cluster_sizes_fixed, nmin)

        rows.append({
            "frame": frame_index,
            "gdot": gdot,
            "panel": "nmin_sensitivity",
            "parameter": f"N_min = {nmin}",
            "r_conn": RCONN_FIXED,
            "nmin": nmin,
            "n_al_total": n_al_total,
            "n_rich_al_total": n_rich_al_total,
            "n_clusters": n_clusters,
            "max_cluster_size": max_cluster_size,
        })

    return rows


def analyze_one_dump_worker(case: Dict[str, Any], last_n_frames: int) -> pd.DataFrame:
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
# 7. Main extraction
# ============================================================
def main() -> None:
    root_dir = DATA_ROOT.resolve()
    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / RAW_CSV_NAME
    summary_csv = out_dir / SUMMARY_CSV_NAME

    print("=" * 80)
    print("Fig. S2 fs40 Al-rich core cluster number sensitivity analysis")
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
        by=["panel", "gdot", "r_conn", "nmin", "frame"]
    ).reset_index(drop=True)

    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved raw frame results: {raw_csv}")

    summary_df = (
        raw_df
        .groupby(["panel", "gdot", "parameter", "r_conn", "nmin"], as_index=False)
        .agg(
            mean_n_clusters=("n_clusters", "mean"),
            std_n_clusters=("n_clusters", "std"),
            mean_max_cluster_size=("max_cluster_size", "mean"),
            std_max_cluster_size=("max_cluster_size", "std"),
            n_frames=("n_clusters", "count"),
        )
    )

    summary_df = summary_df.sort_values(
        by=["panel", "gdot", "r_conn", "nmin"]
    ).reset_index(drop=True)

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary results: {summary_csv}")

    print("\nSummary preview:")
    with pd.option_context("display.max_rows", 80, "display.max_columns", 20, "display.width", 200):
        print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
