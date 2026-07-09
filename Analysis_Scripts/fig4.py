import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
# =========================
# Global parameters
# =========================
root = Path.cwd()

fs_values = [20, 40, 60, 80]
gdot_target = 0.0

min_cluster_size = 5
last_fraction = 0.20        
# =========================
# Helper functions
# =========================
def parse_case_from_folder(folder_name):
    """
    Parse fs, T and gdot from folder name.
    Example:
    AZ91_fs20_T865.0_shearYZ_gdot0.0
    """
    m = re.search(
        r"AZ91_fs(?P<fs>\d+)_T(?P<T>[0-9.]+)_shearYZ_gdot(?P<gdot>[0-9.]+)",
        folder_name
    )

    if m is None:
        return None

    return {
        "fs": int(m.group("fs")),
        "T": float(m.group("T")),
        "gdot": float(m.group("gdot"))
    }


def find_case_folder(fs, gdot):
    candidates = []

    for p in root.iterdir():
        if not p.is_dir():
            continue

        info = parse_case_from_folder(p.name)

        if info is None:
            continue

        if info["fs"] == fs and abs(info["gdot"] - gdot) < 1e-9:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Cannot find folder for fs={fs}, gdot={gdot}")

    if len(candidates) > 1:
        print(f"Warning: multiple folders found for fs={fs}, gdot={gdot}. Use {candidates[0].name}")

    return candidates[0]


def find_dump_file(folder):
    patterns = [
        "dump.production.Alrich*.lammpstrj",
        "dump.production*.lammpstrj",
        "*.lammpstrj"
    ]

    for pat in patterns:
        files = sorted(folder.glob(pat))
        if files:
            return files[0]

    raise FileNotFoundError(f"No dump file found in {folder}")


def find_col_index(header, aliases):
    norm = {c.lower(): i for i, c in enumerate(header)}

    for a in aliases:
        if a.lower() in norm:
            return norm[a.lower()]

    for i, c in enumerate(header):
        cl = c.lower()
        for a in aliases:
            al = a.lower()
            if al in cl or cl in al:
                return i

    return None


def analyze_dump_all_frames(dump_file):
    """
    Read all frames in the production dump, but only store per-frame statistics.

    Per frame:
    - F_eta>=2
    - F_eta>=3
    - number of Al-rich Al core clusters with N >= 5
    - maximum cluster size
    """
    records = []
    frame_index = 0

    with open(dump_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()

            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break

            try:
                timestep = int(float(timestep_line.strip()))
            except Exception:
                timestep = frame_index

            n_atoms = None
            atom_header = None

            # Read until ITEM: ATOMS
            while True:
                line = f.readline()

                if not line:
                    break

                if line.startswith("ITEM: NUMBER OF ATOMS"):
                    n_atoms = int(f.readline().strip())

                elif line.startswith("ITEM: ATOMS"):
                    atom_header = line.split()[2:]
                    break

            if n_atoms is None or atom_header is None:
                break

            cid_idx = find_col_index(
                atom_header,
                [
                    "c_cidRichAl",
                    "c_cidrichal",
                    "c_cid_alrich",
                    "c_cidRich",
                    "c_clusterRichAl"
                ]
            )

            eta_idx = find_col_index(
                atom_header,
                [
                    "v_etaAl",
                    "v_etaal",
                    "etaAl",
                    "eta_al"
                ]
            )

            if cid_idx is None:
                raise RuntimeError(
                    f"Cannot find Al-rich cluster ID column in {dump_file}\n"
                    f"Available columns: {atom_header}"
                )

            cluster_counts = defaultdict(int)

            n_alrich_al = 0
            eta_valid_count = 0
            eta_ge2_count = 0
            eta_ge3_count = 0

            for _ in range(n_atoms):
                atom_line = f.readline()
                if not atom_line:
                    break

                parts = atom_line.split()

                if len(parts) < len(atom_header):
                    continue

                # Cluster ID of Al-rich Al core atoms
                try:
                    cid = int(float(parts[cid_idx]))
                except Exception:
                    cid = 0

                if cid > 0:
                    cluster_counts[cid] += 1
                    n_alrich_al += 1

                # etaAl statistics
                if eta_idx is not None:
                    try:
                        eta = float(parts[eta_idx])
                    except Exception:
                        eta = np.nan

                    if np.isfinite(eta):
                        eta_valid_count += 1

                        if eta >= 2.0:
                            eta_ge2_count += 1

                        if eta >= 3.0:
                            eta_ge3_count += 1

            sizes = np.array(list(cluster_counts.values()), dtype=int)

            if sizes.size > 0:
                max_size = int(np.max(sizes))
                n_ge5 = int(np.sum(sizes >= min_cluster_size))
                n_ge7 = int(np.sum(sizes >= 7))
            else:
                max_size = 0
                n_ge5 = 0
                n_ge7 = 0

            if eta_valid_count > 0:
                f_eta_ge2 = eta_ge2_count / eta_valid_count * 100.0
                f_eta_ge3 = eta_ge3_count / eta_valid_count * 100.0
            else:
                f_eta_ge2 = np.nan
                f_eta_ge3 = np.nan

            records.append({
                "frame_index": frame_index,
                "timestep": timestep,
                "n_atoms": n_atoms,
                "n_alrich_al": n_alrich_al,
                "f_eta_ge2_percent": f_eta_ge2,
                "f_eta_ge3_percent": f_eta_ge3,
                "n_ge5": n_ge5,
                "n_ge7": n_ge7,
                "max_size": max_size,
            })

            frame_index += 1

    if len(records) == 0:
        raise RuntimeError(f"No frames were read from {dump_file}")

    return pd.DataFrame(records)


def summarize_last_fraction(frame_df, frac=0.20):
    """
    Average over the last fraction of production frames.
    """
    n_total = len(frame_df)
    n_keep = max(1, int(math.ceil(n_total * frac)))

    last_df = frame_df.tail(n_keep).copy()

    metrics = [
        "n_alrich_al",
        "f_eta_ge2_percent",
        "f_eta_ge3_percent",
        "n_ge5",
        "n_ge7",
        "max_size",
    ]

    out = {
        "n_frames_total": n_total,
        "n_frames_used": n_keep,
        "timestep_start_used": int(last_df["timestep"].iloc[0]),
        "timestep_end_used": int(last_df["timestep"].iloc[-1]),
    }

    for col in metrics:
        vals = pd.to_numeric(last_df[col], errors="coerce")

        out[f"{col}_mean"] = vals.mean()
        out[f"{col}_std"] = vals.std(ddof=1) if len(vals.dropna()) > 1 else 0.0

    return out, last_df


# =========================
# Read no-shear cases and summarize last 20%
# =========================
summary_records = []
last20_frame_records = []

out_dir = root / "Processed_Data"
out_dir.mkdir(parents=True, exist_ok=True)

for fs in fs_values:
    folder = find_case_folder(fs, gdot_target)
    dump_file = find_dump_file(folder)

    print(f"Reading fs={fs}, gdot={gdot_target}: {dump_file}")

    frame_df = analyze_dump_all_frames(dump_file)

    frame_df["fs"] = fs
    frame_df["gdot"] = gdot_target
    frame_df["folder"] = folder.name
    frame_df["dump_file"] = dump_file.name

    summary, last_df = summarize_last_fraction(frame_df, frac=last_fraction)

    summary_record = {
        "fs": fs,
        "gdot": gdot_target,
        "folder": folder.name,
        "dump_file": dump_file.name,
    }
    summary_record.update(summary)

    summary_records.append(summary_record)
    last20_frame_records.append(last_df)

    print(
        f"  total frames = {summary['n_frames_total']}, "
        f"used last frames = {summary['n_frames_used']}, "
        f"timesteps = {summary['timestep_start_used']}–{summary['timestep_end_used']}"
    )

summary_df = pd.DataFrame(summary_records)
last20_frames_df = pd.concat(last20_frame_records, ignore_index=True)

out_summary_csv = out_dir / "Fig4_noshear_statistics_last20_summary.csv"
out_frames_csv = out_dir / "Fig4_noshear_statistics_last20_frames.csv"

summary_df.to_csv(out_summary_csv, index=False)
last20_frames_df.to_csv(out_frames_csv, index=False)

print("\nLast-20% summary:")
print(summary_df)
print(f"\nSaved summary to: {out_summary_csv}")
print(f"Saved last-20% per-frame data to: {out_frames_csv}")
