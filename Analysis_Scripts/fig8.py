from pathlib import Path

import numpy as np
import pandas as pd

root = Path.cwd()

out_dir = root / "Processed_Data"
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "Fig8_structural_characteristics.csv"


# =========================
# Case order
# =========================
fs_order = ["fs20", "fs40", "fs60", "fs80"]
gdot_order = [0.0, 0.001, 0.005, 0.01, 0.02]


# =========================
# stat.production column order
# Must be consistent with the LAMMPS fix ave/time output.
# =========================
STAT_COLS = [
    "step",
    "fSolid", "fHCP", "fFCC", "fUnknown",
    "nRichRegion", "fRichRegion", "nRichAl", "fRichAlOfAl",
    "avgXAlLoc", "avgEtaAl",
    "nRichSolid", "nRichUnknown",
    "sumPeRich", "sumPeMatrix", "sumPeSolid", "sumPeAll",
    "peRichAve", "peMatrixAve", "dPeRichMatrix",
    "DperpMg", "DperpAl", "DperpZn", "D3DMg", "D3DAl", "D3DZn",
]


# =========================
# Helper functions
# =========================
def parse_case_from_folder(folder_name):
    """
    Example:
      AZ91_fs60_T833.0_shearYZ_gdot0.01
    """
    parts = folder_name.split("_")
    fs = None
    T = None
    gdot = None

    for p in parts:
        if p.startswith("fs"):
            fs = p
        elif p.startswith("T"):
            T = float(p[1:])
        elif p.startswith("gdot"):
            gdot = float(p.replace("gdot", ""))

    return fs, T, gdot


def find_stat_file(fs, gdot):
    """
    Find the stat.production*.dat file for a given solid fraction and shear rate.
    """
    candidates = sorted(
        root.glob(f"AZ91_{fs}_T*_shearYZ_gdot{gdot}*/stat.production*.dat")
    )

    if not candidates:
        all_files = sorted(
            root.glob(f"AZ91_{fs}_T*_shearYZ_gdot*/stat.production*.dat")
        )

        matched = []
        for f in all_files:
            fs0, T0, g0 = parse_case_from_folder(f.parent.name)

            if fs0 == fs and abs(g0 - gdot) < 1e-12:
                matched.append(f)

        candidates = matched

    if not candidates:
        raise FileNotFoundError(f"Cannot find stat.production file for {fs}, gdot={gdot}")

    if len(candidates) > 1:
        print(f"Warning: multiple stat files found for {fs}, gdot={gdot}. Use {candidates[0]}")

    return candidates[0]


def read_stat_file(stat_file):
    """
    Read stat.production*.dat and assign column names.
    """
    stat_file = Path(stat_file)
    arr = np.loadtxt(stat_file, comments="#")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < len(STAT_COLS):
        raise RuntimeError(
            f"Unexpected column number in {stat_file}: {arr.shape[1]}, "
            f"expected at least {len(STAT_COLS)}"
        )

    df = pd.DataFrame(arr[:, :len(STAT_COLS)], columns=STAT_COLS)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    s0 = df["step"].iloc[0]
    s1 = df["step"].iloc[-1]

    if s1 > s0:
        df["progress"] = (df["step"] - s0) / (s1 - s0)
    else:
        df["progress"] = np.linspace(0, 1, len(df))

    return df


def safe_divide(a, b):
    """
    Safe division to avoid division by zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]

    return out


def fs_to_percent(fs):
    """
    Convert 'fs20' to 20.
    """
    return int(str(fs).replace("fs", ""))


# =========================
# Extract last-20% statistics
# =========================
rows = []

for fs in fs_order:
    for gdot in gdot_order:
        stat_file = find_stat_file(fs, gdot)
        df = read_stat_file(stat_file)

        late = df[df["progress"] >= 0.8].copy()

        if late.empty:
            raise RuntimeError(f"No last-20% data selected from {stat_file}")

        # Locally disordered atoms in Al-rich local environments:
        # PTM-unknown atoms in Al-rich local environments.
        f_disordered_rich = safe_divide(
            late["nRichUnknown"].values,
            late["nRichRegion"].values
        )

        # Reference disordered fraction in the whole system.
        f_disordered_overall = late["fUnknown"].values

        # Enrichment ratio of local disorder in Al-rich local environments.
        R_disordered = safe_divide(
            f_disordered_rich,
            f_disordered_overall
        )

        rows.append({
            "fs": fs_to_percent(fs),
            "gdot": gdot,

            "n_frames_total": int(len(df)),
            "n_frames_used": int(len(late)),
            "step_start_used": int(late["step"].iloc[0]),
            "step_end_used": int(late["step"].iloc[-1]),

            # Data directly used for Fig. 8
            "f_disordered_rich_percent": float(np.nanmean(f_disordered_rich) * 100.0),
            "R_disordered": float(np.nanmean(R_disordered)),
            "f_disordered_overall_percent": float(np.nanmean(f_disordered_overall) * 100.0),

            # Original names kept for traceability
            "f_unknown_rich_percent": float(np.nanmean(f_disordered_rich) * 100.0),
            "R_unknown_rich": float(np.nanmean(R_disordered)),
            "fUnknown_overall_percent": float(np.nanmean(f_disordered_overall) * 100.0),

            "source_file": str(stat_file),
        })

fig8_df = pd.DataFrame(rows)
fig8_df = fig8_df.sort_values(["fs", "gdot"]).reset_index(drop=True)

fig8_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"Saved Fig. 8 structural data to: {out_csv}")
print(fig8_df)
