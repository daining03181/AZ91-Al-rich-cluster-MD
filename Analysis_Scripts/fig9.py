from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# Basic parsing utilities
# ============================================================
def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def is_number_token(s: str) -> bool:
    return re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", str(s)) is not None


def parse_case_info(folder_name: str) -> Dict[str, Any]:
    """
    Example:
        AZ91_fs60_T833.0_shearYZ_gdot0.01
    """
    info: Dict[str, Any] = {
        "folder": folder_name,
        "alloy": "",
        "fs": np.nan,
        "T": np.nan,
        "gdot": np.nan,
    }

    m = re.search(
        r"(?P<alloy>AZ\d+)_+(?:fs)?(?P<fs_label>fs\d+)_+T(?P<T>[0-9.]+).*?gdot(?P<gdot>[0-9.]+)",
        folder_name,
        flags=re.IGNORECASE,
    )

    if m:
        gd = m.groupdict()
        info["alloy"] = gd["alloy"]
        info["T"] = float(gd["T"])
        info["gdot"] = float(gd["gdot"])

        mfs = re.search(r"fs(\d+)", gd["fs_label"], flags=re.IGNORECASE)
        if mfs:
            info["fs"] = int(mfs.group(1))

    return info


def find_stat_files(root: Path) -> List[Path]:
    return sorted(root.glob("AZ91_fs*_T*_shearYZ_gdot*/stat.production*.dat"))


def looks_like_header(tokens: List[str]) -> bool:
    """
    Accept possible stat headers such as:
        # TimeStep v_peRichAve v_peMatrixAve v_dPeRichMatrix
        # Step ...
        # timestep ...
    """
    if not tokens:
        return False

    norm = normalize_name(" ".join(tokens))

    key_terms = [
        "step",
        "timestep",
        "time",
        "perichave",
        "pematrixave",
        "dperichmatrix",
    ]

    return any(k in norm for k in key_terms)


def read_stat_file(path: Path) -> pd.DataFrame:
    """
    Robust whitespace reader for stat.production*.dat.

    It uses the last header-like line before numeric data and reads all
    following numeric rows.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    header_tokens: Optional[List[str]] = None
    data_lines: List[str] = []

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        if raw.startswith("#"):
            content = raw.lstrip("#").strip()
            toks = content.split()

            if looks_like_header(toks):
                header_tokens = toks

            continue

        toks = raw.split()

        if not toks:
            continue

        if is_number_token(toks[0]):
            data_lines.append(raw)
        else:
            if looks_like_header(toks):
                header_tokens = toks

    if header_tokens is None:
        preview = [ln.strip() for ln in lines if ln.strip()][:10]
        raise RuntimeError(
            f"Cannot find header line in {path}. "
            f"First non-empty lines: {' | '.join(preview)}"
        )

    if not data_lines:
        raise RuntimeError(f"No numeric data lines found in {path}")

    rows = []
    for line in data_lines:
        toks = line.split()

        if len(toks) >= len(header_tokens):
            rows.append(toks[:len(header_tokens)])

    if not rows:
        raise RuntimeError(
            f"Numeric rows do not match header length in {path}. "
            f"Header has {len(header_tokens)} columns: {header_tokens}"
        )

    df = pd.DataFrame(rows, columns=header_tokens)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)

    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {normalize_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_name(cand)

        if key in norm_map:
            return norm_map[key]

    for col in df.columns:
        col_norm = normalize_name(col)

        for cand in candidates:
            cand_norm = normalize_name(cand)

            if col_norm.endswith(cand_norm) or cand_norm.endswith(col_norm) or cand_norm in col_norm:
                return col

    return None


# ============================================================
# Energy extraction
# ============================================================
def summarize_energy_for_fig9(df: pd.DataFrame, last_frac: float) -> Dict[str, Any]:
    n_total = len(df)
    n_used = max(1, int(round(n_total * last_frac)))
    late = df.tail(n_used).copy()

    wanted = {
        "Step": ["Step", "step", "TimeStep", "timestep", "Time"],
        "peRichAve": ["peRichAve", "v_peRichAve", "v_perichave", "peRich", "pe_rich_ave"],
        "peMatrixAve": ["peMatrixAve", "v_peMatrixAve", "v_pematrixave", "peMatrix", "pe_matrix_ave"],
        "dPeRichMatrix": [
            "dPeRichMatrix",
            "v_dPeRichMatrix",
            "v_dperichmatrix",
            "dPe",
            "deltaPe",
            "peRichMinusMatrix",
        ],
    }

    step_col = find_column(df, wanted["Step"])
    pe_rich_col = find_column(df, wanted["peRichAve"])
    pe_matrix_col = find_column(df, wanted["peMatrixAve"])
    dpe_col = find_column(df, wanted["dPeRichMatrix"])

    if pe_rich_col is None:
        raise RuntimeError(f"Cannot find peRichAve column. Available columns: {list(df.columns)}")

    if pe_matrix_col is None:
        raise RuntimeError(f"Cannot find peMatrixAve column. Available columns: {list(df.columns)}")

    pe_rich_tail = pd.to_numeric(late[pe_rich_col], errors="coerce")
    pe_matrix_tail = pd.to_numeric(late[pe_matrix_col], errors="coerce")

    if dpe_col is not None:
        dpe_tail = pd.to_numeric(late[dpe_col], errors="coerce")
        dpe_source = dpe_col
    else:
        dpe_tail = pe_rich_tail - pe_matrix_tail
        dpe_source = "calculated: peRichAve - peMatrixAve"

    out: Dict[str, Any] = {
        "n_frames_total": int(n_total),
        "n_frames_used": int(n_used),
        "last_fraction": float(last_frac),

        "peRichAve_col": pe_rich_col,
        "peMatrixAve_col": pe_matrix_col,
        "dPeRichMatrix_col": dpe_source,

        "E_rich_eV_atom": float(pe_rich_tail.mean()),
        "E_rich_std_eV_atom": float(pe_rich_tail.std(ddof=1)) if pe_rich_tail.dropna().size > 1 else 0.0,

        "E_matrix_eV_atom": float(pe_matrix_tail.mean()),
        "E_matrix_std_eV_atom": float(pe_matrix_tail.std(ddof=1)) if pe_matrix_tail.dropna().size > 1 else 0.0,

        "dE_rich_matrix_eV_atom": float(dpe_tail.mean()),
        "dE_rich_matrix_std_eV_atom": float(dpe_tail.std(ddof=1)) if dpe_tail.dropna().size > 1 else 0.0,
    }

    if step_col is not None:
        out["step_start_used"] = float(late[step_col].iloc[0])
        out["step_end_used"] = float(late[step_col].iloc[-1])

    return out


def add_baseline_response(fig9_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the baseline and relative energy-response columns used for traceability.
    The direct Fig. 9 panels use E_rich, E_matrix, and dE.
    """
    out = fig9_df.copy()

    out["dE_baseline_eV_atom"] = np.nan
    out["delta_dE_vs_noshear_eV_atom"] = np.nan
    out["energy_stabilization_eV_atom"] = np.nan

    for fs in sorted(out["fs"].dropna().unique()):
        sub = out[out["fs"] == fs]

        base = sub[np.isclose(sub["gdot"], 0.0)]

        if base.empty:
            print(f"Warning: no gdot = 0 baseline found for fs = {fs}.")
            continue

        base_dE = float(base.iloc[0]["dE_rich_matrix_eV_atom"])
        mask = out["fs"] == fs

        out.loc[mask, "dE_baseline_eV_atom"] = base_dE
        out.loc[mask, "delta_dE_vs_noshear_eV_atom"] = out.loc[mask, "dE_rich_matrix_eV_atom"] - base_dE
        out.loc[mask, "energy_stabilization_eV_atom"] = -out.loc[mask, "delta_dE_vs_noshear_eV_atom"]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Fig. 9 energy-characteristics data from stat.production*.dat files."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root folder containing AZ91_fs*_T*_shearYZ_gdot* directories.",
    )
    parser.add_argument(
        "--last-frac",
        type=float,
        default=0.2,
        help="Average over the last fraction of rows. Default: 0.2.",
    )
    parser.add_argument(
        "--out",
        default="Processed_Data/Fig9_energy_characteristics.csv",
        help="Output CSV path. Default: Processed_Data/Fig9_energy_characteristics.csv",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)

    if not out_path.is_absolute():
        out_path = root / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stat_files = find_stat_files(root)

    if not stat_files:
        raise FileNotFoundError(
            f"No stat.production*.dat files found under: {root}\n"
            "Expected folder pattern: AZ91_fs*_T*_shearYZ_gdot*/stat.production*.dat"
        )

    rows: List[Dict[str, Any]] = []

    print(f"Found {len(stat_files)} stat.production files.")
    print(f"Average over last fraction = {args.last_frac}")

    for idx, stat_file in enumerate(stat_files, start=1):
        folder_name = stat_file.parent.name
        case_info = parse_case_info(folder_name)

        print(f"[{idx}/{len(stat_files)}] {folder_name}")

        row: Dict[str, Any] = {}
        row.update(case_info)
        row["stat_file"] = str(stat_file.relative_to(root))

        try:
            df = read_stat_file(stat_file)
            row.update(summarize_energy_for_fig9(df, last_frac=args.last_frac))
            print(
                f"  OK: E_rich={row['E_rich_eV_atom']:.6g}, "
                f"E_matrix={row['E_matrix_eV_atom']:.6g}, "
                f"dE={row['dE_rich_matrix_eV_atom']:.6g}"
            )

        except Exception as exc:
            row["error"] = repr(exc)
            print(f"  ERROR: {repr(exc)}")

        rows.append(row)

    fig9_df = pd.DataFrame(rows)

    # Keep only the four solid fractions and five shear rates used in the manuscript,
    # when parsing was successful.
    if "fs" in fig9_df.columns and "gdot" in fig9_df.columns:
        fig9_df = fig9_df[
            fig9_df["fs"].isin([20, 40, 60, 80]) &
            fig9_df["gdot"].isin([0.0, 0.001, 0.005, 0.01, 0.02])
        ].copy()

        fig9_df = fig9_df.sort_values(["fs", "gdot"]).reset_index(drop=True)

    fig9_df = add_baseline_response(fig9_df)

    # Put the most important columns first.
    preferred_cols = [
        "fs", "T", "gdot",
        "folder", "stat_file",
        "n_frames_total", "n_frames_used", "last_fraction",
        "step_start_used", "step_end_used",
        "E_rich_eV_atom", "E_rich_std_eV_atom",
        "E_matrix_eV_atom", "E_matrix_std_eV_atom",
        "dE_rich_matrix_eV_atom", "dE_rich_matrix_std_eV_atom",
        "dE_baseline_eV_atom",
        "delta_dE_vs_noshear_eV_atom",
        "energy_stabilization_eV_atom",
        "peRichAve_col", "peMatrixAve_col", "dPeRichMatrix_col",
        "error",
    ]

    ordered_cols = [c for c in preferred_cols if c in fig9_df.columns]
    remaining_cols = [c for c in fig9_df.columns if c not in ordered_cols]
    fig9_df = fig9_df[ordered_cols + remaining_cols]

    fig9_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n============================================================")
    print(f"Saved Fig. 9 energy-characteristics data to: {out_path}")
    print("Direct Fig. 9 columns:")
    print("  E_rich_eV_atom")
    print("  E_matrix_eV_atom")
    print("  dE_rich_matrix_eV_atom")
    print("============================================================")

    with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 240):
        cols_show = [
            c for c in [
                "fs", "T", "gdot",
                "E_rich_eV_atom",
                "E_matrix_eV_atom",
                "dE_rich_matrix_eV_atom",
                "error",
            ]
            if c in fig9_df.columns
        ]
        print(fig9_df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
