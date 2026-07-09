from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FIG6_SUMMARY_NAME = "Fig6_shear_statistics_last20_summary.csv"

OUT_RELATIVE_NAME = "Fig7_relative_shear_response_data.csv"
OUT_CORRELATION_NAME = "Fig7_response_correlation.csv"


def find_default_input(script_dir: Path) -> Path:
    """
    Find Fig6_shear_statistics_last20_summary.csv from common locations.
    """
    data_root = script_dir.parent

    candidates = [
        script_dir / "Processed_Data" / FIG6_SUMMARY_NAME,
        data_root / "Processed_Data" / FIG6_SUMMARY_NAME,
        Path.cwd() / "Processed_Data" / FIG6_SUMMARY_NAME,
        Path.cwd() / FIG6_SUMMARY_NAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    msg = "Cannot find Fig6_shear_statistics_last20_summary.csv. Tried:\n"
    msg += "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(msg)


def default_outdir(script_dir: Path) -> Path:
    """
    Default output folder is the Processed_Data folder in the project root.
    """
    return script_dir.parent / "Processed_Data"

def validate_fig6_summary(df: pd.DataFrame, input_csv: Path) -> None:
    required_cols = {
        "fs",
        "gdot",
        "n_alrich_al_mean",
        "f_eta_ge3_percent_mean",
        "n_ge5_mean",
        "max_size_mean",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Input file is missing required columns: {sorted(missing)}\n"
            f"Input file: {input_csv}\n"
            f"Available columns: {list(df.columns)}"
        )


def build_relative_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative shear response using gdot = 0 as the baseline
    for each solid fraction.

    relative_change_percent = (value - baseline) / baseline * 100
    """
    df = df.copy()

    fs_values = [20, 40, 60, 80]
    gdot_values = [0.001, 0.005, 0.01, 0.02]

    metric_columns = {
        "alrich_al_atoms": "n_alrich_al_mean",
        "f_eta_ge3": "f_eta_ge3_percent_mean",
        "n_clusters_ge5": "n_ge5_mean",
        "nmax": "max_size_mean",
    }

    records = []

    for fs in fs_values:
        sub_fs = df[df["fs"] == fs].copy()

        if sub_fs.empty:
            raise RuntimeError(f"No rows found for fs = {fs} in Fig. 6 summary.")

        base_row = sub_fs[np.isclose(sub_fs["gdot"], 0.0)]

        if base_row.empty:
            raise RuntimeError(f"No gdot = 0 baseline row found for fs = {fs}.")

        base_row = base_row.iloc[0]

        for gdot in gdot_values:
            row = sub_fs[np.isclose(sub_fs["gdot"], gdot)]

            if row.empty:
                raise RuntimeError(f"No row found for fs = {fs}, gdot = {gdot}.")

            row = row.iloc[0]

            rec = {
                "fs": fs,
                "gdot": gdot,
            }

            for metric_name, col in metric_columns.items():
                base_value = float(base_row[col])
                value = float(row[col])

                if abs(base_value) < 1e-12:
                    response = np.nan
                else:
                    response = (value - base_value) / base_value * 100.0

                rec[f"{metric_name}_value"] = value
                rec[f"{metric_name}_baseline"] = base_value
                rec[f"{metric_name}_relative_change_percent"] = response

            records.append(rec)

    response_df = pd.DataFrame(records)
    response_df = response_df.sort_values(["fs", "gdot"]).reset_index(drop=True)

    return response_df


def build_correlation_data(relative_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns directly used for Fig. 7 response-correlation plots.
    """
    cols = [
        "fs",
        "gdot",
        "alrich_al_atoms_relative_change_percent",
        "f_eta_ge3_relative_change_percent",
        "n_clusters_ge5_relative_change_percent",
        "nmax_relative_change_percent",
    ]

    return relative_df[cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fig. 7 response-correlation CSV files from Fig. 6 summary data."
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Path to Fig6_shear_statistics_last20_summary.csv. Default: auto search.",
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: the Processed_Data folder next to this script.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.input is None:
        input_csv = find_default_input(script_dir)
    else:
        input_csv = Path(args.input).resolve()

    if args.outdir is None:
        out_dir = default_outdir(script_dir)
    else:
        out_dir = Path(args.outdir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    out_relative_csv = out_dir / OUT_RELATIVE_NAME
    out_correlation_csv = out_dir / OUT_CORRELATION_NAME

    print("=" * 80)
    print("Fig. 7 response-correlation data extraction")
    print(f"Input Fig. 6 summary : {input_csv}")
    print(f"Output folder        : {out_dir}")
    print(f"Relative response CSV: {out_relative_csv.name}")
    print(f"Correlation CSV      : {out_correlation_csv.name}")
    print("=" * 80)

    fig6_df = pd.read_csv(input_csv)
    validate_fig6_summary(fig6_df, input_csv)

    relative_df = build_relative_response(fig6_df)
    correlation_df = build_correlation_data(relative_df)

    relative_df.to_csv(out_relative_csv, index=False, encoding="utf-8-sig")
    correlation_df.to_csv(out_correlation_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_relative_csv}")
    print(f"Saved: {out_correlation_csv}")

    print("\nCorrelation data preview:")
    with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 200):
        print(correlation_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
