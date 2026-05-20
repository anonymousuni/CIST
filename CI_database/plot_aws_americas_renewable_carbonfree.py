#!/usr/bin/env python3
"""
Plot Renewable Energy Percentage and Carbon-Free Energy Percentage
for AWS Americas regions using the Electricity Maps CSV downloaded earlier.

It creates:
  1) all regions together plots
  2) one separate plot per region

Default input directory:
  electricitymaps_aws_americas_72h_past_now_future_all_signals/

Usage:
  python3 plot_aws_americas_renewable_carbonfree.py

Optional:
  python3 plot_aws_americas_renewable_carbonfree.py --input-dir YOUR_FOLDER
  python3 plot_aws_americas_renewable_carbonfree.py --local-timezone America/Chicago
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SIGNAL_ALIASES = {
    "renewable": {
        "pretty": "Renewable Energy Percentage",
        "file_slug": "renewable_energy_percentage",
        "possible_signal_names": {
            "renewable_percentage",
            "renewable_energy_percentage",
            "renewableEnergyPercentage",
            "renewablePercentage",
        },
        "possible_value_columns": {
            "renewablePercentage",
            "renewableEnergyPercentage",
            "renewable_percentage",
            "renewable_energy_percentage",
            "value",
        },
        "ylabel": "Renewable Energy Percentage (%)",
    },
    "carbon_free": {
        "pretty": "Carbon-Free Energy Percentage",
        "file_slug": "carbon_free_energy_percentage",
        "possible_signal_names": {
            "carbon_free_percentage",
            "carbon_free_energy_percentage",
            "carbonFreePercentage",
            "carbonFreeEnergyPercentage",
        },
        "possible_value_columns": {
            "carbonFreePercentage",
            "carbonFreeEnergyPercentage",
            "carbon_free_percentage",
            "carbon_free_energy_percentage",
            "value",
        },
        "ylabel": "Carbon-Free Energy Percentage (%)",
    },
}


def find_combined_csv(input_dir: Path) -> Path:
    candidates = [
        input_dir / "all_aws_americas_signals_past_latest_forecast.csv",
        input_dir / "all_aws_us_signals_past_latest_forecast.csv",
        input_dir / "all_aws_us_past_latest_forecast_ci_72h_past_72h_future_5_minutes.csv",
        input_dir / "all_aws_us_ci_cleaned_for_plot.csv",
    ]

    for c in candidates:
        if c.exists():
            return c

    # Fallback: choose the largest CSV that looks like a combined signal table
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    preferred = [f for f in csv_files if "signal" in f.name.lower() or "past_latest_forecast" in f.name.lower()]
    if preferred:
        return max(preferred, key=lambda p: p.stat().st_size)

    return max(csv_files, key=lambda p: p.stat().st_size)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Region column
    if "region" not in df.columns:
        for c in ["dataCenterRegion", "data_center_region", "aws_region"]:
            if c in df.columns:
                df["region"] = df[c]
                break
    if "region" not in df.columns:
        raise ValueError(f"Could not find region column. Columns are: {list(df.columns)}")

    # Datetime column
    if "datetime" not in df.columns:
        for c in ["time", "timestamp", "startTime", "updatedAt"]:
            if c in df.columns:
                df["datetime"] = df[c]
                break
    if "datetime" not in df.columns:
        raise ValueError(f"Could not find datetime column. Columns are: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return df


def extract_signal_df(df: pd.DataFrame, signal_key: str) -> pd.DataFrame:
    meta = SIGNAL_ALIASES[signal_key]
    possible_signal_names = meta["possible_signal_names"]
    possible_value_columns = meta["possible_value_columns"]

    # Case 1: long format with columns signal + value
    signal_col = None
    for c in ["signal", "metric", "dataType", "type"]:
        if c in df.columns:
            signal_col = c
            break

    if signal_col is not None:
        mask = df[signal_col].astype(str).isin(possible_signal_names)
        out = df.loc[mask].copy()

        # Find value column
        value_col = None
        for c in possible_value_columns:
            if c in out.columns:
                value_col = c
                break
        if value_col is None:
            for c in ["percentage", "value", "carbonIntensity"]:
                if c in out.columns:
                    value_col = c
                    break
        if value_col is None:
            raise ValueError(
                f"Could not find value column for {meta['pretty']}. Columns are: {list(out.columns)}"
            )

        out["value"] = pd.to_numeric(out[value_col], errors="coerce")
        out = out.dropna(subset=["datetime", "region", "value"])
        return out[["region", "datetime", "value"]].copy()

    # Case 2: wide format with direct percentage columns
    value_col = None
    for c in possible_value_columns:
        if c in df.columns:
            value_col = c
            break

    if value_col is None:
        # Try fuzzy matching
        lower_map = {c.lower(): c for c in df.columns}
        for target in possible_value_columns:
            target_norm = target.lower().replace("_", "")
            for lower_name, original_name in lower_map.items():
                if lower_name.replace("_", "") == target_norm:
                    value_col = original_name
                    break
            if value_col:
                break

    if value_col is None:
        raise ValueError(
            f"Could not find {meta['pretty']} in this CSV. Columns are: {list(df.columns)}"
        )

    out = df.copy()
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["datetime", "region", "value"])
    return out[["region", "datetime", "value"]].copy()


def make_all_regions_plot(signal_df: pd.DataFrame, signal_key: str, output_dir: Path, local_timezone: str | None) -> None:
    meta = SIGNAL_ALIASES[signal_key]
    plot_df = signal_df.copy()

    xlabel = "Time (UTC)"
    now = pd.Timestamp.now(tz="UTC")

    if local_timezone:
        plot_df["plot_time"] = plot_df["datetime"].dt.tz_convert(local_timezone)
        now = now.tz_convert(local_timezone)
        xlabel = f"Time ({local_timezone})"
    else:
        plot_df["plot_time"] = plot_df["datetime"]

    plot_df = plot_df.drop_duplicates(subset=["region", "plot_time"])
    plot_df = plot_df.sort_values(["region", "plot_time"])

    plt.figure(figsize=(14, 7))
    for region in sorted(plot_df["region"].unique()):
        r = plot_df[plot_df["region"] == region]
        plt.plot(r["plot_time"], r["value"], linewidth=1.7, label=region)

    plt.axvline(now, linestyle="--", linewidth=1.4, label="now")
    plt.xlabel(xlabel)
    plt.ylabel(meta["ylabel"])
    plt.title(f"AWS Americas Regions: {meta['pretty']} from Past 72h to Future 72h")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{meta['file_slug']}_all_regions.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def make_separate_region_plots(signal_df: pd.DataFrame, signal_key: str, output_dir: Path, local_timezone: str | None) -> None:
    meta = SIGNAL_ALIASES[signal_key]
    plot_df = signal_df.copy()

    xlabel = "Time (UTC)"
    now = pd.Timestamp.now(tz="UTC")

    if local_timezone:
        plot_df["plot_time"] = plot_df["datetime"].dt.tz_convert(local_timezone)
        now = now.tz_convert(local_timezone)
        xlabel = f"Time ({local_timezone})"
    else:
        plot_df["plot_time"] = plot_df["datetime"]

    plot_df = plot_df.drop_duplicates(subset=["region", "plot_time"])
    plot_df = plot_df.sort_values(["region", "plot_time"])

    signal_out_dir = output_dir / meta["file_slug"]
    signal_out_dir.mkdir(parents=True, exist_ok=True)

    for region in sorted(plot_df["region"].unique()):
        r = plot_df[plot_df["region"] == region]

        plt.figure(figsize=(13, 5))
        plt.plot(r["plot_time"], r["value"], linewidth=1.8, marker=".", markersize=2, label=region)
        plt.axvline(now, linestyle="--", linewidth=1.4, label="now")
        plt.xlabel(xlabel)
        plt.ylabel(meta["ylabel"])
        plt.title(f"AWS {region}: {meta['pretty']} from Past 72h to Future 72h")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        safe_region = str(region).replace("/", "_")
        out = signal_out_dir / f"aws_{safe_region}_{meta['file_slug']}_past72_future72.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="electricitymaps_aws_americas_72h_past_now_future_all_signals",
        help="Folder containing the combined Electricity Maps CSV.",
    )
    parser.add_argument(
        "--local-timezone",
        default=None,
        help="Optional timezone for x-axis, e.g., America/Chicago. Default uses UTC.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_path = find_combined_csv(input_dir)
    print(f"Reading: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    all_regions_dir = input_dir / "plots" / "all_regions_together"
    separate_dir = input_dir / "plots" / "separate_by_signal_and_region"

    for signal_key in ["renewable", "carbon_free"]:
        signal_df = extract_signal_df(df, signal_key)
        signal_df = signal_df.drop_duplicates(subset=["region", "datetime"])
        signal_df = signal_df.sort_values(["region", "datetime"])

        meta = SIGNAL_ALIASES[signal_key]
        cleaned_csv = input_dir / f"cleaned_{meta['file_slug']}_for_plot.csv"
        signal_df.to_csv(cleaned_csv, index=False)
        print(f"Saved cleaned CSV: {cleaned_csv}")

        print(f"\nSummary for {meta['pretty']}:")
        print(
            signal_df.groupby("region").agg(
                rows=("value", "count"),
                min_value=("value", "min"),
                max_value=("value", "max"),
                first_time=("datetime", "min"),
                last_time=("datetime", "max"),
            )
        )

        make_all_regions_plot(signal_df, signal_key, all_regions_dir, args.local_timezone)
        make_separate_region_plots(signal_df, signal_key, separate_dir, args.local_timezone)

    print("\nDone.")
    print(f"All-regions plots: {all_regions_dir}")
    print(f"Separate plots:    {separate_dir}")


if __name__ == "__main__":
    main()
