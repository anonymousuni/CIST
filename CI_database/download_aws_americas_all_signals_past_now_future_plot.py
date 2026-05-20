#!/usr/bin/env python3
"""
Download Electricity Maps signals for AWS regions and plot them.

Signals:
  1) Carbon Intensity
  2) Renewable Energy Percentage
  3) Carbon-Free Energy Percentage

Time coverage:
  - past N hours, downloaded in chunks because 5-minute history has range limits
  - latest/current
  - future N hours forecast

Default regions:
  AWS U.S. + Canada + South America commercial regions

Usage:
  export EMAPS_TOKEN="YOUR_API_KEY"
  python3 download_aws_americas_all_signals_past_now_future_plot_v5.py

Plot only from existing combined CSV:
  python3 download_aws_americas_all_signals_past_now_future_plot_v5.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import matplotlib.pyplot as plt

BASE_URL = "https://api.electricitymap.org/v4"

DEFAULT_AWS_REGIONS = [
    # United States
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    # Canada
    "ca-central-1",
    "ca-west-1",
    # South America
    "sa-east-1",
]


@dataclass(frozen=True)
class EndpointCandidate:
    endpoint: str
    field_candidates: Tuple[str, ...]
    include_emission_factor_type: bool = False


@dataclass(frozen=True)
class SignalConfig:
    key: str
    label: str
    y_label: str
    unit: str
    candidates: Tuple[EndpointCandidate, ...]


SIGNALS: Dict[str, SignalConfig] = {
    "carbon_intensity": SignalConfig(
        key="carbon_intensity",
        label="Carbon Intensity",
        y_label="Carbon Intensity (gCO2eq/kWh)",
        unit="gCO2eq/kWh",
        candidates=(
            EndpointCandidate(
                endpoint="carbon-intensity",
                field_candidates=("carbonIntensity",),
                include_emission_factor_type=True,
            ),
        ),
    ),
    "renewable_energy_percentage": SignalConfig(
        key="renewable_energy_percentage",
        label="Renewable Energy Percentage",
        y_label="Renewable Energy Percentage (%)",
        unit="%",
        candidates=(
            # Main signal endpoint names used by the new API/docs/playground.
            EndpointCandidate(
                endpoint="renewable-energy-percentage",
                field_candidates=(
                    "renewablePercentage",
                    "renewableEnergyPercentage",
                    "percentage",
                    "value",
                ),
            ),
            # Fallback names, in case the account/API uses older or shorter slugs.
            EndpointCandidate(
                endpoint="renewable-percentage",
                field_candidates=(
                    "renewablePercentage",
                    "renewableEnergyPercentage",
                    "percentage",
                    "value",
                ),
            ),
            # Power-breakdown often includes renewablePercentage.
            EndpointCandidate(
                endpoint="power-breakdown",
                field_candidates=(
                    "renewablePercentage",
                    "renewableEnergyPercentage",
                ),
            ),
        ),
    ),
    "carbon_free_energy_percentage": SignalConfig(
        key="carbon_free_energy_percentage",
        label="Carbon-Free Energy Percentage",
        y_label="Carbon-Free Energy Percentage (%)",
        unit="%",
        candidates=(
            # Main signal endpoint names used by the new API/docs/playground.
            EndpointCandidate(
                endpoint="carbon-free-energy-percentage",
                field_candidates=(
                    "carbonFreePercentage",
                    "carbonFreeEnergyPercentage",
                    "fossilFreePercentage",
                    "percentage",
                    "value",
                ),
            ),
            # Fallback names.
            EndpointCandidate(
                endpoint="carbon-free-percentage",
                field_candidates=(
                    "carbonFreePercentage",
                    "carbonFreeEnergyPercentage",
                    "fossilFreePercentage",
                    "percentage",
                    "value",
                ),
            ),
            EndpointCandidate(
                endpoint="fossil-free-percentage",
                field_candidates=(
                    "fossilFreePercentage",
                    "carbonFreePercentage",
                    "carbonFreeEnergyPercentage",
                    "percentage",
                    "value",
                ),
            ),
            # Older power-breakdown responses use fossilFreePercentage.
            EndpointCandidate(
                endpoint="power-breakdown",
                field_candidates=(
                    "carbonFreePercentage",
                    "carbonFreeEnergyPercentage",
                    "fossilFreePercentage",
                ),
            ),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and plot Electricity Maps CI, renewable %, and carbon-free % for AWS regions."
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_AWS_REGIONS,
        help="AWS regions to query. Default: AWS US + Canada + South America commercial regions.",
    )
    parser.add_argument("--past-hours", type=int, default=72)
    parser.add_argument("--future-hours", type=int, default=72)
    parser.add_argument("--past-chunk-hours", type=int, default=24)
    parser.add_argument(
        "--granularity",
        default="5_minutes",
        choices=["5_minutes", "15_minutes", "hourly"],
    )
    parser.add_argument(
        "--emission-factor-type",
        default="lifecycle",
        choices=["lifecycle", "direct"],
        help="Used only for carbon intensity.",
    )
    parser.add_argument(
        "--output-dir",
        default="electricitymaps_aws_americas_72h_past_now_future_all_signals",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between requests.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--local-timezone",
        default=None,
        help="Optional timezone for plots, e.g., America/Chicago. Data CSV remains UTC.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip download and only plot from existing combined CSV.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download data but skip plotting.",
    )
    return parser.parse_args()


def granularity_minutes(granularity: str) -> int:
    if granularity == "5_minutes":
        return 5
    if granularity == "15_minutes":
        return 15
    if granularity == "hourly":
        return 60
    raise ValueError(f"Unsupported granularity: {granularity}")


def floor_datetime(dt: datetime, minutes: int) -> datetime:
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    total_minutes = dt.hour * 60 + dt.minute
    floored_total = (total_minutes // minutes) * minutes
    hour = floored_total // 60
    minute = floored_total % 60
    return dt.replace(hour=hour, minute=minute)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace("-", "-")


def request_json(
    endpoint: str,
    temporality: str,
    region: str,
    token: str,
    args: argparse.Namespace,
    extra_params: Optional[Dict[str, Any]] = None,
    include_emission_factor_type: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    url = f"{BASE_URL}/{endpoint}/{temporality}"

    params: Dict[str, Any] = {
        "dataCenterProvider": "aws",
        "dataCenterRegion": region,
    }

    if temporality in {"forecast", "past-range"}:
        params["temporalGranularity"] = args.granularity

    if temporality == "forecast":
        params["horizonHours"] = args.future_hours

    if include_emission_factor_type:
        params["emissionFactorType"] = args.emission_factor_type

    if extra_params:
        params.update(extra_params)

    headers = {"auth-token": token}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=args.timeout)
    except Exception as exc:  # network / timeout
        return None, {
            "endpoint": endpoint,
            "temporality": temporality,
            "region": region,
            "url": url,
            "params": params,
            "exception": repr(exc),
        }

    if r.status_code != 200:
        return None, {
            "endpoint": endpoint,
            "temporality": temporality,
            "region": region,
            "url": url,
            "params": params,
            "status_code": r.status_code,
            "response_text": r.text[:2000],
        }

    try:
        return r.json(), None
    except Exception as exc:
        return None, {
            "endpoint": endpoint,
            "temporality": temporality,
            "region": region,
            "url": url,
            "params": params,
            "status_code": r.status_code,
            "response_text": r.text[:2000],
            "exception": f"JSON decode failed: {exc!r}",
        }


def find_records(data: Dict[str, Any], temporality: str) -> List[Dict[str, Any]]:
    if temporality == "latest":
        return [data]

    # Different endpoints use different list keys.
    candidate_keys = [
        "forecast",
        "history",
        "pastRange",
        "data",
        "items",
        "records",
    ]
    for key in candidate_keys:
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    # Some APIs may directly return a list, but request_json expects dict. Kept for safety.
    return []


def extract_value(record: Dict[str, Any], field_candidates: Iterable[str]) -> Tuple[Optional[float], Optional[str]]:
    for field in field_candidates:
        if field in record and record[field] is not None:
            try:
                return float(record[field]), field
            except Exception:
                return None, field
    return None, None


def response_to_rows(
    data: Dict[str, Any],
    signal: SignalConfig,
    candidate: EndpointCandidate,
    temporality: str,
    region: str,
) -> List[Dict[str, Any]]:
    records = find_records(data, temporality)
    rows: List[Dict[str, Any]] = []

    for rec in records:
        value, source_field = extract_value(rec, candidate.field_candidates)
        dt = rec.get("datetime") or rec.get("createdAt") or data.get("datetime") or data.get("createdAt")
        if value is None or dt is None:
            continue

        rows.append(
            {
                "signal_key": signal.key,
                "signal_label": signal.label,
                "region": region,
                "datetime": dt,
                "value": value,
                "unit": signal.unit,
                "temporality": temporality,
                "zone": data.get("zone") or rec.get("zone"),
                "dataCenterProvider": data.get("dataCenterProvider"),
                "dataCenterRegion": data.get("dataCenterRegion") or region,
                "endpoint_used": candidate.endpoint,
                "field_used": source_field,
            }
        )

    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def try_candidates_for_one_request(
    signal: SignalConfig,
    temporality: str,
    region: str,
    token: str,
    args: argparse.Namespace,
    extra_params: Optional[Dict[str, Any]],
    raw_dir: Path,
    tag: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Try endpoint candidates until one returns usable rows."""
    errors: List[Dict[str, Any]] = []

    for candidate in signal.candidates:
        data, error = request_json(
            endpoint=candidate.endpoint,
            temporality=temporality,
            region=region,
            token=token,
            args=args,
            extra_params=extra_params,
            include_emission_factor_type=candidate.include_emission_factor_type,
        )
        time.sleep(args.sleep)

        if error is not None:
            errors.append({"signal": signal.key, **error})
            continue

        rows = response_to_rows(data, signal, candidate, temporality, region)
        raw_path = raw_dir / signal.key / temporality / f"aws_{region}_{tag}_{candidate.endpoint}.json"
        save_json(raw_path, data)

        if rows:
            return rows, {
                "signal": signal.key,
                "region": region,
                "temporality": temporality,
                "endpoint_used": candidate.endpoint,
                "rows": len(rows),
                "raw_path": str(raw_path),
            }, errors

        errors.append(
            {
                "signal": signal.key,
                "endpoint": candidate.endpoint,
                "temporality": temporality,
                "region": region,
                "reason": "Request succeeded but no usable rows were extracted.",
                "available_top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
                "raw_path": str(raw_path),
                "field_candidates": list(candidate.field_candidates),
            }
        )

    return [], None, errors


def build_past_chunks(end: datetime, past_hours: int, chunk_hours: int) -> List[Tuple[datetime, datetime]]:
    start = end - timedelta(hours=past_hours)
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(hours=chunk_hours), end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def download_all(args: argparse.Namespace, token: str, out_dir: Path) -> pd.DataFrame:
    raw_dir = out_dir / "raw_json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    now_floor = floor_datetime(datetime.now(timezone.utc), granularity_minutes(args.granularity))
    past_chunks = build_past_chunks(now_floor, args.past_hours, args.past_chunk_hours)

    print("Downloading Electricity Maps signals for AWS regions")
    print("Regions:", ", ".join(args.regions))
    print("Signals:", ", ".join(s.label for s in SIGNALS.values()))
    print("Past range:", iso_z(now_floor - timedelta(hours=args.past_hours)), "to", iso_z(now_floor))
    print("Future horizon:", args.future_hours, "hours")
    print("Granularity:", args.granularity)
    print("Output directory:", out_dir)
    print()

    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    successes: List[Dict[str, Any]] = []

    for region in args.regions:
        print(f"=== {region} ===")
        for signal in SIGNALS.values():
            print(f"  Signal: {signal.label}")

            # Past in chunks
            past_rows_total = 0
            for idx, (start, end) in enumerate(past_chunks, start=1):
                extra = {"start": iso_z(start), "end": iso_z(end)}
                tag = f"past_chunk_{idx}_{iso_z(start).replace(':','').replace('-','').replace('T','_').replace('Z','Z')}_to_{iso_z(end).replace(':','').replace('-','').replace('T','_').replace('Z','Z')}"
                rows, success, errors = try_candidates_for_one_request(
                    signal=signal,
                    temporality="past-range",
                    region=region,
                    token=token,
                    args=args,
                    extra_params=extra,
                    raw_dir=raw_dir,
                    tag=tag,
                )
                failures.extend(errors)
                if success:
                    successes.append(success)
                all_rows.extend(rows)
                past_rows_total += len(rows)

            print(f"    past rows: {past_rows_total}")

            # Latest
            rows, success, errors = try_candidates_for_one_request(
                signal=signal,
                temporality="latest",
                region=region,
                token=token,
                args=args,
                extra_params=None,
                raw_dir=raw_dir,
                tag="latest",
            )
            failures.extend(errors)
            if success:
                successes.append(success)
            all_rows.extend(rows)
            print(f"    latest rows: {len(rows)}")

            # Forecast
            rows, success, errors = try_candidates_for_one_request(
                signal=signal,
                temporality="forecast",
                region=region,
                token=token,
                args=args,
                extra_params=None,
                raw_dir=raw_dir,
                tag=f"forecast_{args.future_hours}h_{args.granularity}",
            )
            failures.extend(errors)
            if success:
                successes.append(success)
            all_rows.extend(rows)
            print(f"    forecast rows: {len(rows)}")
        print()

    save_json(out_dir / "failed_requests.json", failures)
    save_json(out_dir / "successful_requests.json", successes)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows were downloaded. Check failed_requests.json.")
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value", "region", "signal_key"])
    df = df.drop_duplicates(subset=["signal_key", "region", "datetime", "temporality"])
    df = df.sort_values(["signal_key", "region", "datetime", "temporality"])

    combined_path = out_dir / "all_aws_americas_signals_past_latest_forecast.csv"
    df.to_csv(combined_path, index=False)

    # Also save one CSV per signal.
    for signal_key, signal_df in df.groupby("signal_key"):
        signal_df.to_csv(out_dir / f"{signal_key}_past_latest_forecast.csv", index=False)

    summary = (
        df.groupby(["signal_label", "region", "temporality"])
        .agg(
            rows=("value", "count"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            mean_value=("value", "mean"),
            first_time=("datetime", "min"),
            last_time=("datetime", "max"),
            zone=("zone", lambda x: x.dropna().iloc[0] if len(x.dropna()) else None),
            endpoint_used=("endpoint_used", lambda x: ",".join(sorted(set(x.dropna())))),
            field_used=("field_used", lambda x: ",".join(sorted(set(x.dropna())))),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_by_signal_region_temporality.csv", index=False)

    print("Saved combined CSV:", combined_path)
    print("Saved summary CSV:", out_dir / "summary_by_signal_region_temporality.csv")
    print("Saved failed log:", out_dir / "failed_requests.json")
    print()
    print("Rows by signal:")
    print(df.groupby("signal_label")["value"].count())
    print()

    return df


def load_existing(out_dir: Path) -> pd.DataFrame:
    candidates = [
        out_dir / "all_aws_americas_signals_past_latest_forecast.csv",
        out_dir / "all_aws_us_signals_past_latest_forecast.csv",
        out_dir / "all_aws_us_past_latest_forecast.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna(subset=["datetime", "value", "region", "signal_key"])
    raise FileNotFoundError(
        f"No combined CSV found in {out_dir}. Run without --plot-only first."
    )


def maybe_convert_timezone(series: pd.Series, tz: Optional[str]) -> pd.Series:
    if not tz:
        return series
    return series.dt.tz_convert(tz)


def plot_all(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    if df.empty:
        print("No data to plot.")
        return

    plots_all_dir = out_dir / "plots" / "all_regions_together"
    plots_sep_dir = out_dir / "plots" / "separate_by_signal_and_region"
    plots_all_dir.mkdir(parents=True, exist_ok=True)
    plots_sep_dir.mkdir(parents=True, exist_ok=True)

    # Clean and sort.
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "value", "region", "signal_key"])
    df = df.drop_duplicates(subset=["signal_key", "region", "datetime"])
    df = df.sort_values(["signal_key", "region", "datetime"])

    now_utc = pd.Timestamp.now(tz="UTC")
    now_plot = now_utc.tz_convert(args.local_timezone) if args.local_timezone else now_utc
    x_label = f"Time ({args.local_timezone})" if args.local_timezone else "Time (UTC)"

    # Use a stable ordering.
    signal_order = [
        "carbon_intensity",
        "renewable_energy_percentage",
        "carbon_free_energy_percentage",
    ]

    for signal_key in signal_order:
        signal_df = df[df["signal_key"] == signal_key].copy()
        if signal_df.empty:
            print(f"No data found for {signal_key}; no plot created.")
            continue

        signal_label = signal_df["signal_label"].iloc[0]
        unit = signal_df["unit"].iloc[0] if "unit" in signal_df.columns else ""
        y_label = SIGNALS.get(signal_key).y_label if signal_key in SIGNALS else f"{signal_label} ({unit})"

        signal_df["plot_datetime"] = maybe_convert_timezone(signal_df["datetime"], args.local_timezone)

        # All regions together.
        plt.figure(figsize=(15, 7))
        for region in sorted(signal_df["region"].unique()):
            region_df = signal_df[signal_df["region"] == region].sort_values("plot_datetime")
            plt.plot(region_df["plot_datetime"], region_df["value"], linewidth=1.6, label=region)

        plt.axvline(now_plot, linestyle="--", linewidth=1.5, label="now")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"AWS Americas Regions: {signal_label} from Past {args.past_hours}h to Future {args.future_hours}h")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_file = plots_all_dir / f"{signal_key}_all_regions.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print("Saved:", out_file)

        # Separate region plots.
        signal_sep_dir = plots_sep_dir / signal_key
        signal_sep_dir.mkdir(parents=True, exist_ok=True)

        for region in sorted(signal_df["region"].unique()):
            region_df = signal_df[signal_df["region"] == region].sort_values("plot_datetime")

            plt.figure(figsize=(13, 5))
            plt.plot(
                region_df["plot_datetime"],
                region_df["value"],
                linewidth=1.8,
                marker=".",
                markersize=2,
                label=region,
            )
            plt.axvline(now_plot, linestyle="--", linewidth=1.5, label="now")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"AWS {region}: {signal_label} from Past {args.past_hours}h to Future {args.future_hours}h")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            out_file = signal_sep_dir / f"aws_{region}_{signal_key}_past{args.past_hours}_future{args.future_hours}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print("Saved:", out_file)

    print()
    print("All-region plots folder:", plots_all_dir)
    print("Separate plots folder:", plots_sep_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        df = load_existing(out_dir)
        plot_all(df, args, out_dir)
        return

    token = os.getenv("EMAPS_TOKEN")
    if not token:
        raise SystemExit("EMAPS_TOKEN is not set. Run: export EMAPS_TOKEN='YOUR_API_KEY'")

    df = download_all(args, token, out_dir)

    if not args.download_only:
        plot_all(df, args, out_dir)


if __name__ == "__main__":
    main()
