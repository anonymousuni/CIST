"""
Spatiotemporal Carbon-Aware Video Encoding Scheduler
=====================================================

Algorithm (per segment s):
  1. Execution time ĥₛ from encoding data.csv column duration [s] (VEED traces; + optional safety margin)
  2. Option A — best temporal window in LOCAL region
                (lowest-CI 5-min slot between now and deadline − ĥₛ)
  3. Option B — best spatial region right now
                (lowest current CI across all permitted regions)
  4. Compare A vs B:
       if CI_temporal ≤ CI_spatial  → TEMPORAL  (wait locally)
       elif geo_restricted          → CONSTRAINED_LOCAL  (best local slot, or now)
       else                         → SPATIAL  (route to cleanest region)

Set ``scheduling_mode`` in scheduler_config.json to one of:

  - ``"temporal"`` — local region only; defer when TEMPORAL wins vs running now locally.
  - ``"spatial"`` — at each 5-min window, route to the lowest-CI region with capacity;
    advance windows when all regions are full at the current time.
  - ``"spatiotemporal"`` (default) — compare local temporal deferral (Option A) vs
    cross-region choice (Option B); when spatial wins, use the best (region, time) pair.

Usage:
  python3 scheduler.py                        # local region: default_local_region in config
  python3 scheduler.py us-west-1              # override local region
  python3 scheduler.py us-west-1 restricted   # legacy: no cross-region SPATIAL
  python3 scheduler.py --dump-forecasts       # write full CI tables (CSV per region)
  python3 scheduler.py us-west-1 --show-forecast=12   # print first 12 rows per region
  # Outputs include scheduling_summary_*.png, savings_pct_vs_baseline_*.png,
  # savings_pct_by_batch_*.png, results CSV; temporal/spatial/spatiotemporal also
  # forecast_ci_horizon_*.png and encoding_duration_*.txt

Data (defaults relative to this repo):
  video_energy_time/data.csv  — encoding runs (duration, energy, …; VEED traces)
  CI_database/.../forecast/aws_<region>_forecast_72h_5_minutes_carbon-intensity.json

Tuning (edit CIST_scheduler/scheduler_config.json — no code changes):
  safety_margin, deadline_hours, n_jobs (first n rows of data.csv), default_local_region,
  all_regions, scheduling_mode (temporal | spatial | spatiotemporal),
  max_processing_rate (optional int: per region per UTC window for spatial /
  spatiotemporal; per local window for temporal; omit = unlimited).
    When set, ci_baseline / carbon_baseline_g use a FIFO local baseline: jobs fill each 5-min
    slot from t_now to deadline in order (max_per_slot per slot), not “everyone at once now.”

Optional env: HOTCARBON_VEED_CSV, HOTCARBON_FORECAST_DIR, HOTCARBON_OUTPUT_DIR, HOTCARBON_CONFIG
"""

import json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta

# Match Electricity Maps snapshot figure (carbon_intensity_all_regions).
REGION_COLORS = {
    "ca-central-1": "#1f77b4",
    "ca-west-1": "#ff7f0e",
    "sa-east-1": "#2ca02c",
    "us-east-1": "#006D77",
    "us-east-2": "#006D77",
    "us-west-1": "#d62728",
    "us-west-2": "#9467bd",
}

# Legend label: italic t with upright roman subscript (matplotlib mathtext).
LABEL_T_NOW_START = r"$t_{\mathrm{now}}$ (start)"


def _region_color(region: str, fallback: str = "#333333") -> str:
    return REGION_COLORS.get(region, fallback)


def _write_scheduling_plots(df_res, path_png, subtitle):
    """Bar charts: % of segments per final decision; % per chosen start time (UTC slot)."""
    if df_res.empty:
        return
    try:
        import math
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[plot] matplotlib not installed; skipping summary charts. "
            "pip install matplotlib"
        )
        return

    try:
        n = len(df_res)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        dec_order = ["TEMPORAL", "LOCAL_NOW", "SPATIAL", "CONSTRAINED_LOCAL"]
        vc = df_res["decision"].value_counts()
        labels = [d for d in dec_order if d in vc.index] + [d for d in vc.index if d not in dec_order]
        pct_dec = 100.0 * vc.reindex(labels).fillna(0) / n
        axes[0].bar(range(len(labels)), pct_dec.values, color="#4472c4")
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=22, ha="right")
        axes[0].set_ylabel("% of segments")
        axes[0].set_title(f"Final decision mix (n={n})\n{subtitle}")
        _ymax = float(pct_dec.max())
        if math.isnan(_ymax):
            _ymax = 100.0
        axes[0].set_ylim(0, max(105.0, _ymax * 1.1))

        ts_counts = df_res["chosen_time_utc"].value_counts()
        max_slots = 20
        if len(ts_counts) > max_slots:
            head = ts_counts.head(max_slots)
            other_n = int(ts_counts.iloc[max_slots:].sum())
            ts_plot = pd.concat([head, pd.Series({"Other (combined)": other_n})])
        else:
            ts_plot = ts_counts
        pct_ts = 100.0 * ts_plot / n
        xlabs = []
        for ix in ts_plot.index:
            s = str(ix)
            xlabs.append(s[:22] + "…" if len(s) > 24 else s)
        axes[1].bar(range(len(pct_ts)), pct_ts.values, color="#70ad47")
        axes[1].set_xticks(range(len(pct_ts)))
        axes[1].set_xticklabels(xlabs, rotation=60, ha="right", fontsize=8)
        axes[1].set_ylabel("% of segments")
        axes[1].set_title("Chosen schedule start (forecast slot, UTC)")
        _ymax2 = float(pct_ts.max())
        if math.isnan(_ymax2):
            _ymax2 = 100.0
        axes[1].set_ylim(0, max(105.0, _ymax2 * 1.1))

        fig.tight_layout()
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSummary charts saved -> {path_png}")
    except Exception as exc:
        print(f"\n[plot] Failed to write chart {path_png}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def _write_local_forecast_ci_plot(
    region, fc_df, t_start, t_end, path_png, horizon_hours, df_sched=None
):
    """Line plot: forecast CI vs UTC time for [t_start, t_end]; optional chosen-slot overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print(
            "\n[plot] matplotlib not installed; skipping forecast CI window plot. "
            "pip install matplotlib"
        )
        return

    try:
        w = fc_df[(fc_df.index >= t_start) & (fc_df.index <= t_end)]
        if w.empty:
            print(
                f"\n[plot] No forecast rows in [{t_start}, {t_end}] for {region}; "
                "skipping forecast CI plot.",
                file=sys.stderr,
            )
            return

        line_color = _region_color(region, "#c0504d")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(
            w.index,
            w["ci"].astype(float).values,
            color=line_color,
            linewidth=1.2,
            label="Forecast CI",
            zorder=1,
        )
        ax.axvline(
            t_start,
            color="#333333",
            linestyle="--",
            linewidth=1,
            alpha=0.85,
            label=LABEL_T_NOW_START,
            zorder=2,
        )

        if df_sched is not None and not df_sched.empty and "chosen_time_utc" in df_sched.columns:
            g = (
                df_sched.groupby("chosen_time_utc", sort=False)
                .agg(n_jobs=("chosen_ci", "count"), mean_chosen_ci=("chosen_ci", "mean"))
                .reset_index()
            )
            g["_ts"] = pd.to_datetime(g["chosen_time_utc"], utc=True)
            mask = (g["_ts"] >= pd.Timestamp(t_start)) & (g["_ts"] <= pd.Timestamp(t_end))
            g = g.loc[mask]
            if not g.empty:
                xs = g["_ts"].values
                ys = g["mean_chosen_ci"].astype(float).values
                nj = g["n_jobs"].astype(int).values
                cmax = int(nj.max()) if len(nj) else 1
                sizes = 22 + 130.0 * (nj / cmax) ** 0.5
                ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    c=line_color,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=5,
                    label="Chosen slots",
                )

        ax.set_xlabel("Time (UTC)", fontsize=16)
        ax.set_ylabel("Carbon intensity (gCO2/kWh)", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(loc="upper right", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        path_pdf = path_png.with_suffix(".pdf")
        fig.savefig(path_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\nForecast CI window plot saved -> {path_png}")
        print(f"Forecast CI window plot saved -> {path_pdf}")
    except Exception as exc:
        print(
            f"\n[plot] Failed to write forecast CI plot {path_png}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()


def _write_spatial_forecast_ci_plot(
    regions,
    forecasts_by_region,
    local_region,
    t_start,
    t_end,
    path_png,
    horizon_hours,
    df_sched=None,
):
    """All-region forecast CI lines; circles on each region where jobs are assigned."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.lines import Line2D
    except ImportError:
        print(
            "\n[plot] matplotlib not installed; skipping spatial forecast CI plot. "
            "pip install matplotlib"
        )
        return

    try:
        frames = []
        for r in regions:
            fc = forecasts_by_region[r]
            w = fc[(fc.index >= t_start) & (fc.index <= t_end)][["ci"]].copy()
            if w.empty:
                continue
            w = w.rename(columns={"ci": r})
            frames.append(w)
        if not frames:
            print(
                f"\n[plot] No forecast rows in [{t_start}, {t_end}] for spatial plot; skipping.",
                file=sys.stderr,
            )
            return

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, how="outer")
        merged = merged.sort_index()

        reg_color = {r: _region_color(r, plt.cm.tab10.colors[i % 10]) for i, r in enumerate(regions)}

        fig, ax = plt.subplots(figsize=(13, 6.2))
        for r in regions:
            if r not in merged.columns:
                continue
            is_local = r == local_region
            ax.plot(
                merged.index,
                merged[r].astype(float).values,
                color=reg_color[r],
                linewidth=2.4 if is_local else 1.15,
                alpha=1.0 if is_local else 0.72,
                linestyle="-",
                zorder=2 if is_local else 1,
                label=f"{r} (local)" if is_local else r,
            )

        ax.axvline(
            t_start,
            color="#333333",
            linestyle="--",
            linewidth=1,
            alpha=0.85,
            label=LABEL_T_NOW_START,
            zorder=3,
        )

        origin_color = reg_color.get(local_region, "#1f4e79")
        n_assign_pts = 0
        if (
            df_sched is not None
            and not df_sched.empty
            and "chosen_time_utc" in df_sched.columns
            and "chosen_region" in df_sched.columns
        ):
            g = (
                df_sched.groupby(["chosen_time_utc", "chosen_region"], sort=False)
                .agg(n_jobs=("chosen_ci", "count"), mean_chosen_ci=("chosen_ci", "mean"))
                .reset_index()
            )
            g["_ts"] = pd.to_datetime(g["chosen_time_utc"], utc=True)
            mask = (g["_ts"] >= pd.Timestamp(t_start)) & (g["_ts"] <= pd.Timestamp(t_end))
            g = g.loc[mask]
            if not g.empty:
                cmax = int(g["n_jobs"].max())
                for reg in sorted(g["chosen_region"].unique()):
                    sub = g[g["chosen_region"] == reg].copy()
                    fc = forecasts_by_region.get(reg)
                    if fc is not None:
                        ys = []
                        for ts in sub["_ts"]:
                            if ts in fc.index:
                                ys.append(float(fc.loc[ts, "ci"]))
                            else:
                                ys.append(float(sub.loc[sub["_ts"] == ts, "mean_chosen_ci"].iloc[0]))
                        sub["_y"] = ys
                    else:
                        sub["_y"] = sub["mean_chosen_ci"].astype(float)
                    xs = sub["_ts"].values
                    ys = sub["_y"].astype(float).values
                    nj = sub["n_jobs"].astype(int).values
                    sizes = 28 + 150.0 * (nj / max(cmax, 1)) ** 0.5
                    n_assign_pts += len(sub)
                    # Marker color = origin (local) region; y = CI at chosen region/time.
                    ax.scatter(
                        xs,
                        ys,
                        s=sizes,
                        facecolors=origin_color,
                        edgecolors="#1a1a1a",
                        linewidths=0.7,
                        alpha=0.92,
                        zorder=6,
                    )

        assign_handle = Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor=origin_color,
            markeredgecolor="#1a1a1a",
            markeredgewidth=0.7,
            label="Chosen slots",
        )
        handles, labels = ax.get_legend_handles_labels()
        # Legend order: regions top-to-bottom by CI at plot end, then t_now, then markers.
        x_end = merged.index[-1]
        region_rank = []
        for h, lab in zip(handles, labels):
            if lab.endswith(" (local)"):
                reg = lab[: -len(" (local)")]
            else:
                reg = lab
            if reg in merged.columns:
                y_end = float(merged.loc[x_end, reg])
                if pd.isna(y_end):
                    y_end = float(merged[reg].dropna().iloc[-1])
                region_rank.append((y_end, h, lab))
        region_rank.sort(key=lambda x: x[0], reverse=True)
        sorted_handles = [x[1] for x in region_rank]
        sorted_labels = [x[2] for x in region_rank]
        tnow_handles = [
            h for h, lab in zip(handles, labels) if lab == LABEL_T_NOW_START
        ]
        ax.legend(
            sorted_handles + tnow_handles + [assign_handle],
            sorted_labels + [LABEL_T_NOW_START] + [assign_handle.get_label()],
            loc="upper right",
            fontsize=13,
            framealpha=0.92,
        )

        ax.set_xlabel("Time (UTC)", fontsize=16)
        ax.set_ylabel("Carbon intensity (gCO2/kWh)", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        path_pdf = path_png.with_suffix(".pdf")
        fig.savefig(path_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\nSpatial forecast CI window plot saved -> {path_png}")
        print(f"Spatial forecast CI window plot saved -> {path_pdf}")
    except Exception as exc:
        print(
            f"\n[plot] Failed to write spatial forecast CI plot {path_png}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()


def _fmt_duration(seconds):
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s:.1f}s ({seconds:.1f} seconds)"
    if m:
        return f"{m}m {s:.1f}s ({seconds:.1f} seconds)"
    return f"{s:.1f} seconds"


def _ascii_table(headers, rows):
    """Plain-text table with box borders (headers + list of row tuples)."""
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    widths = [max(w, 3) for w in widths]

    def _fmt_row(cells):
        return (
            "| "
            + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers)))
            + " |"
        )

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [sep, _fmt_row([str(h) for h in headers]), sep]
    for row in str_rows:
        out.append(_fmt_row(row))
    out.append(sep)
    return "\n".join(out)


def _ascii_kv_table(pairs):
    """Two-column key | value table."""
    if not pairs:
        return ""
    pairs = [(str(k), str(v)) for k, v in pairs]
    keys = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    w_key = max(len("Metric"), max(len(k) for k in keys))
    w_val = max(len("Value"), max(len(v) for v in vals))

    def _row(k, v):
        return f"| {k.ljust(w_key)} | {v.ljust(w_val)} |"

    sep = f"+-{'-' * w_key}-+-{'-' * w_val}-+"
    lines = [sep, _row("Metric", "Value"), sep]
    for k, v in pairs:
        lines.append(_row(k, v))
    lines.append(sep)
    return "\n".join(lines)


def _baseline_encoding_metrics(exec_times_s, max_per_slot, t_start):
    """
    Immediate local FIFO baseline: batches of up to max_per_slot segments encode in
    parallel (batch duration = max exec_time); batches run back-to-back from t_start
    in arrival order with no temporal shift or spatial relocation.
    Returns (active_encode_s, n_batches, finish_ts).
    """
    times = [float(x) for x in exec_times_s]
    if not times:
        return 0.0, 0, None
    batch_size = int(max_per_slot) if max_per_slot else len(times)
    batch_size = max(batch_size, 1)
    batch_durations = [
        max(times[i : i + batch_size]) for i in range(0, len(times), batch_size)
    ]
    active_s = float(sum(batch_durations))
    t0 = pd.Timestamp(t_start)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    finish = t0 + pd.Timedelta(seconds=active_s)
    return active_s, len(batch_durations), finish


def _fmt_co2_g(carbon_g):
    """Format grams CO2 for report tables."""
    carbon_g = float(carbon_g)
    return f"{carbon_g * 1000:.3f} mgCO2  ({carbon_g:.6f} gCO2)"


def _co2_reduction_pct_vs(reference_g, actual_g):
    """Percent less CO2 than reference (positive = actual is lower)."""
    reference_g = float(reference_g)
    actual_g = float(actual_g)
    if reference_g <= 0:
        return 0.0
    return (1.0 - actual_g / reference_g) * 100.0


def _encoding_vs_baseline_label(scheduled_s, baseline_s, *, metric="active encoding"):
    """Human-readable faster/slower vs baseline encoding time."""
    scheduled_s = float(scheduled_s)
    baseline_s = float(baseline_s)
    if baseline_s <= 0:
        return "n/a"
    delta = scheduled_s - baseline_s
    pct = abs(delta) / baseline_s * 100.0
    if delta < -1e-9:
        return f"{_fmt_duration(-delta)} faster ({pct:.1f}% less {metric})"
    if delta > 1e-9:
        return f"{_fmt_duration(delta)} slower (+{pct:.1f}% more {metric})"
    return "same as baseline (0% difference)"


def _write_encoding_duration_report(
    df_res,
    t_start,
    path_txt,
    n_jobs,
    max_per_slot,
    assign_elapsed_s,
    mode,
    local_region=None,
):
    """Text summary: encoding wall-clock and assignment time."""
    if df_res.empty:
        return
    if mode not in ("temporal", "spatial", "spatiotemporal"):
        raise ValueError(
            f"mode must be 'temporal', 'spatial', or 'spatiotemporal', got {mode!r}"
        )
    try:
        d = df_res.copy()
        d["_start"] = pd.to_datetime(d["chosen_time_utc"], utc=True)
        d["_exec_s"] = d["exec_time_s"].astype(float)
        d["_finish"] = d["_start"] + pd.to_timedelta(d["_exec_s"], unit="s")

        t0 = pd.Timestamp(t_start)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        else:
            t0 = t0.tz_convert("UTC")

        if mode in ("spatial", "spatiotemporal"):
            batch_groups = d.groupby(["chosen_time_utc", "chosen_region"], sort=False)
            batch_label = (
                "Batches (parallel region × forecast window, up to max_per_slot each)"
            )
        else:
            batch_groups = d.groupby("chosen_time_utc", sort=False)
            batch_label = (
                f"Batches (local region {local_region}, one per forecast window, "
                "up to max_per_slot each)"
            )

        n_parallel_batches = batch_groups.ngroups
        n_forecast_windows = int(d["_start"].nunique())
        if max_per_slot is not None:
            batch_sizes = batch_groups.size()
            n_full_batches = int((batch_sizes >= max_per_slot).sum())
            n_partial_batches = n_parallel_batches - n_full_batches
        else:
            n_full_batches = None
            n_partial_batches = None

        first_start = d["_start"].min()
        last_start = d["_start"].max()
        last_batch = d[d["_start"] == last_start]
        max_exec_last = float(last_batch["_exec_s"].max())
        finish_last_batch = last_start + pd.Timedelta(seconds=max_exec_last)

        finish_all = d["_finish"].max()
        sum_exec_all = float(d["_exec_s"].sum())

        per_window_max = d.groupby("_start", sort=True)["_exec_s"].max()
        parallel_batch_encode_s = float(per_window_max.sum())

        # Batches run back-to-back (next starts when previous batch finishes).
        cursor = None
        for win_start, max_ex in per_window_max.items():
            win_start = pd.Timestamp(win_start)
            max_ex = float(max_ex)
            if cursor is None:
                cursor = win_start
            cursor = cursor + pd.Timedelta(seconds=max_ex)
        back_to_back_finish = cursor
        back_to_back_encode_s = (
            (back_to_back_finish - first_start).total_seconds()
            if cursor is not None
            else 0.0
        )

        calendar_first_to_last_s = (finish_all - first_start).total_seconds()
        calendar_t_now_to_finish_s = (finish_all - t0).total_seconds()
        wait_before_first_s = (first_start - t0).total_seconds()

        baseline_active_s, baseline_n_batches, baseline_finish = (
            _baseline_encoding_metrics(d["_exec_s"], max_per_slot, t0)
        )
        baseline_calendar_s = (
            (baseline_finish - t0).total_seconds()
            if baseline_finish is not None
            else 0.0
        )
        baseline_region = local_region or "local"

        per_seg_ms = (
            1000.0 * float(assign_elapsed_s) / len(d)
            if assign_elapsed_s is not None and len(d)
            else 0.0
        )

        title = {
            "spatial": "Spatial-only encoding duration",
            "temporal": "Temporal-only encoding duration",
            "spatiotemporal": "Spatiotemporal encoding duration",
        }[mode]

        run_rows = [
            ("Segments scheduled (n_jobs cap)", n_jobs),
            ("Segments in this run", len(d)),
            ("Scheduling start (t_now)", t0),
            (
                "max_processing_rate",
                max_per_slot if max_per_slot is not None else "unlimited",
            ),
            ("Mode", mode),
        ]
        if mode == "temporal":
            run_rows.append(("Local region", local_region))

        carbon_rows = []
        co2_comparison_rows = []
        co2_vs_strategy_rows = []
        if "carbon_g" in d.columns and "carbon_baseline_g" in d.columns:
            total_carbon = float(d["carbon_g"].sum())
            total_baseline = float(d["carbon_baseline_g"].sum())
            saving_pct = _co2_reduction_pct_vs(total_baseline, total_carbon)
            if mode == "spatiotemporal" and {
                "carbon_temporal_only_g",
                "carbon_spatial_only_g",
            }.issubset(d.columns):
                total_temporal = float(d["carbon_temporal_only_g"].sum())
                total_spatial = float(d["carbon_spatial_only_g"].sum())
                co2_comparison_rows = [
                    (
                        "Baseline (immediate FIFO local)",
                        _fmt_co2_g(total_baseline),
                        "0.0% (reference)",
                    ),
                    (
                        "Temporal-only counterfactual",
                        _fmt_co2_g(total_temporal),
                        f"{_co2_reduction_pct_vs(total_baseline, total_temporal):+.1f}%",
                    ),
                    (
                        "Spatial-only counterfactual",
                        _fmt_co2_g(total_spatial),
                        f"{_co2_reduction_pct_vs(total_baseline, total_spatial):+.1f}%",
                    ),
                    (
                        "Spatiotemporal (scheduled)",
                        _fmt_co2_g(total_carbon),
                        f"{saving_pct:+.1f}%",
                    ),
                ]
                co2_vs_strategy_rows = [
                    (
                        "vs baseline",
                        f"{saving_pct:+.1f}% CO2 reduction "
                        f"({_fmt_co2_g(total_baseline - total_carbon)} saved)",
                    ),
                    (
                        "vs temporal-only",
                        f"{_co2_reduction_pct_vs(total_temporal, total_carbon):+.1f}% CO2 reduction "
                        f"({_fmt_co2_g(total_temporal - total_carbon)} saved)",
                    ),
                    (
                        "vs spatial-only",
                        f"{_co2_reduction_pct_vs(total_spatial, total_carbon):+.1f}% CO2 reduction "
                        f"({_fmt_co2_g(total_spatial - total_carbon)} saved)",
                    ),
                ]
            else:
                carbon_rows = [
                    (
                        "Total CO2 (scheduled)",
                        _fmt_co2_g(total_carbon),
                    ),
                    (
                        "Total CO2 (baseline)",
                        _fmt_co2_g(total_baseline),
                    ),
                    ("Net vs baseline", f"{saving_pct:+.1f}%"),
                ]

        batch_rows = [
            ("Forecast windows used", n_forecast_windows),
            ("Parallel batches (total)", n_parallel_batches),
        ]
        if max_per_slot is not None:
            batch_rows.extend(
                [
                    (f"Full batches ({max_per_slot} segments)", n_full_batches),
                    (f"Partial batches (<{max_per_slot})", n_partial_batches),
                ]
            )

        assign_rows = [
            ("Elapsed", _fmt_duration(assign_elapsed_s)),
            ("Per segment", f"{per_seg_ms:.3f} ms"),
        ]

        encode_rows = [
            (
                "Scheduled active encoding",
                f"{_fmt_duration(parallel_batch_encode_s)}  (sum of slowest per batch)",
            ),
            (
                "Scheduled back-to-back span",
                f"{_fmt_duration(back_to_back_encode_s)}  (no idle between batches)",
            ),
            ("First batch starts", first_start),
            ("Last segment finishes", finish_all),
            (
                "Single-encoder serial (reference only)",
                f"{_fmt_duration(sum_exec_all)}  (sum of all exec_time_s, no batching)",
            ),
        ]

        baseline_rows = [
            (
                "Policy",
                "Immediate FIFO in home region; sequential batches; no deferral or "
                "spatial move",
            ),
            ("Home region", baseline_region),
            ("Batches (up to max_per_slot)", baseline_n_batches),
            (
                "Active encoding time",
                f"{_fmt_duration(baseline_active_s)}  (sum of slowest per batch)",
            ),
            ("First batch starts", t0),
            ("Last segment finishes", baseline_finish),
            (
                "t_now → last finish",
                f"{_fmt_duration(baseline_calendar_s)}  (back-to-back batches)",
            ),
        ]

        vs_baseline_rows = [
            (
                "Scheduled active encoding",
                _fmt_duration(parallel_batch_encode_s),
            ),
            (
                "Baseline active encoding",
                _fmt_duration(baseline_active_s),
            ),
            (
                "Active encoding vs baseline",
                _encoding_vs_baseline_label(
                    parallel_batch_encode_s, baseline_active_s
                ),
            ),
            (
                "Scheduled t_now → last finish",
                _fmt_duration(calendar_t_now_to_finish_s),
            ),
            (
                "Baseline t_now → last finish",
                _fmt_duration(baseline_calendar_s),
            ),
            (
                "Calendar vs baseline",
                _encoding_vs_baseline_label(
                    calendar_t_now_to_finish_s,
                    baseline_calendar_s,
                    metric="wall-clock time",
                ),
            ),
        ]

        calendar_rows = [
            ("Wait before first batch", _fmt_duration(wait_before_first_s)),
            (
                "First batch start → last finish",
                _fmt_duration(calendar_first_to_last_s),
            ),
            (
                "t_now → last finish",
                f"{_fmt_duration(calendar_t_now_to_finish_s)}  (includes deferral wait)",
            ),
        ]

        last_win_rows = [
            ("Window start", last_start),
            ("Segments in window", len(last_batch)),
            ("Longest exec_time_s", f"{max_exec_last:.1f}"),
            ("Last batch finishes at", finish_last_batch),
        ]
        if mode in ("spatial", "spatiotemporal"):
            last_win_rows.insert(
                2,
                (
                    "Regions used",
                    ", ".join(sorted(last_batch["chosen_region"].unique())),
                ),
            )
        else:
            last_win_rows.insert(2, ("Region", local_region))

        window_table_rows = []
        for win_start, grp in d.groupby("_start", sort=True):
            max_ex = float(grp["_exec_s"].max())
            fin = win_start + pd.Timedelta(seconds=max_ex)
            if mode in ("spatial", "spatiotemporal"):
                n_b_in_win = grp.groupby("chosen_region", sort=False).ngroups
                regs = ", ".join(
                    f"{r}({int(c)})"
                    for r, c in grp["chosen_region"].value_counts().items()
                )
                window_table_rows.append(
                    (
                        str(win_start),
                        str(len(grp)),
                        str(n_b_in_win),
                        f"{max_ex:.1f}",
                        str(fin),
                        regs,
                    )
                )
            else:
                window_table_rows.append(
                    (
                        str(win_start),
                        str(len(grp)),
                        f"{max_ex:.1f}",
                        str(fin),
                    )
                )

        if mode in ("spatial", "spatiotemporal"):
            win_headers = [
                "Window start (UTC)",
                "Segments",
                "Batches",
                "Max exec (s)",
                "Batch ends (UTC)",
                "Regions (count)",
            ]
        else:
            win_headers = [
                "Window start (UTC)",
                "Segments",
                "Max exec (s)",
                "Batch ends (UTC)",
            ]

        sections = [
            title,
            "=" * 72,
            "",
            "Run configuration",
            _ascii_kv_table(run_rows),
            "",
            (
                "Carbon emissions comparison (sum over all segments)"
                if co2_comparison_rows
                else "Carbon emissions (sum over all segments)"
            ),
            (
                _ascii_table(
                    ["Policy", "Total CO2", "Reduction vs baseline"],
                    co2_comparison_rows,
                )
                if co2_comparison_rows
                else (
                    _ascii_kv_table(carbon_rows)
                    if carbon_rows
                    else "(carbon columns not in results)"
                )
            ),
            "",
        ]
        if co2_vs_strategy_rows:
            sections.extend(
                [
                    "Spatiotemporal CO2 reduction vs counterfactuals",
                    "Counterfactuals replay temporal-only and spatial-only policies with "
                    "separate slot state in job arrival order (same workload and caps).",
                    _ascii_table(
                        ["Comparison", "Result"],
                        co2_vs_strategy_rows,
                    ),
                    "",
                ]
            )
        sections.extend(
            [
            batch_label,
            _ascii_kv_table(batch_rows),
            "",
            "Assignment algorithm time (scheduling loop only)",
            "(excludes loading forecasts/CSV, plotting, and writing outputs)",
            _ascii_kv_table(assign_rows),
            "",
            "Encoding time — scheduled run (parallel batches)",
            "Segments in the same 5-min window encode in parallel; batch duration =",
            "max(exec_time_s) in that window.",
            _ascii_kv_table(encode_rows),
            "",
            "Baseline encoding (immediate local FIFO)",
            "Batches processed sequentially from t_now in job arrival order; up to "
            "max_processing_rate segments may run in parallel per batch in the "
            "home region.",
            _ascii_kv_table(baseline_rows),
            "",
            "Scheduled vs baseline",
            _ascii_kv_table(vs_baseline_rows),
            "",
            "Calendar span — scheduled run (includes deferred forecast slots)",
            _ascii_kv_table(calendar_rows),
            "",
            "Last forecast window",
            _ascii_kv_table(last_win_rows),
            "",
            "Per-window breakdown (chronological)",
            _ascii_table(win_headers, window_table_rows),
            "",
            ]
        )
        lines = sections

        path_txt.parent.mkdir(parents=True, exist_ok=True)
        path_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nEncoding duration report saved -> {path_txt}")
        print(
            f"  Parallel batches: {n_parallel_batches}  |  "
            f"Assignment time: {_fmt_duration(assign_elapsed_s)}"
        )
        print(
            f"  Encoding (scheduled): {_fmt_duration(parallel_batch_encode_s)}  |  "
            f"Baseline (immediate FIFO): {_fmt_duration(baseline_active_s)}  |  "
            f"{_encoding_vs_baseline_label(parallel_batch_encode_s, baseline_active_s)}"
        )
        if co2_vs_strategy_rows and "carbon_g" in d.columns:
            total_st = float(d["carbon_g"].sum())
            total_bl = float(d["carbon_baseline_g"].sum())
            total_temp = float(d["carbon_temporal_only_g"].sum())
            total_spat = float(d["carbon_spatial_only_g"].sum())
            print(
                f"  CO2 (scheduled): {_fmt_co2_g(total_st)}  |  "
                f"vs baseline {_co2_reduction_pct_vs(total_bl, total_st):+.1f}%  |  "
                f"vs temporal-only {_co2_reduction_pct_vs(total_temp, total_st):+.1f}%  |  "
                f"vs spatial-only {_co2_reduction_pct_vs(total_spat, total_st):+.1f}%"
            )
    except Exception as exc:
        print(
            f"\n[report] Failed to write encoding duration {path_txt}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()


def _write_savings_pct_plot(df_res, path_png, subtitle, fifo_baseline=False):
    """Histogram of per-segment carbon savings (%) vs baseline."""
    if df_res.empty or "saving_pct" not in df_res.columns:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[plot] matplotlib not installed; skipping savings vs baseline plot. "
            "pip install matplotlib"
        )
        return

    try:
        s = df_res["saving_pct"].astype(float)
        n = len(s)
        nbins = max(10, min(50, int(max(1, n**0.5))))
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.hist(
            s,
            bins=nbins,
            color="#4472c4",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        _mn = float(s.mean())
        _md = float(s.median())
        ax.axvline(_mn, color="#c0504d", linestyle="--", linewidth=1.2, label=f"Mean {_mn:.1f}%")
        ax.axvline(_md, color="#70ad47", linestyle=":", linewidth=1.2, label=f"Median {_md:.1f}%")
        ax.axvline(0.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.45, label="Zero savings")
        bl_note = (
            "Baseline = FIFO local forecast slots (see max_processing_rate)"
            if fifo_baseline
            else "Baseline = local CI at t_now for all segments"
        )
        ax.set_xlabel("Carbon savings vs baseline (%)")
        ax.set_ylabel("Number of segments")
        ax.set_title(
            f"Per-segment savings vs baseline (n={n})\n{subtitle}\n({bl_note})"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSavings vs baseline plot saved -> {path_png}")
    except Exception as exc:
        print(
            f"\n[plot] Failed to write savings plot {path_png}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()


def _write_savings_by_batch_line_plot(
    df_res, path_png, subtitle, batch_size, batch_from_config
):
    """Line plot: mean saving_pct vs batch index (jobs in CSV order, fixed batch size)."""
    if df_res.empty or "saving_pct" not in df_res.columns:
        return
    if batch_size < 1:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[plot] matplotlib not installed; skipping savings-by-batch plot. "
            "pip install matplotlib"
        )
        return

    try:
        d = df_res.reset_index(drop=True)
        d["_batch"] = d.index // int(batch_size)
        g = d.groupby("_batch", sort=True)["saving_pct"].mean()
        x = (g.index.astype(int) + 1).to_numpy()
        y = g.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(
            x,
            y,
            color="#4472c4",
            marker="o",
            markersize=4,
            linewidth=1.45,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        ax.set_xlabel(f"Batch index (CSV order, {batch_size} segments per batch)")
        ax.set_ylabel("Mean carbon savings vs baseline (%)")
        bs_note = (
            f"Batches align with max_processing_rate={batch_size}"
            if batch_from_config
            else (
                f"Batches use size {batch_size} (set max_processing_rate "
                "in config to match your slot cap)"
            )
        )
        ax.set_title(
            f"Mean savings vs baseline by batch (n={len(df_res)} segments)\n"
            f"{subtitle}\n({bs_note})"
        )
        ax.grid(True, alpha=0.35)
        ax.set_xticks(x[:: max(1, len(x) // 25)] if len(x) > 25 else x)
        fig.tight_layout()
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSavings by batch plot saved -> {path_png}")
    except Exception as exc:
        print(
            f"\n[plot] Failed to write savings-by-batch plot {path_png}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()


def _parse_cli(argv):
    """Split positional [region, restricted] from optional flags."""
    positional = []
    dump_forecasts = False
    show_forecast_n = 0
    for a in argv[1:]:
        if a == "--dump-forecasts":
            dump_forecasts = True
        elif a.startswith("--show-forecast="):
            try:
                show_forecast_n = max(0, int(a.split("=", 1)[1].strip() or "0"))
            except ValueError:
                print(f"Invalid integer in {a}", file=sys.stderr)
                sys.exit(1)
        elif a.startswith("--") and a != "--":
            print(f"Unknown flag: {a}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(a)
    return positional, dump_forecasts, show_forecast_n


_cli_positional, DUMP_FORECASTS, SHOW_FORECAST_N = _parse_cli(sys.argv)

_SCHEDULER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCHEDULER_DIR.parent

_DEFAULT_SCHEDULER_CFG = {
    "safety_margin": 1.1,
    "deadline_hours": 2.0,
    "n_jobs": 50,
    "default_local_region": "ca-west-1",
    "all_regions": [
        "ca-central-1",
        "ca-west-1",
        "sa-east-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ],
    "scheduling_mode": "spatiotemporal",
    "max_processing_rate": None,
}

_VALID_SCHEDULING_MODES = ("temporal", "spatial", "spatiotemporal")


def _load_json_config(path: Path) -> dict:
    """Parse JSON config; allows // line comments and /* block */ comments."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return json.loads(text)


def _load_scheduler_config():
    path = Path(os.environ.get("HOTCARBON_CONFIG", _SCHEDULER_DIR / "scheduler_config.json"))
    cfg = dict(_DEFAULT_SCHEDULER_CFG)
    if not path.is_file():
        return cfg, path
    user = _load_json_config(path)
    if not isinstance(user, dict):
        sys.exit(f"Config must be a JSON object: {path}")
    cfg.update(user)
    # Optional cap: null/missing = unlimited. Legacy key max_segments_per_time_slot still read.
    mpr = cfg.get("max_processing_rate")
    if mpr is None and "max_segments_per_time_slot" in cfg:
        mpr = cfg.get("max_segments_per_time_slot")
    if mpr is not None:
        try:
            cfg["max_processing_rate"] = int(mpr)
        except (TypeError, ValueError):
            sys.exit(
                f"max_processing_rate must be int or null in {path}: {mpr!r}"
            )
        if cfg["max_processing_rate"] < 1:
            sys.exit(f"max_processing_rate must be >= 1 in {path}")
    else:
        cfg["max_processing_rate"] = None
    # Normalize all_regions: must be non-empty list of strings
    ar = cfg.get("all_regions")
    if ar is None or not isinstance(ar, list) or not ar:
        cfg["all_regions"] = list(_DEFAULT_SCHEDULER_CFG["all_regions"])
    else:
        cfg["all_regions"] = [str(x) for x in ar]
    return cfg, path


def _resolve_scheduling_mode(cfg: dict) -> str:
    """Read scheduling_mode from config (temporal | spatial | spatiotemporal)."""
    raw = cfg.get("scheduling_mode", "spatiotemporal")
    mode = str(raw).strip().lower()
    if mode not in _VALID_SCHEDULING_MODES:
        sys.exit(
            f"scheduling_mode must be one of {_VALID_SCHEDULING_MODES}, got {raw!r}"
        )
    return mode


_CFG, CONFIG_PATH = _load_scheduler_config()

# ── Paths (repo root) ───────────────────────────────────────────────────────
_DEFAULT_FORECAST_DIR = (
    _REPO_ROOT
    / "CI_database"
    / "electricitymaps_aws_americas_72h_past_now_future_all_signals"
    / "raw_json"
    / "carbon_intensity"
    / "forecast"
)
CSV_PATH = Path(
    os.environ.get(
        "HOTCARBON_VEED_CSV",
        _REPO_ROOT / "video_energy_time" / "data.csv",
    )
)
FORECAST_DIR = Path(os.environ.get("HOTCARBON_FORECAST_DIR", _DEFAULT_FORECAST_DIR))
OUTPUT_DIR = Path(os.environ.get("HOTCARBON_OUTPUT_DIR", _REPO_ROOT / "CIST_scheduler" / "outputs"))

try:
    SAFETY_MARGIN = float(_CFG["safety_margin"])
    DEADLINE_HOURS = float(_CFG["deadline_hours"])
    N_JOBS = int(_CFG["n_jobs"])
except (TypeError, ValueError) as exc:
    sys.exit(f"Invalid numeric value in {CONFIG_PATH}: {exc}")

ALL_REGIONS = list(_CFG["all_regions"])
LOCAL_REGION = (
    _cli_positional[0]
    if len(_cli_positional) > 0
    else str(_CFG["default_local_region"])
)
GEO_RESTRICTED = len(_cli_positional) > 1 and _cli_positional[1] == "restricted"
SCHEDULING_MODE = _resolve_scheduling_mode(_CFG)
TEMPORAL_ONLY = SCHEDULING_MODE == "temporal"
SPATIAL_ONLY = SCHEDULING_MODE == "spatial"
SPATIOTEMPORAL = SCHEDULING_MODE == "spatiotemporal"
MAX_PROCESSING_RATE = _CFG.get("max_processing_rate")

if SPATIAL_ONLY and GEO_RESTRICTED:
    sys.exit(
        "spatial scheduling requires cross-region routing; do not use CLI 'restricted'."
    )

if SAFETY_MARGIN <= 0:
    sys.exit(f"safety_margin must be > 0 in {CONFIG_PATH}")
if DEADLINE_HOURS <= 0:
    sys.exit(f"deadline_hours must be > 0 in {CONFIG_PATH}")
if N_JOBS < 1:
    sys.exit(f"n_jobs must be >= 1 in {CONFIG_PATH}")
if LOCAL_REGION not in ALL_REGIONS:
    sys.exit(
        f"Local region {LOCAL_REGION!r} must be listed in all_regions ({CONFIG_PATH})"
    )

if not CSV_PATH.is_file():
    sys.exit(f"Missing encoding CSV: {CSV_PATH}")
if not FORECAST_DIR.is_dir():
    sys.exit(f"Missing forecast directory: {FORECAST_DIR}")

# ── LOAD FORECASTS ────────────────────────────────────────────────────────────
def load_forecast(region):
    path = FORECAST_DIR / f"aws_{region}_forecast_72h_5_minutes_carbon-intensity.json"
    if not path.is_file():
        sys.exit(f"Missing forecast file: {path}")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = [{"datetime": pd.to_datetime(e["datetime"], utc=True),
             "ci": float(e["carbonIntensity"])} for e in d["forecast"]]
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def _build_forecast_grid(forecasts_by_region, regions):
    """Single time × region matrix of CI for fast vectorized search."""
    parts = [
        forecasts_by_region[r][["ci"]].rename(columns={"ci": r}) for r in regions
    ]
    return pd.concat(parts, axis=1).sort_index()


_SCHEDULE_PROGRESS_EVERY = 250

print("=" * 65)
print(f"  Spatiotemporal Carbon Scheduler")
print(f"  Config file     : {CONFIG_PATH}"
      f"{'' if CONFIG_PATH.is_file() else ' (missing — using built-in defaults)'}")
print(f"  safety_margin   : {SAFETY_MARGIN}")
print(f"  deadline_hours  : {DEADLINE_HOURS}")
print(f"  n_jobs (cap)    : {N_JOBS}")
print(
    f"  max_per_slot    : "
    f"{MAX_PROCESSING_RATE if MAX_PROCESSING_RATE is not None else 'unlimited'}"
)
print(f"  scheduling_mode : {SCHEDULING_MODE}")
print(f"  Local region  : {LOCAL_REGION}")
print(f"  Geo-restricted: {GEO_RESTRICTED}")
print(f"  Encoding CSV  : {CSV_PATH}")
print(f"  Forecast dir  : {FORECAST_DIR}")
print(f"  Output dir    : {OUTPUT_DIR}")
print("=" * 65)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n[1] Loading carbon intensity forecasts...")
forecasts = {r: load_forecast(r) for r in ALL_REGIONS}
forecast_grid = _build_forecast_grid(forecasts, ALL_REGIONS)
t_now     = max(df.index[0] for df in forecasts.values())
deadline  = t_now + timedelta(hours=DEADLINE_HOURS)
print(f"    t_now    = {t_now}")
print(f"    deadline = {deadline}")
print(f"    forecast grid: {len(forecast_grid)} times × {len(ALL_REGIONS)} regions")

if DUMP_FORECASTS:
    print("\n[1b] Writing time-indexed forecast tables (full) -> CSV ...")
    for r, df in forecasts.items():
        out_fc = OUTPUT_DIR / f"forecast_ci_{r}.csv"
        df.reset_index().rename(columns={"datetime": "datetime_utc"}).to_csv(
            out_fc, index=False
        )
        print(f"      {out_fc}  ({len(df)} rows)")

if SHOW_FORECAST_N > 0:
    print(f"\n[1c] Preview of time-indexed tables (first {SHOW_FORECAST_N} rows each):")
    for r, df in forecasts.items():
        tag = " <- LOCAL" if r == LOCAL_REGION else ""
        print(f"\n  --- {r}{tag}  (ci = gCO2/kWh) ---")
        prev = df.head(SHOW_FORECAST_N).reset_index()
        prev.columns = ["datetime_utc", "ci"]
        print(prev.to_string(index=False))

def current_ci(region):
    fc = forecasts[region]
    fut = fc[fc.index >= t_now]
    return float(fut.iloc[0]["ci"]) if not fut.empty else float("inf")

print("\n    Current CI snapshot:")
for r in ALL_REGIONS:
    marker = " <- LOCAL" if r == LOCAL_REGION else ""
    print(f"      {r:20s}  {current_ci(r):5.0f} gCO2/kWh{marker}")

# ── LOAD ENCODING RUNS ────────────────────────────────────────────────────────
print("\n[2] Loading encoding runs (duration [s] from encoding CSV)...")
df_enc = pd.read_csv(CSV_PATH)
if "duration [s]" not in df_enc.columns:
    sys.exit("CSV must include column: duration [s]")
print(f"    {len(df_enc)} rows  (+{int((SAFETY_MARGIN - 1) * 100)}% safety margin on duration for slack)")


def exec_seconds_for_scheduling(duration_s):
    """Seconds reserved on the timeline: measured duration × safety margin (min 1s)."""
    return max(float(duration_s) * SAFETY_MARGIN, 1.0)


def _slot_key(ts):
    """Stable UTC key for counting assignments per forecast / chosen start time."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()


def _spatial_slot_key(ts, region):
    """Per (UTC window, region) cap for spatial_only."""
    return f"{_slot_key(ts)}|{region}"


# ── SCHEDULING FUNCTIONS ──────────────────────────────────────────────────────
def _slot_has_capacity(ts, region, slot_counts, spatial_slot_counts, max_per_slot):
    """Return True if (time [, region]) is full under max_per_slot."""
    if max_per_slot is None:
        return False
    if spatial_slot_counts is not None:
        return (
            spatial_slot_counts.get(_spatial_slot_key(ts, region), 0) >= max_per_slot
        )
    if slot_counts is not None:
        return slot_counts.get(_slot_key(ts), 0) >= max_per_slot
    return False


def best_temporal(
    region,
    exec_s,
    slot_counts=None,
    spatial_slot_counts=None,
    max_per_slot=None,
):
    """Lowest-CI feasible 5-min slot in region before deadline.

    Uses ``spatial_slot_counts`` when set (spatiotemporal / per-region caps);
    otherwise ``slot_counts`` keyed by UTC time only (temporal_only).
    """
    latest_start = deadline - timedelta(seconds=exec_s)
    if latest_start <= t_now:
        return float("inf"), None
    sub = forecast_grid.loc[
        (forecast_grid.index >= t_now) & (forecast_grid.index <= latest_start),
        region,
    ]
    if sub.empty:
        return float("inf"), None
    times = sub.index.to_numpy()
    cis = sub.to_numpy(dtype=float).ravel()
    for k in np.argsort(cis):
        ci = float(cis[k])
        if np.isnan(ci):
            continue
        t = times[k]
        if _slot_has_capacity(
            t, region, slot_counts, spatial_slot_counts, max_per_slot
        ):
            continue
        return ci, t
    return float("inf"), None


def best_spatiotemporal(regions, exec_s, spatial_slot_counts=None, max_per_slot=None):
    """Lowest-CI (region, start time) over the full horizon (Option B + deferral)."""
    latest_start = deadline - timedelta(seconds=exec_s)
    if latest_start <= t_now:
        return float("inf"), None, None
    sub = forecast_grid.loc[
        (forecast_grid.index >= t_now) & (forecast_grid.index <= latest_start),
        list(regions),
    ]
    if sub.empty:
        return float("inf"), None, None
    times = sub.index.to_numpy()
    reg_names = list(sub.columns)
    vals = sub.to_numpy(dtype=float)
    for k in np.argsort(vals, axis=None):
        ti, ri = np.unravel_index(k, vals.shape)
        ci = float(vals[ti, ri])
        if np.isnan(ci):
            continue
        t = times[ti]
        region = reg_names[ri]
        if _slot_has_capacity(
            t, region, None, spatial_slot_counts, max_per_slot
        ):
            continue
        return ci, region, t
    return float("inf"), None, None

def best_spatial(permitted):
    """Lowest current-CI region among permitted."""
    opts = [(current_ci(r), r) for r in permitted]
    return min(opts, key=lambda x: x[0])


def _all_regions_full_at_time(t, regions, spatial_slot_counts, max_per_slot):
    for region in regions:
        if spatial_slot_counts.get(_spatial_slot_key(t, region), 0) < max_per_slot:
            return False
    return True


def best_spatial_only(regions, exec_s, spatial_slot_counts=None, max_per_slot=None):
    """spatial_only: earliest UTC window, then lowest-CI region with capacity there.

    Walks 5-min slots from t_now forward. Within each slot, only regions under
    max_per_slot are eligible. Skips to the next time only when every region is
    full at the current window.
    """
    latest_start = deadline - timedelta(seconds=exec_s)
    if latest_start <= t_now:
        return float("inf"), None, None
    fc_ref = forecasts[LOCAL_REGION]
    times = fc_ref[
        (fc_ref.index >= t_now) & (fc_ref.index <= latest_start)
    ].sort_index().index
    for t in times:
        if max_per_slot is not None and spatial_slot_counts is not None:
            if _all_regions_full_at_time(t, regions, spatial_slot_counts, max_per_slot):
                continue
        best_ci = float("inf")
        best_r = None
        for region in regions:
            if max_per_slot is not None and spatial_slot_counts is not None:
                if (
                    spatial_slot_counts.get(_spatial_slot_key(t, region), 0)
                    >= max_per_slot
                ):
                    continue
            fc = forecasts[region]
            if t not in fc.index:
                continue
            ci = float(fc.loc[t, "ci"])
            if ci < best_ci:
                best_ci, best_r = ci, region
        if best_r is not None:
            return best_ci, best_r, t
    return float("inf"), None, None


def _spatial_only_decision(chosen_region, chosen_time):
    """Label spatial_only assignments for summary charts."""
    if chosen_region != LOCAL_REGION:
        return "SPATIAL"
    if _slot_key(chosen_time) == _slot_key(t_now):
        return "LOCAL_NOW"
    return "TEMPORAL"


def _shadow_temporal_only_carbon_g(h_hat, energy, shadow_slot_counts, ci_now):
    """Counterfactual carbon if this job followed temporal-only policy."""
    ci_a, t_star = best_temporal(
        LOCAL_REGION,
        h_hat,
        slot_counts=shadow_slot_counts,
        spatial_slot_counts=None,
        max_per_slot=MAX_PROCESSING_RATE,
    )
    if t_star is not None and ci_a <= ci_now:
        chosen_ci, chosen_time = ci_a, pd.Timestamp(t_star)
    else:
        chosen_ci, chosen_time = ci_now, t_now

    if MAX_PROCESSING_RATE is not None:
        k0 = _slot_key(chosen_time)
        if shadow_slot_counts.get(k0, 0) >= MAX_PROCESSING_RATE:
            ci_alt, t_alt = best_temporal(
                LOCAL_REGION,
                h_hat,
                slot_counts=shadow_slot_counts,
                spatial_slot_counts=None,
                max_per_slot=MAX_PROCESSING_RATE,
            )
            if t_alt is not None:
                chosen_time = pd.Timestamp(t_alt)
                chosen_ci = float(forecasts[LOCAL_REGION].loc[t_alt, "ci"])

    sk = _slot_key(chosen_time)
    shadow_slot_counts[sk] = shadow_slot_counts.get(sk, 0) + 1
    return energy * chosen_ci


def _shadow_spatial_only_carbon_g(h_hat, energy, shadow_spatial_counts):
    """Counterfactual carbon if this job followed spatial-only policy."""
    ci_pick, r_pick, t_pick = best_spatial_only(
        ALL_REGIONS,
        h_hat,
        shadow_spatial_counts,
        MAX_PROCESSING_RATE,
    )
    if t_pick is None:
        chosen_ci, chosen_region = best_spatial(ALL_REGIONS)
        chosen_time = t_now
    else:
        chosen_time = pd.Timestamp(t_pick)
        chosen_region = r_pick
        chosen_ci = float(forecasts[chosen_region].loc[t_pick, "ci"])

    if MAX_PROCESSING_RATE is not None:
        k0 = _spatial_slot_key(chosen_time, chosen_region)
        if shadow_spatial_counts.get(k0, 0) >= MAX_PROCESSING_RATE:
            ci_alt, r_alt, t_alt = best_spatial_only(
                ALL_REGIONS,
                h_hat,
                shadow_spatial_counts,
                MAX_PROCESSING_RATE,
            )
            if t_alt is not None:
                chosen_time = pd.Timestamp(t_alt)
                chosen_region = r_alt
                chosen_ci = float(forecasts[chosen_region].loc[t_alt, "ci"])

    sk = _spatial_slot_key(chosen_time, chosen_region)
    shadow_spatial_counts[sk] = shadow_spatial_counts.get(sk, 0) + 1
    return energy * chosen_ci


# ── SCHEDULE ──────────────────────────────────────────────────────────────────
n_sample = min(N_JOBS, len(df_enc))
print(
    f"\n[3] Scheduling first {n_sample} rows of CSV in order "
    f"(n_jobs cap={N_JOBS}, total rows={len(df_enc)})..."
)
if SPATIOTEMPORAL and n_sample >= _SCHEDULE_PROGRESS_EVERY:
    print(
        f"    (progress every {_SCHEDULE_PROGRESS_EVERY} jobs; "
        "spatiotemporal mode may take several minutes for large n_jobs)",
        flush=True,
    )
jobs = df_enc.iloc[:n_sample].reset_index(drop=True)
results = []
slot_counts = {}
spatial_slot_counts = {}
_shadow_temporal_slots = {}
_shadow_spatial_slots = {}
_slot_cap_overflow_warned = False
_fifo_baseline_overflow_warned = False

_fc_local = forecasts[LOCAL_REGION]
_baseline_slot_times = None
if MAX_PROCESSING_RATE is not None:
    _baseline_slot_times = (
        _fc_local[
            (_fc_local.index >= t_now) & (_fc_local.index <= deadline)
        ]
        .sort_index()
        .index.to_list()
    )

ci_now_local = current_ci(LOCAL_REGION)

_assign_elapsed_s = None
_assign_t0 = time.perf_counter()

_use_spatial_caps = SCHEDULING_MODE in ("spatial", "spatiotemporal")

for job_i, (_, row) in enumerate(jobs.iterrows()):
    if job_i > 0 and job_i % _SCHEDULE_PROGRESS_EVERY == 0:
        _elapsed = time.perf_counter() - _assign_t0
        _rate = job_i / _elapsed if _elapsed > 0 else 0.0
        _eta = (n_sample - job_i) / _rate if _rate > 0 else 0.0
        print(
            f"    scheduled {job_i}/{n_sample} "
            f"({_rate:.1f} jobs/s, ~{_eta:.0f}s remaining)...",
            flush=True,
        )

    w, h, br = int(row["width"]), int(row["height"]), int(row["bitrate [kb/s]"])
    h_hat = exec_seconds_for_scheduling(row["duration [s]"])
    energy = float(row["model_cpu_energy [kWh]"])

    if SPATIOTEMPORAL:
        carbon_temporal_only = _shadow_temporal_only_carbon_g(
            h_hat, energy, _shadow_temporal_slots, ci_now_local
        )
        carbon_spatial_only = _shadow_spatial_only_carbon_g(
            h_hat, energy, _shadow_spatial_slots
        )

    _local_slots = None if _use_spatial_caps else slot_counts
    _cap_slots = spatial_slot_counts if _use_spatial_caps else None

    ci_a, t_star = best_temporal(
        LOCAL_REGION,
        h_hat,
        slot_counts=_local_slots,
        spatial_slot_counts=_cap_slots,
        max_per_slot=MAX_PROCESSING_RATE,
    )
    permitted = (
        [LOCAL_REGION] if (TEMPORAL_ONLY or GEO_RESTRICTED) else ALL_REGIONS
    )
    ci_b, r_star = best_spatial(permitted)

    if SPATIAL_ONLY:
        chosen_ci, chosen_region, t_pick = best_spatial_only(
            ALL_REGIONS,
            h_hat,
            spatial_slot_counts,
            MAX_PROCESSING_RATE,
        )
        if t_pick is None:
            chosen_ci, chosen_region, chosen_time = ci_b, r_star, t_now
        else:
            chosen_time = pd.Timestamp(t_pick)
            chosen_ci = float(forecasts[chosen_region].loc[t_pick, "ci"])
        decision = _spatial_only_decision(chosen_region, chosen_time)
    elif TEMPORAL_ONLY:
        if t_star is not None and ci_a <= ci_now_local:
            decision, chosen_ci, chosen_region, chosen_time = (
                "TEMPORAL",
                ci_a,
                LOCAL_REGION,
                t_star,
            )
        else:
            decision, chosen_ci, chosen_region, chosen_time = (
                "LOCAL_NOW",
                ci_now_local,
                LOCAL_REGION,
                t_now,
            )
    elif GEO_RESTRICTED:
        if t_star is not None:
            decision, chosen_ci, chosen_region, chosen_time = (
                "CONSTRAINED_LOCAL",
                ci_a,
                LOCAL_REGION,
                t_star,
            )
        else:
            decision, chosen_ci, chosen_region, chosen_time = (
                "CONSTRAINED_LOCAL",
                ci_now_local,
                LOCAL_REGION,
                t_now,
            )
    else:
        # spatiotemporal: compare local temporal (A) vs spatial; defer in space-time if B wins.
        if t_star is not None and ci_a <= ci_b:
            decision, chosen_ci, chosen_region, chosen_time = (
                "TEMPORAL",
                ci_a,
                LOCAL_REGION,
                t_star,
            )
        else:
            ci_g, r_g, t_g = best_spatiotemporal(
                ALL_REGIONS,
                h_hat,
                spatial_slot_counts,
                MAX_PROCESSING_RATE,
            )
            if t_g is not None and r_g is not None and ci_g < float("inf"):
                decision, chosen_ci, chosen_region, chosen_time = (
                    "SPATIAL",
                    ci_g,
                    r_g,
                    pd.Timestamp(t_g),
                )
            else:
                decision, chosen_ci, chosen_region, chosen_time = (
                    "SPATIAL",
                    ci_b,
                    r_star,
                    t_now,
                )

    if MAX_PROCESSING_RATE is not None:
        if _use_spatial_caps:
            k0 = _spatial_slot_key(chosen_time, chosen_region)
            slot_full = spatial_slot_counts.get(k0, 0) >= MAX_PROCESSING_RATE
        else:
            k0 = _slot_key(chosen_time)
            slot_full = slot_counts.get(k0, 0) >= MAX_PROCESSING_RATE
        if slot_full:
            if SPATIAL_ONLY:
                ci_alt, r_alt, t_alt = best_spatial_only(
                    ALL_REGIONS,
                    h_hat,
                    spatial_slot_counts,
                    MAX_PROCESSING_RATE,
                )
            elif SPATIOTEMPORAL:
                ci_alt, r_alt, t_alt = best_spatiotemporal(
                    ALL_REGIONS,
                    h_hat,
                    spatial_slot_counts,
                    MAX_PROCESSING_RATE,
                )
            else:
                ci_alt, t_alt = best_temporal(
                    LOCAL_REGION,
                    h_hat,
                    slot_counts=_local_slots,
                    spatial_slot_counts=_cap_slots,
                    max_per_slot=MAX_PROCESSING_RATE,
                )
                r_alt = LOCAL_REGION
            if t_alt is not None:
                chosen_time = pd.Timestamp(t_alt)
                chosen_region = r_alt
                chosen_ci = float(forecasts[chosen_region].loc[t_alt, "ci"])
                if SPATIAL_ONLY:
                    decision = _spatial_only_decision(chosen_region, chosen_time)
                elif GEO_RESTRICTED:
                    decision = "CONSTRAINED_LOCAL"
                elif chosen_region == LOCAL_REGION and _slot_key(chosen_time) != _slot_key(
                    t_now
                ):
                    decision = "TEMPORAL"
                else:
                    decision = "SPATIAL"
                if TEMPORAL_ONLY:
                    ci_a = chosen_ci
            else:
                if not _slot_cap_overflow_warned:
                    scope = (
                        "cross-region"
                        if _use_spatial_caps
                        else "local"
                    )
                    print(
                        f"\n[warn] max_processing_rate: no slot under cap "
                        f"left in {scope} window for a job that hit a full time; "
                        "assigning anyway (slot may exceed limit).",
                        file=sys.stderr,
                    )
                    _slot_cap_overflow_warned = True

    if _use_spatial_caps:
        sk_fin = _spatial_slot_key(chosen_time, chosen_region)
        spatial_slot_counts[sk_fin] = spatial_slot_counts.get(sk_fin, 0) + 1
    else:
        sk_fin = _slot_key(chosen_time)
        slot_counts[sk_fin] = slot_counts.get(sk_fin, 0) + 1

    if MAX_PROCESSING_RATE is None:
        ci_baseline_val = ci_now_local
    elif _baseline_slot_times:
        slot_rank = job_i // MAX_PROCESSING_RATE
        if slot_rank >= len(_baseline_slot_times):
            if not _fifo_baseline_overflow_warned:
                print(
                    "\n[warn] FIFO baseline: more jobs than slots×max_per_slot in "
                    f"[t_now, deadline]; extra jobs use CI at final slot {_baseline_slot_times[-1]}.",
                    file=sys.stderr,
                )
                _fifo_baseline_overflow_warned = True
            slot_rank = len(_baseline_slot_times) - 1
        t_base = _baseline_slot_times[slot_rank]
        ci_baseline_val = float(_fc_local.loc[t_base, "ci"])
    else:
        ci_baseline_val = ci_now_local

    carbon        = energy * chosen_ci
    carbon_base   = energy * ci_baseline_val

    job_result = {
        "segment":           row["input_file_name"],
        "width":             w, "height": h,
        "exec_time_s":       round(h_hat, 1),
        "decision":          decision,
        "chosen_region":     chosen_region,
        "chosen_time_utc":   str(chosen_time),
        "ci_temporal":       round(ci_a, 1),
        "ci_spatial_now":    round(ci_b, 1),
        "chosen_ci":         round(chosen_ci, 1),
        "ci_baseline":       round(ci_baseline_val, 1),
        "energy_kwh":        round(energy, 8),
        "carbon_g":          round(carbon, 6),
        "carbon_baseline_g": round(carbon_base, 6),
        "saving_pct":        round((1 - carbon/carbon_base)*100, 1) if carbon_base > 0 else 0.0,
    }
    if SPATIOTEMPORAL:
        job_result["carbon_temporal_only_g"] = round(carbon_temporal_only, 6)
        job_result["carbon_spatial_only_g"] = round(carbon_spatial_only, 6)
    results.append(job_result)

_assign_elapsed_s = time.perf_counter() - _assign_t0

df_res = pd.DataFrame(results)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESULTS")
print("="*65)

print("\nDecision breakdown:")
n_run = len(df_res)
for dec, cnt in df_res["decision"].value_counts().items():
    print(f"  {dec:25s}  {cnt:3d} jobs  ({100 * cnt / n_run:.0f}%)")

total_carbon   = df_res["carbon_g"].sum()
total_baseline = df_res["carbon_baseline_g"].sum()
print(f"\nCarbon emissions:")
if MAX_PROCESSING_RATE is None:
    _bl = "Baseline  (all jobs at local CI now)"
else:
    _bl = (
        f"Baseline  (FIFO local: {MAX_PROCESSING_RATE}/slot along forecast to deadline)"
    )
print(f"  {_bl}: {total_baseline*1000:.3f} mgCO2")
print(f"  Scheduled (this algorithm):               {total_carbon*1000:.3f} mgCO2")
print(f"  Total saving:                             {(1-total_carbon/total_baseline)*100:.1f}%")

if SPATIOTEMPORAL and {
    "carbon_temporal_only_g",
    "carbon_spatial_only_g",
}.issubset(df_res.columns):
    total_temporal_cf = df_res["carbon_temporal_only_g"].sum()
    total_spatial_cf = df_res["carbon_spatial_only_g"].sum()
    print(f"\nCO2 comparison (counterfactuals, same job order and caps):")
    print(f"  Temporal-only counterfactual:             {total_temporal_cf*1000:.3f} mgCO2")
    print(f"  Spatial-only counterfactual:              {total_spatial_cf*1000:.3f} mgCO2")
    print(
        f"  Spatiotemporal vs temporal-only:          "
        f"{_co2_reduction_pct_vs(total_temporal_cf, total_carbon):+.1f}% reduction"
    )
    print(
        f"  Spatiotemporal vs spatial-only:           "
        f"{_co2_reduction_pct_vs(total_spatial_cf, total_carbon):+.1f}% reduction"
    )

print(f"\nPer-decision carbon savings:")
for dec in ["TEMPORAL", "LOCAL_NOW", "SPATIAL", "CONSTRAINED_LOCAL"]:
    sub = df_res[df_res["decision"]==dec]
    if sub.empty: continue
    c,b = sub["carbon_g"].sum(), sub["carbon_baseline_g"].sum()
    s   = (1-c/b)*100 if b>0 else 0
    print(f"  {dec:25s}  saving={s:5.1f}%  "
          f"avg_CI_chosen={sub['chosen_ci'].mean():.0f}  "
          f"baseline_CI={sub['ci_baseline'].mean():.0f}")

spatial_jobs = df_res[df_res["decision"]=="SPATIAL"]
if not spatial_jobs.empty:
    print(f"\nSpatial fallback — regions chosen:")
    print(spatial_jobs["chosen_region"].value_counts().to_string())

print(f"\nCI summary:")
print(f"  Local ({LOCAL_REGION}) now:          {current_ci(LOCAL_REGION):.0f} gCO2/kWh")
print(f"  Avg CI this scheduler achieved:  {df_res['chosen_ci'].mean():.0f} gCO2/kWh")
best_ci, best_r = best_spatial(ALL_REGIONS)
print(f"  Best region available now:       {best_ci:.0f} gCO2/kWh ({best_r})")

# ── EXPORT ────────────────────────────────────────────────────────────────────
if TEMPORAL_ONLY:
    suffix = f"_{LOCAL_REGION}_temporal_only"
elif SPATIAL_ONLY:
    suffix = f"_{LOCAL_REGION}_spatial_only"
elif GEO_RESTRICTED:
    suffix = f"_{LOCAL_REGION}_restricted"
else:
    suffix = f"_{LOCAL_REGION}_spatiotemporal"

if TEMPORAL_ONLY:
    _sub = f"{LOCAL_REGION}, temporal_only (local time-shift only)"
elif SPATIAL_ONLY:
    _sub = (
        f"{LOCAL_REGION}, spatial_only "
        "(fill current window across regions, then next window)"
    )
elif GEO_RESTRICTED:
    _sub = f"{LOCAL_REGION}, restricted (no cross-region)"
else:
    _sub = f"{LOCAL_REGION}, spatiotemporal (local deferral vs cross-region space-time)"
_write_scheduling_plots(
    df_res,
    OUTPUT_DIR / f"scheduling_summary_pct{suffix}.png",
    _sub,
)
_write_savings_pct_plot(
    df_res,
    OUTPUT_DIR / f"savings_pct_vs_baseline{suffix}.png",
    _sub,
    fifo_baseline=MAX_PROCESSING_RATE is not None,
)
_batch_sz = MAX_PROCESSING_RATE if MAX_PROCESSING_RATE is not None else 80
_write_savings_by_batch_line_plot(
    df_res,
    OUTPUT_DIR / f"savings_pct_by_batch{suffix}.png",
    _sub,
    batch_size=_batch_sz,
    batch_from_config=MAX_PROCESSING_RATE is not None,
)
if TEMPORAL_ONLY:
    _write_local_forecast_ci_plot(
        LOCAL_REGION,
        forecasts[LOCAL_REGION],
        t_now,
        deadline,
        OUTPUT_DIR / f"forecast_ci_horizon{suffix}.png",
        DEADLINE_HOURS,
        df_sched=df_res,
    )
    _write_encoding_duration_report(
        df_res,
        t_now,
        OUTPUT_DIR / f"encoding_duration{suffix}.txt",
        n_run,
        MAX_PROCESSING_RATE,
        _assign_elapsed_s,
        mode="temporal",
        local_region=LOCAL_REGION,
    )
elif SPATIOTEMPORAL or SPATIAL_ONLY:
    _write_spatial_forecast_ci_plot(
        ALL_REGIONS,
        forecasts,
        LOCAL_REGION,
        t_now,
        deadline,
        OUTPUT_DIR / f"forecast_ci_horizon{suffix}.png",
        DEADLINE_HOURS,
        df_sched=df_res,
    )
    _write_encoding_duration_report(
        df_res,
        t_now,
        OUTPUT_DIR / f"encoding_duration{suffix}.txt",
        n_run,
        MAX_PROCESSING_RATE,
        _assign_elapsed_s,
        mode="spatiotemporal" if SPATIOTEMPORAL else "spatial",
        local_region=LOCAL_REGION,
    )
out    = OUTPUT_DIR / f"scheduling_results{suffix}.csv"
df_res.to_csv(out, index=False)
print(f"\nResults saved -> {out}")

print("\nSample (first 8 jobs):")
print(df_res[["segment","width","height","decision",
              "chosen_region","chosen_ci","ci_baseline","saving_pct"]
      ].head(8).to_string(index=False))
