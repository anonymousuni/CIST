# CIST — Carbon-aware video encoding scheduler

**CIST** is a Python experiment for **spatiotemporal carbon-aware scheduling** of video encoding jobs. It uses encoding durations and carbon-intensity forecasts for AWS regions, and simulates scheduling decisions across three modes: wait locally (temporal), route to another region (spatial), or combine both (spatiotemporal). 

This **repository** contains the scheduler code under **`CIST_scheduler/`**, the carbon-intensity (CI) dataset and download tools under **`CI_database/`**, and encoding traces under **`video_energy_time/`**. Paths below are relative to the **repository root**.

## Repository layout

| Path | Purpose |
|------|---------|
| **`CIST_scheduler/`** | Main simulation: `scheduler.py`, `scheduler_config.json`, `requirements.txt`, `outputs/`, and `result examples/`. |
| **`CI_database/`** | **CI dataset**: Electricity Maps JSON (forecasts, past-range, latest) for AWS Americas regions, plus Python scripts to download and plot signals. |
| **`video_energy_time/`** | Encoding benchmark CSV (`data.csv`). |

## CI_database — carbon intensity (CI) dataset

**`CI_database/`** holds the **carbon-intensity and related signals** used by CIST (5-minute resolution, 72 h horizon forecasts, etc.). The data come from the **Electricity Maps** API ([electricitymaps.com](https://www.electricitymaps.com/)).

After downloading, the large tree usually looks like this (names may vary slightly with API export):

- **`CI_database/electricitymaps_aws_americas_72h_past_now_future_all_signals/`**
  - **`raw_json/carbon_intensity/forecast/`** — `aws_<region>_forecast_72h_5_minutes_carbon-intensity.json` 
  - Other subfolders may include `past-range/`, `latest/`, plus optional renewable / carbon-free percentage signals

**`successful_requests.json`** (under that bundle) records which endpoints were fetched and where files were written.

### Helper scripts (run from repo root or from `CI_database/`)

| Script | Role |
|--------|------|
| **`CI_database/download_aws_americas_all_signals_past_now_future_plot.py`** | Download CI (and related) signals for default AWS Americas regions; optional plotting. Requires an API token. |
| **`CI_database/plot_aws_americas_renewable_carbonfree.py`** | Plot from an existing combined CSV under the bundle directory. |

**Authentication:** set an Electricity Maps API token, for example:

```bash
export EMAPS_TOKEN="YOUR_API_KEY"
cd CI_database
python3 download_aws_americas_all_signals_past_now_future_plot.py
```

Use **`--help`** on each script for options (output directory, plot-only, granularity, etc.). Default download output subfolder name is typically **`electricitymaps_aws_americas_72h_past_now_future_all_signals`** (created under the current working directory unless you pass **`--output-dir`**).

### Linking CIST to this dataset

CIST resolves forecast JSON paths from **`HOTCARBON_FORECAST_DIR`**; if unset, it defaults to:

`CI_database/electricitymaps_aws_americas_72h_past_now_future_all_signals/raw_json/carbon_intensity/forecast`

# CIST scheduler
## Requirements 

- Python 3.10+ (3.13 works with the listed packages)
- Dependencies: **`pandas`**, **`matplotlib`** (see **`CIST_scheduler/requirements.txt`**)

Install (recommended: virtualenv **inside** `CIST_scheduler/`):

```bash
cd CIST_scheduler
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## CIST defaults

| Input | Default path (from repo root) |
|-------|-------------------------------|
| Encoding jobs | `video_energy_time/data.csv` |
| Per-region CI forecasts (5-minute, 72 h horizon; Electricity Maps) | `CI_database/electricitymaps_aws_americas_72h_past_now_future_all_signals/raw_json/carbon_intensity/forecast/aws_<region>_forecast_72h_5_minutes_carbon-intensity.json` |

## Configuration (`CIST_scheduler/scheduler_config.json`)

Edit this file to change behavior without editing code. Important keys:

- **`scheduling_mode`**: one of `temporal`, `spatial`, `spatiotemporal`.
- **`safety_margin`**: multiplier on estimated segment duration.
- **`deadline_hours`**: horizon from “now” for scheduling.
- **`n_jobs`**: number of video encoding tasks.
- **`default_local_region`**: AWS region treated as “local” when you do not pass a region on the CLI.
- **`all_regions`**: regions considered for spatial and spatiotemporal modes.
- **`max_processing_rate`**: maximum processing load per instance per 5-minute slot; omit or set to `null` for unlimited.

## Usage (CIST)

Run from the **repository root** so default paths resolve correctly:

```bash
cd /path/to/this-repo
python3 CIST_scheduler/scheduler.py
```

Examples:

```bash
python3 CIST_scheduler/scheduler.py
python3 CIST_scheduler/scheduler.py us-west-1
python3 CIST_scheduler/scheduler.py us-west-1 --show-forecast=12
python3 CIST_scheduler/scheduler.py --dump-forecasts
```

Each run writes scheduling CSVs, encoding duration summaries, and PNG plots under *`CIST_scheduler/outputs/`*. For sample outputs, see **`CIST_scheduler/outputs/result examples/`**.

## References

Video complexity dataset:

> **VCD: Video Complexity Dataset.** In *Proceedings of the 13th ACM Multimedia Systems Conference (MMSys '22)*, 2022, pp. 234–239.

Encoding benchmark traces (execution time, energy consumption, and CO₂ on different AWS EC2 instances):

> **VEED: Video Encoding Energy and CO₂ Emissions Dataset for AWS EC2 Instances.** In *Proceedings of the 15th ACM Multimedia Systems Conference*, 2024, pp. 332–338.

Encoding duration, energy, and CO₂ prediction:

> **VEEP: Video Encoding Energy and CO₂ Emission Prediction.** In *Proceedings of the Second International ACM Green Multimedia Systems Workshop*, 2024, pp. 16–21.

Instance capacity prediction and sustainable distribution of encoding on cloud and edge:

> **X4-MATCH: Sustainable Prediction-based Distribution of Video Encoding on Cloud and Edge.** *IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2026.

Carbon-intensity forecasts: **Electricity Maps** (local copy under **`CI_database/`**).
