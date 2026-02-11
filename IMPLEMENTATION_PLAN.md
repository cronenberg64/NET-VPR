# NET-VPR: Implementation Roadmap

> **NET: Neuromorphic Event Triage for Robust Visual Place Recognition in High-Entropy Environments**
> *Optimizing Ultra-Low-Energy Localization via Spatio-Temporal Noise Filtering*

---

## Phase 1: Environment & Project Scaffolding ✅

- `requirements.txt` — numpy, h5py, matplotlib, scipy, tqdm, tonic, numba
- `.gitignore` — Python + generated data/results exclusions
- `README.md` — Research-grade with Mermaid architecture diagram, HDF5 spec, metrics table
- `src/__init__.py` — Package init with version metadata
- Directory structure: `src/`, `data/raw/`, `data/processed/`, `results/`, `scripts/`

---

## Phase 2: The "Nerve" — Data Generation ✅

### `src/generator.py` — DVSSimulator

- **Structured signal**: Clean diagonal edge sweeping across sensor (1.5px Gaussian jitter)
- **Event-Storm noise**: Poisson-distributed random spikes (`storm_rate` configurable, default 0.6)
- **Structural noise**: Fixed-pixel high-frequency flickering (LED/fluorescent artifacts)
- **HDF5 output**: `events/x`, `events/y`, `events/t`, `events/p` — gzip compressed — LENS/Event-LAB compatible
- **Visualization**: Dark-themed 2D scatter plot, ON=cyan / OFF=magenta

Run: `python -m src.generator`

---

## Phase 3: The "Cortex" — Spatio-Temporal Poisson Filter

### `src/filter.py`

- `SpatioTemporalPoissonFilter` class with configurable `radius`, `time_window_ms`, `k`
- **Core loop accelerated with `@numba.njit`**: for each event, count neighbors within `r` pixels AND `Δt` ms → keep if count ≥ `k`
- Time-sorted bucketing for cache locality
- `compute_epps(events, resolution, duration_s)` — Events Per Pixel Per Second
- `compute_snr(raw_events, filtered_events, resolution)` — Signal-to-Noise Ratio
- `compare(raw, filtered)` — prints reduction ratio, EPPS, SNR
- **Baseline wrapper**: `tonic_denoise_baseline(events, filter_time_us)` — wraps `tonic.transforms.Denoise` for comparison

---

## Phase 4: The "Synapse" — Adaptive Feedback Controller

### `src/controller.py`

- `AdaptiveEventController` class — processes stream in configurable time chunks (default 50ms)
- `compute_event_density(chunk)` — events per pixel per ms
- `adapt_filter_params(density, current_k)` — proportional controller: increase `k` on Event-Storm, relax when density drops
- `process_stream(events, filter_instance)` — chunk-by-chunk processing
- `ControlLog` dataclass: per-chunk `[timestamp, density, k_value, events_in, events_out]`

---

## Phase 5: The "Thesis" — Benchmarking & Visualization

### `scripts/benchmark.py`

- Full pipeline: generate → tonic baseline → static NET → adaptive NET
- **Metrics**: Data Reduction Ratio, EPPS, SNR, theoretical Recall@N
- **4 publication-quality figures**:
  1. Event Rate vs Time (raw, tonic-denoised, NET-static, NET-adaptive)
  2. Spatial heatmaps (raw vs. NET-filtered)
  3. Adaptive controller trace (`k` over time + density overlay)
  4. Metrics comparison bar chart (EPPS, SNR, Recall@N across methods)
- Output: `results/` as high-DPI PNGs + summary table to stdout

---

## Setup (Any OS)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
```


RAW IMPLENTATION PLAN:

NET: Neuromorphic Event Triage for Robust Visual Place Recognition
Implement a complete Python-based neuromorphic research project that simulates DVS event streams, filters noise via a Spatio-Temporal Poisson Filter (Numba-accelerated), applies an adaptive feedback loop, and produces publication-quality benchmarks with tonic.transforms.Denoise as the standard baseline.

User Review Required
IMPORTANT

Revised based on feedback: tonic re-introduced for baseline, Numba JIT for filter performance, events/ HDF5 group layout for LENS/Event-LAB compatibility, and metrics changed to EPPS + SNR + Recall@N.

Proposed Changes
Phase 1: Project Scaffolding
[MODIFY] 
README.md
Full research-grade README: title, abstract, architecture diagram (Mermaid), installation, usage, project structure, citation placeholder, Apache 2.0 badge
[NEW] 
requirements.txt
numpy, h5py, matplotlib, scipy, tqdm, tonic, numba
[NEW] 
.gitignore
Standard Python + data/, results/, *.hdf5
[NEW] 
src/__init__.py
Package init with version string
Phase 2: The "Nerve" — Data Generation
[NEW] 
src/generator.py
DVSSimulator class:
generate_structured_events(n_events) — moving edge stimulus producing spatially correlated events (x, y, t, p)
inject_event_storm(events, storm_rate) — Poisson-distributed random spike noise
inject_structural_noise(events, flicker_freq) — high-frequency flickering at fixed pixels (LED/fluorescent artifacts)
save_hdf5(events, path) — HDF5 layout: events/x, events/y, events/t, events/p (LENS / Event-LAB compatible)
visualize_events(events, duration_ms=100) — 2D scatter plot, color-coded by polarity
generate_sample_dataset() — convenience function → saves to data/raw/
Phase 3: The "Cortex" — Spatio-Temporal Poisson Filter
[NEW] 
src/filter.py
SpatioTemporalPoissonFilter class:
__init__(self, radius=3, time_window_ms=5.0, k=2) — configurable r, Δt, k
filter(events) — @numba.njit-accelerated inner loop: for each event, count neighbors within r-pixel spatial window + last Δt ms. Keep if count ≥ k. Time-sorted bucketing for cache locality.
compute_epps(events, resolution, duration_s) — Events Per Pixel Per Second
compute_snr(raw_events, filtered_events, resolution) — Signal-to-Noise Ratio (structured event energy vs. noise floor)
compare(raw, filtered) — prints reduction ratio, EPPS, SNR
Baseline wrapper: tonic_denoise_baseline(events, filter_time_us) — wraps tonic.transforms.Denoise for direct comparison against our filter
Phase 4: The "Synapse" — Adaptive Feedback Controller
[NEW] 
src/controller.py
AdaptiveEventController class:
__init__(self, chunk_duration_ms=50, target_rate=None, density_threshold=None)
compute_event_density(chunk) — events per pixel per ms
adapt_filter_params(density, current_k) — proportional controller: increase k on Event-Storm, relax when density drops
process_stream(events, filter_instance) — chunk-by-chunk processing, returns ControlLog
ControlLog dataclass: per-chunk [timestamp, density, k_value, events_in, events_out]
Phase 5: The "Thesis" — Benchmarking & Visualization
[NEW] 
scripts/benchmark.py
End-to-end benchmark:
Load or generate raw event stream
Apply tonic.transforms.Denoise (standard baseline)
Apply static NET filter (fixed k)
Apply adaptive NET filter (dynamic k)
Compute metrics: Data Reduction Ratio, EPPS, SNR, theoretical Recall@N
Generate 4 publication-quality Matplotlib figures:
Fig 1: Event Rate vs Time — raw, tonic-denoised, NET-static, NET-adaptive (shows noise peak clipping)
Fig 2: Spatial heatmaps — raw vs. NET-filtered
Fig 3: Adaptive controller trace — k over time with density overlay
Fig 4: Metrics comparison bar chart — EPPS, SNR, Recall@N across all methods
Save to results/ as high-DPI PNGs + print summary table
Directory Structure (Final)
NET-VPR/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── generator.py      # Phase 2 — DVS simulator
│   ├── filter.py          # Phase 3 — Numba-JIT Poisson filter + tonic baseline
│   └── controller.py      # Phase 4 — Adaptive feedback loop
├── scripts/
│   └── benchmark.py       # Phase 5 — Benchmarking & figures
├── data/
│   ├── raw/               # Generated HDF5 (events/x, events/y, events/t, events/p)
│   └── processed/
└── results/               # Benchmark figures & logs
Verification Plan
Automated Tests
# 1. Install dependencies
pip install -r requirements.txt
# 2. Run the full pipeline
python scripts/benchmark.py
Success criteria:

Script completes without errors
data/raw/sample_events.hdf5 exists with events/x, events/y, events/t, events/p datasets
results/ contains 4 PNG figures
Console shows: reduction ratio, EPPS, SNR, Recall@N, per-chunk adaptation log
Tonic baseline comparison included in all figures
Event Rate vs Time graph shows NET clipping noise peaks better than tonic baseline
Manual Verification
Inspect PNG figures for publication quality
Verify HDF5 structure with h5py matches LENS/Event-LAB format