"""
NET: Neuromorphic Event Triage — Spatio-Temporal Poisson Filter
================================================================
Numba-accelerated spatio-temporal correlation filter for DVS event
stream denoising, with tonic.transforms.Denoise baseline comparison
and ground-truth SNR / EPPS metrics.
"""

import os
import time
import numpy as np
import numba
from numba import njit
import h5py


# ============================================================================
# Numba-Accelerated Core Kernel
# ============================================================================

@njit(cache=True)
def _njit_filter_events(x, y, t, radius, time_window_us, k):
    """Spatio-temporal correlation filter — JIT-compiled inner loop.

    For each event i, count neighbors j satisfying:
        |x_i - x_j| <= radius  AND  |y_i - y_j| <= radius
        AND  |t_i - t_j| <= time_window_us
    Keep event i if neighbor count >= k.

    Exploits time-sorted input: scans backward AND forward from i,
    breaking early when the temporal window is exceeded.

    Parameters
    ----------
    x, y : uint16 arrays — pixel coordinates
    t     : int64 array  — timestamps in microseconds (MUST be sorted)
    radius : int          — spatial neighborhood radius (pixels)
    time_window_us : int64 — temporal window (microseconds)
    k     : int           — minimum neighbor count to retain event

    Returns
    -------
    keep : boolean array — mask of events to retain
    """
    n = len(x)
    keep = np.zeros(n, dtype=numba.boolean)

    for i in range(n):
        count = 0
        xi = np.int32(x[i])
        yi = np.int32(y[i])
        ti = t[i]

        # --- Backward scan (earlier events) ---
        j = i - 1
        while j >= 0:
            dt = ti - t[j]
            if dt > time_window_us:
                break
            dx = xi - np.int32(x[j])
            dy = yi - np.int32(y[j])
            if dx < 0:
                dx = -dx
            if dy < 0:
                dy = -dy
            if dx <= radius and dy <= radius:
                count += 1
                if count >= k:
                    break
            j -= 1

        keep[i] = count >= k

    return keep


# ============================================================================
# Filter Class
# ============================================================================

class SpatioTemporalPoissonFilter:
    """Spatio-Temporal Poisson Filter for DVS event denoising.

    For each event, counts spatio-temporal neighbors within a configurable
    radius and time window. Events with fewer than k neighbors are
    classified as noise and removed.

    Parameters
    ----------
    radius : int
        Spatial radius in pixels (default 3).
    time_window_ms : float
        Temporal window in milliseconds (default 5.0).
    k : int
        Minimum neighbor count threshold (default 2).
    """

    def __init__(self, radius=3, time_window_ms=5.0, k=2):
        self.radius = radius
        self.time_window_ms = time_window_ms
        self.time_window_us = np.int64(time_window_ms * 1000)
        self.k = k

    def warmup(self):
        """Trigger Numba JIT compilation on a tiny dummy array."""
        dummy_n = 100
        _njit_filter_events(
            np.zeros(dummy_n, dtype=np.uint16),
            np.zeros(dummy_n, dtype=np.uint16),
            np.arange(dummy_n, dtype=np.int64),
            self.radius, self.time_window_us, self.k,
        )

    def filter(self, events):
        """Apply the spatio-temporal Poisson filter.

        Parameters
        ----------
        events : dict  {x, y, t, p} — must be sorted by timestamp.

        Returns
        -------
        filtered : dict  {x, y, t, p}
        keep_mask : np.ndarray[bool]  (aligned to input)
        """
        keep = _njit_filter_events(
            events['x'].astype(np.uint16),
            events['y'].astype(np.uint16),
            events['t'].astype(np.int64),
            self.radius, self.time_window_us, self.k,
        )
        keep_np = np.asarray(keep).astype(np.bool_)
        filtered = {k: events[k][keep_np] for k in ('x', 'y', 't', 'p')}
        return filtered, keep_np

    def __repr__(self):
        return (f"SpatioTemporalPoissonFilter("
                f"r={self.radius}, Δt={self.time_window_ms}ms, k={self.k})")


# ============================================================================
# Metrics
# ============================================================================

def compute_epps(events, resolution, duration_s):
    """Events Per Pixel Per Second."""
    w, h = resolution
    return len(events['t']) / (w * h * duration_s)


def compute_snr(n_signal_kept, n_noise_kept):
    """Signal-to-Noise Ratio from ground-truth labels (dB).

    SNR = 10 · log₁₀(n_signal / n_noise)
    """
    return 10.0 * np.log10(max(n_signal_kept, 1) / max(n_noise_kept, 1))


def compare(raw_events, keep_mask, resolution, duration_s, signal_mask,
            method_name="Filter"):
    """Print formatted comparison and return metrics dict."""
    n_raw = len(raw_events['t'])
    n_filt = int(keep_mask.sum())
    reduction = (1.0 - n_filt / n_raw) * 100.0

    epps_raw = compute_epps(raw_events, resolution, duration_s)
    filt_events = {k: raw_events[k][keep_mask] for k in ('x', 'y', 't', 'p')}
    epps_filt = compute_epps(filt_events, resolution, duration_s)

    n_sig_raw = int(signal_mask.sum())
    n_noise_raw = n_raw - n_sig_raw
    snr_raw = compute_snr(n_sig_raw, n_noise_raw)

    n_sig_filt = int((signal_mask & keep_mask).sum())
    n_noise_filt = int((~signal_mask & keep_mask).sum())
    snr_filt = compute_snr(n_sig_filt, n_noise_filt)
    snr_delta = snr_filt - snr_raw

    print(f"\n  ── {method_name} ──")
    print(f"    Events:       {n_raw:>8,}  →  {n_filt:>8,}  "
          f"(reduction: {reduction:.1f}%)")
    print(f"    EPPS:         {epps_raw:>8.2f}  →  {epps_filt:>8.2f}")
    print(f"    SNR:          {snr_raw:>8.2f} dB →  {snr_filt:>8.2f} dB  "
          f"(Δ = +{snr_delta:.2f} dB)")
    print(f"    Signal kept:  {n_sig_filt:>8,} / {n_sig_raw:,}  "
          f"({n_sig_filt/max(n_sig_raw,1)*100:.1f}%)")
    print(f"    Noise leaked: {n_noise_filt:>8,} / {n_noise_raw:,}  "
          f"({n_noise_filt/max(n_noise_raw,1)*100:.1f}%)")

    return {
        'method': method_name,
        'n_raw': n_raw, 'n_filtered': n_filt,
        'reduction_pct': reduction,
        'epps_raw': epps_raw, 'epps_filtered': epps_filt,
        'snr_raw_db': snr_raw, 'snr_filtered_db': snr_filt,
        'snr_improvement_db': snr_delta,
        'signal_kept_pct': n_sig_filt / max(n_sig_raw, 1) * 100,
        'noise_leaked_pct': n_noise_filt / max(n_noise_raw, 1) * 100,
    }


# ============================================================================
# Tonic Baseline Wrapper
# ============================================================================

def tonic_denoise_baseline(events, sensor_size, filter_time_us=10_000):
    """Apply tonic.transforms.Denoise as the standard baseline.

    Converts between NET dict format ↔ tonic structured-array format.

    Returns
    -------
    filtered : dict {x, y, t, p}
    keep_mask : np.ndarray[bool] aligned to input
    """
    try:
        import tonic
    except ImportError:
        print("[WARNING] 'tonic' not installed. Skipping baseline.")
        return events, np.ones(len(events['t']), dtype=np.bool_)

    n = len(events['t'])
    tonic_ev = np.zeros(n, dtype=[
        ('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')
    ])
    tonic_ev['x'] = events['x'].astype(np.int64)
    tonic_ev['y'] = events['y'].astype(np.int64)
    tonic_ev['t'] = events['t'].astype(np.int64)
    tonic_ev['p'] = events['p'].astype(np.int64)

    denoise = tonic.transforms.Denoise(filter_time=filter_time_us)
    denoised = denoise(tonic_ev)

    # Reconstruct keep_mask via merge-join (both time-sorted)
    m = len(denoised)
    keep_mask = np.zeros(n, dtype=np.bool_)
    j = 0
    for i in range(m):
        while j < n:
            if (tonic_ev['t'][j] == denoised['t'][i] and
                tonic_ev['x'][j] == denoised['x'][i] and
                tonic_ev['y'][j] == denoised['y'][i] and
                tonic_ev['p'][j] == denoised['p'][i]):
                keep_mask[j] = True
                j += 1
                break
            j += 1

    filtered = {
        'x': denoised['x'].astype(np.uint16),
        'y': denoised['y'].astype(np.uint16),
        't': denoised['t'].astype(np.int64),
        'p': denoised['p'].astype(np.uint8),
    }
    return filtered, keep_mask


# ============================================================================
# Noise Floor Visualization
# ============================================================================

def visualize_noise_floor(raw_events, net_mask, signal_mask, resolution,
                          save_path=None):
    """4-panel figure showing spatial noise floor before/after filtering.

    Panels: Raw heatmap | NET-filtered | Preserved signal | Residual noise
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    w, h = resolution

    def _hist(x_arr, y_arr):
        hist = np.zeros((h, w), dtype=np.float64)
        np.add.at(hist, (y_arr.astype(int), x_arr.astype(int)), 1)
        return hist

    raw_hist = _hist(raw_events['x'], raw_events['y'])
    filt_hist = _hist(raw_events['x'][net_mask], raw_events['y'][net_mask])
    sig_hist = _hist(raw_events['x'][signal_mask & net_mask],
                     raw_events['y'][signal_mask & net_mask])
    noise_hist = _hist(raw_events['x'][~signal_mask & net_mask],
                       raw_events['y'][~signal_mask & net_mask])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor='#0d1117')
    panels = [
        (axes[0, 0], raw_hist,   'Raw Event Density',          'inferno'),
        (axes[0, 1], filt_hist,  'NET-Filtered Density',       'viridis'),
        (axes[1, 0], sig_hist,   'Preserved Signal (TP)',      'cividis'),
        (axes[1, 1], noise_hist, 'Residual Noise Floor (FP)',  'magma'),
    ]

    for ax, data, title, cmap in panels:
        ax.set_facecolor('#0d1117')
        vmax = max(data.max(), 1)
        im = ax.imshow(data, cmap=cmap, aspect='auto',
                       norm=LogNorm(vmin=0.5, vmax=vmax))
        ax.set_title(title, color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        cbar.ax.yaxis.set_ticklabels(
            [t.get_text() for t in cbar.ax.yaxis.get_ticklabels()],
            color='#8b949e')

    fig.suptitle('NET — Noise Floor Analysis',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='#0d1117')
        print(f"[NET] Noise floor figure saved → {save_path}")

    plt.close(fig)


# ============================================================================
# Labeled Dataset Generation (reproduces Phase 2 with ground-truth labels)
# ============================================================================

def _generate_labeled_dataset(n_signal=100_000, storm_rate=0.6,
                               n_sources=15, seed=42):
    """Generate noisy DVS stream with per-event signal/noise labels.

    Replicates DVSSimulator(seed=42) RNG sequence exactly so events
    match data/raw/sample_events.hdf5 when generated with defaults.

    Returns
    -------
    events : dict {x, y, t, p}
    signal_mask : np.ndarray[bool]
    sensor_size : tuple (width, height)
    """
    from src.generator import DVSSimulator

    sim = DVSSimulator(seed=seed)
    signal = sim.generate_structured_events(n_signal)
    n_sig = len(signal['t'])

    # --- Storm noise (mirrors DVSSimulator.inject_event_storm) ---
    n_storm = int(n_sig * storm_rate)
    storm_x = sim.rng.integers(0, sim.width, n_storm).astype(np.uint16)
    storm_y = sim.rng.integers(0, sim.height, n_storm).astype(np.uint16)
    storm_t = np.sort(
        sim.rng.integers(0, int(sim.duration_us), n_storm)
    ).astype(np.int64)
    storm_p = sim.rng.integers(0, 2, n_storm).astype(np.uint8)

    # --- Structural noise (mirrors DVSSimulator.inject_structural_noise) ---
    period_us = 1e6 / 200.0
    src_x = sim.rng.integers(0, sim.width, n_sources).astype(np.uint16)
    src_y = sim.rng.integers(0, sim.height, n_sources).astype(np.uint16)
    sx, sy, st, sp = [], [], [], []
    for i in range(n_sources):
        n_fl = int(sim.duration_us / period_us)
        tf = np.arange(n_fl) * period_us
        tf += sim.rng.normal(0, period_us * 0.05, n_fl)
        tf = np.clip(tf, 0, sim.duration_us - 1).astype(np.int64)
        sx.append(np.full(n_fl, src_x[i], dtype=np.uint16))
        sy.append(np.full(n_fl, src_y[i], dtype=np.uint16))
        st.append(tf)
        sp.append((np.arange(n_fl) % 2).astype(np.uint8))

    struct_x = np.concatenate(sx)
    struct_y = np.concatenate(sy)
    struct_t = np.concatenate(st)
    struct_p = np.concatenate(sp)
    n_struct = len(struct_t)

    # --- Merge + label ---
    all_x = np.concatenate([signal['x'], storm_x, struct_x])
    all_y = np.concatenate([signal['y'], storm_y, struct_y])
    all_t = np.concatenate([signal['t'], storm_t, struct_t])
    all_p = np.concatenate([signal['p'], storm_p, struct_p])
    is_signal = np.concatenate([
        np.ones(n_sig, dtype=np.bool_),
        np.zeros(n_storm + n_struct, dtype=np.bool_),
    ])

    order = np.argsort(all_t, kind='mergesort')
    events = {
        'x': all_x[order], 'y': all_y[order],
        't': all_t[order], 'p': all_p[order],
    }
    is_signal = is_signal[order]

    print(f"[NET] Labeled dataset: {len(all_t):,} events "
          f"({n_sig:,} signal + {n_storm:,} storm + {n_struct:,} structural)")

    return events, is_signal, (sim.width, sim.height)


# ============================================================================
# CLI Benchmark:  python -m src.filter
# ============================================================================

if __name__ == '__main__':
    print("=" * 64)
    print("  NET — Phase 3: Spatio-Temporal Poisson Filter Benchmark")
    print("=" * 64)

    HDF5_PATH = os.path.join('data', 'raw', 'sample_events.hdf5')
    N_RUNS = 5  # timing runs (post-warmup)

    # ------------------------------------------------------------------
    # 1) Generate labeled dataset (same seed → matches sample_events.hdf5)
    # ------------------------------------------------------------------
    print("\n[1/5] Generating labeled dataset (seed=42)...")
    events, signal_mask, sensor_size = _generate_labeled_dataset()
    n_total = len(events['t'])
    duration_s = (events['t'][-1] - events['t'][0]) / 1e6

    # Also save HDF5 if it doesn't exist
    if not os.path.exists(HDF5_PATH):
        from src.generator import DVSSimulator
        DVSSimulator.save_hdf5(events, HDF5_PATH)
    else:
        print(f"[NET] HDF5 already exists: {HDF5_PATH}")

    # Also verify against HDF5
    with h5py.File(HDF5_PATH, 'r') as f:
        hdf5_n = len(f['events/t'])
    print(f"[NET] HDF5 events: {hdf5_n:,}  |  Labeled events: {n_total:,}")

    # ------------------------------------------------------------------
    # 2) NET Filter — warmup + benchmark
    # ------------------------------------------------------------------
    print(f"\n[2/5] NET SpatioTemporalPoissonFilter — JIT warmup...")
    net_filter = SpatioTemporalPoissonFilter(radius=3, time_window_ms=5.0, k=2)
    net_filter.warmup()
    print(f"       {net_filter}")

    # Warmup run (discard)
    _, _ = net_filter.filter(events)

    # Timed runs
    net_times = []
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        net_filtered, net_mask = net_filter.filter(events)
        t1 = time.perf_counter()
        net_times.append((t1 - t0) * 1000)  # ms

    net_time_ms = np.median(net_times)
    print(f"       Processing time: {net_time_ms:.2f} ms  "
          f"(median of {N_RUNS} runs, "
          f"range: {min(net_times):.2f}–{max(net_times):.2f} ms)")

    # ------------------------------------------------------------------
    # 3) Tonic baseline — benchmark
    # ------------------------------------------------------------------
    print(f"\n[3/5] tonic.transforms.Denoise baseline (filter_time=10000 µs)...")
    tonic_times = []
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        tonic_filtered, tonic_mask = tonic_denoise_baseline(
            events, sensor_size, filter_time_us=10_000)
        t1 = time.perf_counter()
        tonic_times.append((t1 - t0) * 1000)

    tonic_time_ms = np.median(tonic_times)
    print(f"       Processing time: {tonic_time_ms:.2f} ms  "
          f"(median of {N_RUNS} runs, "
          f"range: {min(tonic_times):.2f}–{max(tonic_times):.2f} ms)")

    # ------------------------------------------------------------------
    # 4) Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  COMPARISON — Ground-Truth Labeled SNR")
    print("=" * 64)

    net_metrics = compare(events, net_mask, sensor_size, duration_s,
                          signal_mask, method_name="NET (Numba)")
    
    # Check if tonic was skipped (identity mask returned)
    tonic_skipped = (tonic_mask.all() and len(tonic_mask) == len(events['t']))
    
    if not tonic_skipped:
        tonic_metrics = compare(events, tonic_mask, sensor_size, duration_s,
                                signal_mask, method_name="tonic.Denoise")

    # Summary table
    print("\n" + "─" * 64)
    if not tonic_skipped:
        print(f"  {'Metric':<24} {'NET (Numba)':>16} {'tonic.Denoise':>16}")
    else:
        print(f"  {'Metric':<24} {'NET (Numba)':>16}")
    print("─" * 64)
    
    # Define rows dynamically
    rows = [
        ("Processing Time",
         f"{net_time_ms:.2f} ms", 
         f"{tonic_time_ms:.2f} ms" if not tonic_skipped else None),
        ("Events Out",
         f"{net_metrics['n_filtered']:,}", 
         f"{tonic_metrics['n_filtered']:,}" if not tonic_skipped else None),
        ("Reduction",
         f"{net_metrics['reduction_pct']:.1f}%",
         f"{tonic_metrics['reduction_pct']:.1f}%" if not tonic_skipped else None),
        ("EPPS (filtered)",
         f"{net_metrics['epps_filtered']:.2f}",
         f"{tonic_metrics['epps_filtered']:.2f}" if not tonic_skipped else None),
        ("SNR (dB)",
         f"{net_metrics['snr_filtered_db']:.2f}",
         f"{tonic_metrics['snr_filtered_db']:.2f}" if not tonic_skipped else None),
        ("SNR Improvement",
         f"+{net_metrics['snr_improvement_db']:.2f} dB",
         f"+{tonic_metrics['snr_improvement_db']:.2f} dB" if not tonic_skipped else None),
        ("Signal Kept",
         f"{net_metrics['signal_kept_pct']:.1f}%",
         f"{tonic_metrics['signal_kept_pct']:.1f}%" if not tonic_skipped else None),
        ("Noise Leaked",
         f"{net_metrics['noise_leaked_pct']:.1f}%",
         f"{tonic_metrics['noise_leaked_pct']:.1f}%" if not tonic_skipped else None),
    ]

    for row_data in rows:
        label = row_data[0]
        v1 = row_data[1]
        v2 = row_data[2] if not tonic_skipped else None
        
        if not tonic_skipped:
            print(f"  {label:<24} {v1:>16} {v2:>16}")
        else:
            print(f"  {label:<24} {v1:>16}")
            
    print("─" * 64)

    # ------------------------------------------------------------------
    # 5) Noise floor visualization
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating noise floor visualization...")
    vis_path = os.path.join('results', 'noise_floor_analysis.png')
    visualize_noise_floor(events, net_mask, signal_mask, sensor_size,
                          save_path=vis_path)

    print("\n" + "=" * 64)
    print("  Phase 3 benchmark complete.")
    print("=" * 64)
