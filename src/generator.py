"""
NET: Neuromorphic Event Triage — DVS Event Stream Generator
============================================================

Simulates a Dynamic Vision Sensor (DVS) producing structured event streams
with configurable noise injection for benchmarking spatio-temporal filters.

Event format: (x, y, t, p) where
    x, y  = pixel coordinates (uint16)
    t     = timestamp in microseconds (int64)
    p     = polarity, 0=OFF / 1=ON (uint8)

HDF5 layout follows LENS / Event-LAB convention:
    events/x, events/y, events/t, events/p (gzip compressed)
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# DVS Simulator
# ---------------------------------------------------------------------------

class DVSSimulator:
    """Synthetic Dynamic Vision Sensor simulator.

    Generates structured events (moving diagonal edge) and injects two
    categories of noise to create realistic 'Event-Storm' scenarios:
        - Event-Storm noise: Poisson-distributed random spikes across the
          entire sensor array.
        - Structural noise: High-frequency flickering at fixed pixel
          locations (simulating LED / fluorescent light artifacts).

    Parameters
    ----------
    resolution : tuple of int
        Sensor resolution as (width, height). Default (346, 260) matches
        the DAVIS346 sensor used in many VPR datasets.
    duration_ms : float
        Total simulation duration in milliseconds.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, resolution=(346, 260), duration_ms=500.0, seed=42):
        self.width, self.height = resolution
        self.duration_ms = duration_ms
        self.duration_us = duration_ms * 1000.0  # microseconds
        self.rng = np.random.default_rng(seed)

    # ----- Structured signal -----

    def generate_structured_events(self, n_events=50_000):
        """Generate a clean diagonal edge sweeping across the sensor.

        The edge moves from top-left to bottom-right over the full
        duration, producing spatially correlated ON/OFF event pairs
        along its leading and trailing borders.

        Parameters
        ----------
        n_events : int
            Approximate number of signal events to generate.

        Returns
        -------
        events : dict with keys 'x', 'y', 't', 'p' (numpy arrays)
        """
        # Time axis — uniformly spaced across the full duration
        t = np.sort(self.rng.uniform(0, self.duration_us, n_events)).astype(np.int64)

        # Normalised progress [0, 1] across the duration
        progress = t / self.duration_us

        # Diagonal edge: a line y = x + offset, offset sweeps from
        # -height to +width so the edge traverses the entire frame.
        edge_offset = -self.height + progress * (self.width + self.height)

        # Events cluster tightly around the edge with small Gaussian jitter
        # to keep the signal *clean and distinguishable* from noise.
        edge_jitter = self.rng.normal(0, 1.5, n_events)  # tight 1.5px sigma

        x = np.clip(
            (np.arange(n_events) % self.width).astype(np.float64) * 0.0
            + self.rng.uniform(0, self.width, n_events),
            0, self.width - 1
        ).astype(np.uint16)

        # y follows the diagonal: y = x - offset + jitter
        y_float = x.astype(np.float64) - edge_offset + edge_jitter
        y = np.clip(y_float, 0, self.height - 1).astype(np.uint16)

        # Only retain events that actually lie on / near the edge
        # (discard those that fell outside the sensor after clamping).
        # This gives a *sharp* diagonal line.
        on_edge = np.abs(y_float - y.astype(np.float64)) < 2.0
        mask = on_edge & (y > 0) & (y < self.height - 1)

        # Polarity: leading edge = ON(1), trailing = OFF(0)
        p = (edge_jitter > 0).astype(np.uint8)

        events = {
            'x': x[mask],
            'y': y[mask],
            't': t[mask],
            'p': p[mask],
        }

        # Re-sort by timestamp after masking
        order = np.argsort(events['t'])
        for key in events:
            events[key] = events[key][order]

        return events

    # ----- Noise injection -----

    def inject_event_storm(self, events, storm_rate=0.6):
        """Add Poisson-distributed random spike noise ('Event-Storm').

        Parameters
        ----------
        events : dict
            Clean event stream.
        storm_rate : float
            Noise-to-signal ratio. 0.6 means 60% as many noise events
            as signal events are added.

        Returns
        -------
        noisy_events : dict
            Combined signal + storm noise, sorted by timestamp.
        """
        n_noise = int(len(events['t']) * storm_rate)

        noise_x = self.rng.integers(0, self.width, n_noise).astype(np.uint16)
        noise_y = self.rng.integers(0, self.height, n_noise).astype(np.uint16)
        noise_t = np.sort(
            self.rng.integers(0, int(self.duration_us), n_noise)
        ).astype(np.int64)
        noise_p = self.rng.integers(0, 2, n_noise).astype(np.uint8)

        return self._merge_events(events, {
            'x': noise_x, 'y': noise_y, 't': noise_t, 'p': noise_p
        })

    def inject_structural_noise(self, events, n_sources=15,
                                 flicker_freq_hz=200.0):
        """Add high-frequency flickering at fixed pixel locations.

        Simulates LED indicator lights or fluorescent lighting artifacts
        that produce periodic ON/OFF events at fixed positions.

        Parameters
        ----------
        events : dict
            Event stream (may already contain storm noise).
        n_sources : int
            Number of flickering pixel sources.
        flicker_freq_hz : float
            Flickering frequency in Hz.

        Returns
        -------
        noisy_events : dict
            Combined stream with structural noise added.
        """
        period_us = 1e6 / flicker_freq_hz  # period in microseconds

        # Fixed source positions scattered across the sensor
        src_x = self.rng.integers(0, self.width, n_sources).astype(np.uint16)
        src_y = self.rng.integers(0, self.height, n_sources).astype(np.uint16)

        all_x, all_y, all_t, all_p = [], [], [], []

        for i in range(n_sources):
            # Each source flickers for the entire duration
            n_flickers = int(self.duration_us / period_us)
            t_flicker = np.arange(n_flickers) * period_us
            # Add small temporal jitter to avoid perfectly periodic artifacts
            t_flicker += self.rng.normal(0, period_us * 0.05, n_flickers)
            t_flicker = np.clip(t_flicker, 0, self.duration_us - 1).astype(np.int64)

            all_x.append(np.full(n_flickers, src_x[i], dtype=np.uint16))
            all_y.append(np.full(n_flickers, src_y[i], dtype=np.uint16))
            all_t.append(t_flicker)
            # Alternating polarity
            all_p.append((np.arange(n_flickers) % 2).astype(np.uint8))

        struct_noise = {
            'x': np.concatenate(all_x),
            'y': np.concatenate(all_y),
            't': np.concatenate(all_t),
            'p': np.concatenate(all_p),
        }

        return self._merge_events(events, struct_noise)

    # ----- I/O -----

    @staticmethod
    def save_hdf5(events, path):
        """Save events to HDF5 using LENS / Event-LAB layout.

        Layout
        ------
        events/x  (uint16, gzip)
        events/y  (uint16, gzip)
        events/t  (int64,  gzip)
        events/p  (uint8,  gzip)

        Parameters
        ----------
        events : dict
            Event stream with keys 'x', 'y', 't', 'p'.
        path : str
            Output file path (*.hdf5).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with h5py.File(path, 'w') as f:
            grp = f.create_group('events')
            grp.create_dataset('x', data=events['x'], compression='gzip', compression_opts=4)
            grp.create_dataset('y', data=events['y'], compression='gzip', compression_opts=4)
            grp.create_dataset('t', data=events['t'], compression='gzip', compression_opts=4)
            grp.create_dataset('p', data=events['p'], compression='gzip', compression_opts=4)

        size_kb = os.path.getsize(path) / 1024
        print(f"[NET] Saved {len(events['t']):,} events → {path}  ({size_kb:.1f} KB)")

    @staticmethod
    def load_hdf5(path):
        """Load events from LENS / Event-LAB HDF5 format.

        Returns
        -------
        events : dict with keys 'x', 'y', 't', 'p'
        """
        with h5py.File(path, 'r') as f:
            events = {
                'x': f['events/x'][:],
                'y': f['events/y'][:],
                't': f['events/t'][:],
                'p': f['events/p'][:],
            }
        print(f"[NET] Loaded {len(events['t']):,} events ← {path}")
        return events

    # ----- Visualization -----

    @staticmethod
    def visualize_events(events, duration_ms=100, title="DVS Event Stream",
                          save_path=None):
        """2D scatter plot of events within the first `duration_ms`.

        ON events (p=1) are shown in cyan, OFF events (p=0) in magenta,
        following standard DVS visualization conventions.

        Parameters
        ----------
        events : dict
        duration_ms : float
            Only plot events within [0, duration_ms].
        title : str
        save_path : str or None
            If provided, save figure to this path.
        """
        cutoff_us = duration_ms * 1000
        mask = events['t'] <= cutoff_us

        x = events['x'][mask]
        y = events['y'][mask]
        p = events['p'][mask]

        fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')

        # Custom colormap: OFF=magenta, ON=cyan
        colors = np.where(p == 1, '#00d4ff', '#ff006e')

        ax.scatter(x, y, c=colors, s=0.4, alpha=0.7, edgecolors='none')
        ax.set_xlim(0, max(x.max() + 10, 346))
        ax.set_ylim(max(y.max() + 10, 260), 0)  # invert y-axis (image coords)
        ax.set_xlabel('x (pixels)', color='white', fontsize=12)
        ax.set_ylabel('y (pixels)', color='white', fontsize=12)
        ax.set_title(f'{title}  (first {duration_ms:.0f} ms — '
                     f'{mask.sum():,} events)', color='white', fontsize=14,
                     fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='#0d1117', markerfacecolor='#00d4ff',
                   markersize=8, label='ON (p=1)'),
            Line2D([0], [0], marker='o', color='#0d1117', markerfacecolor='#ff006e',
                   markersize=8, label='OFF (p=0)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight',
                        facecolor='#0d1117')
            print(f"[NET] Figure saved → {save_path}")

        plt.show()
        plt.close(fig)

    # ----- Helpers -----

    @staticmethod
    def _merge_events(a, b):
        """Merge two event dicts and sort by timestamp."""
        merged = {
            key: np.concatenate([a[key], b[key]])
            for key in ('x', 'y', 't', 'p')
        }
        order = np.argsort(merged['t'], kind='mergesort')
        return {key: merged[key][order] for key in merged}


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def generate_sample_dataset(output_dir='data/raw', n_signal=100_000,
                             storm_rate=0.6, n_flicker_sources=15,
                             visualize=True, save_vis=True):
    """Generate a complete noisy DVS dataset and optionally visualize it.

    Parameters
    ----------
    output_dir : str
        Directory for the HDF5 file.
    n_signal : int
        Number of structured signal events.
    storm_rate : float
        Poisson storm noise ratio (0.6 = 60% of signal count).
    n_flicker_sources : int
        Number of structural noise sources.
    visualize : bool
        Whether to display the scatter plot.
    save_vis : bool
        Whether to save the scatter plot to results/.

    Returns
    -------
    events : dict
        The full noisy event stream.
    hdf5_path : str
        Path to the saved HDF5 file.
    """
    print("=" * 60)
    print("  NET — Neuromorphic Event Triage: Data Generator")
    print("=" * 60)

    sim = DVSSimulator(seed=42)

    # Step 1: Clean structured signal
    print("\n[1/4] Generating structured diagonal-edge signal...")
    clean = sim.generate_structured_events(n_signal)
    print(f"       → {len(clean['t']):,} clean events generated")

    # Step 2: Inject Event-Storm noise
    print(f"[2/4] Injecting Event-Storm noise (rate={storm_rate})...")
    with_storm = sim.inject_event_storm(clean, storm_rate=storm_rate)
    n_storm = len(with_storm['t']) - len(clean['t'])
    print(f"       → +{n_storm:,} storm events")

    # Step 3: Inject Structural noise
    print(f"[3/4] Injecting Structural noise ({n_flicker_sources} sources)...")
    full = sim.inject_structural_noise(with_storm, n_sources=n_flicker_sources)
    n_struct = len(full['t']) - len(with_storm['t'])
    print(f"       → +{n_struct:,} structural events")

    # Summary
    total = len(full['t'])
    signal_pct = len(clean['t']) / total * 100
    print(f"\n       Total events: {total:,}")
    print(f"       Signal:       {len(clean['t']):,}  ({signal_pct:.1f}%)")
    print(f"       Storm noise:  {n_storm:,}  ({n_storm/total*100:.1f}%)")
    print(f"       Struct noise: {n_struct:,}  ({n_struct/total*100:.1f}%)")

    # Step 4: Save
    hdf5_path = os.path.join(output_dir, 'sample_events.hdf5')
    print(f"\n[4/4] Saving to HDF5 (gzip compressed)...")
    DVSSimulator.save_hdf5(full, hdf5_path)

    # Visualize
    if visualize:
        vis_path = os.path.join('results', 'event_storm_raw.png') if save_vis else None
        DVSSimulator.visualize_events(
            full,
            duration_ms=100,
            title="Raw DVS Stream with Event-Storm Noise",
            save_path=vis_path
        )

    return full, hdf5_path


# ---------------------------------------------------------------------------
# CLI entry-point:  python -m src.generator
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    generate_sample_dataset()
