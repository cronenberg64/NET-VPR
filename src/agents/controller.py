"""
NET: Neuromorphic Event Triage — Adaptive Feedback Controller
=============================================================
Real-time adaptive controller that adjusts the filter's K (neighbor threshold)
based on incoming event density (EPPS) to suppress Event-Storms.

Logic:
    - Slices stream into 50ms chunks.
    - Calculates EPPS (Events Per Pixel Per Second).
    - Hysteresis Control:
        * EPPS > 2.0  -> K=4 (Heavy Storm)
        * EPPS > 1.2  -> K=3 (Moderate Noise)
        * Else        -> K=2 (Base State)
    - Damping: Decreasing K requires 2 consecutive chunks below threshold.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple

from src.filter import SpatioTemporalPoissonFilter, compute_epps

# ============================================================================
# Logging Data Structure
# ============================================================================

@dataclass
class ControlLog:
    t_start_us: int
    t_end_us: int
    input_count: int
    output_count: int
    epps: float
    k_value: int
    is_storm: bool


# ============================================================================
# Adaptive Controller
# ============================================================================

class AdaptiveEventController:
    """Feedback controller for dynamic noise filtering."""

    def __init__(self, resolution=(346, 260), chunk_size_ms=50.0):
        self.resolution = resolution
        self.chunk_size_ms = chunk_size_ms
        self.chunk_size_us = int(chunk_size_ms * 1000)
        
        # Control State
        self.current_k = 2
        self.consecutive_low_density = 0
        
        # Filter (re-instantiated only when params change, or just update K)
        # We'll stick to a single filter instance and update its k attribute
        self.filter_instance = SpatioTemporalPoissonFilter(
            radius=3, time_window_ms=5.0, k=self.current_k
        )
        self.filter_instance.warmup()

    def process_stream(self, events) -> Tuple[Dict, List[ControlLog]]:
        """Process event stream in localized time chunks.

        Simulates real-time processing by handling fixed-duration blocks.
        """
        sorted_indices = np.argsort(events['t'], kind='mergesort') # Ensure sorted
        t_sorted = events['t'][sorted_indices]
        
        # Slicing
        t_min = t_sorted[0]
        t_max = t_sorted[-1]
        
        logs = []
        filtered_chunks = {'x': [], 'y': [], 't': [], 'p': []}
        
        # Iterate through chunks
        current_t = t_min
        start_idx = 0
        
        print(f"[NET] Processing {len(t_sorted):,} events in {self.chunk_size_ms}ms chunks...")
        
        while current_t < t_max:
            next_t = current_t + self.chunk_size_us
            
            # Find end index for this chunk
            # (In a real system, we'd just take available buffers)
            end_idx = np.searchsorted(t_sorted, next_t)
            
            # Extract chunk
            chunk_indices = sorted_indices[start_idx:end_idx]
            if len(chunk_indices) == 0:
                current_t = next_t
                start_idx = end_idx
                continue
                
            chunk = {
                'x': events['x'][chunk_indices],
                'y': events['y'][chunk_indices],
                't': events['t'][chunk_indices],
                'p': events['p'][chunk_indices],
            }
            
            # 1. Analyze / Control
            self._update_control_logic(chunk, duration_s=self.chunk_size_ms/1000.0)
            
            # 2. Filter
            self.filter_instance.k = self.current_k
            filt_chunk, _ = self.filter_instance.filter(chunk)
            
            # 3. Log
            epps = compute_epps(chunk, self.resolution, self.chunk_size_ms/1000.0)
            logs.append(ControlLog(
                t_start_us=current_t,
                t_end_us=next_t,
                input_count=len(chunk['t']),
                output_count=len(filt_chunk['t']),
                epps=epps,
                k_value=self.current_k,
                is_storm=(self.current_k > 2)
            ))
            
            # 4. Accumulate
            for key in filtered_chunks:
                filtered_chunks[key].append(filt_chunk[key])
                
            # Prepare next loop
            current_t = next_t
            start_idx = end_idx

        # Merge results
        final_events = {
            key: np.concatenate(val) if val else np.array([], dtype=events[key].dtype)
            for key, val in filtered_chunks.items()
        }
        
        return final_events, logs

    def _update_control_logic(self, chunk, duration_s):
        """Hysteresis & Damping logic updates self.current_k."""
        epps = compute_epps(chunk, self.resolution, duration_s)
        
        target_k = 2
        if epps > 2.0:
            target_k = 4
        elif epps > 1.2:
            target_k = 3
            
        # Logic
        if target_k > self.current_k:
            # Attack: Instant increase
            self.current_k = target_k
            self.consecutive_low_density = 0
            
        elif target_k < self.current_k:
            # Decay: Require damping (2 chunks)
            self.consecutive_low_density += 1
            if self.consecutive_low_density >= 2:
                self.current_k = target_k # Drop down
                self.consecutive_low_density = 0
        else:
            # Steady state
            self.consecutive_low_density = 0


# ============================================================================
# Visualization
# ============================================================================

def plot_adaptation_trace(logs: List[ControlLog], save_path: str):
    """Plot EPPS density vs K-value trace with shaded storm regions."""
    times_s = np.array([l.t_start_us for l in logs]) / 1e6
    times_s -= times_s[0] # Relative time
    
    epps_vals = np.array([l.epps for l in logs])
    k_vals = np.array([l.k_value for l in logs])
    
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
    ax1.set_facecolor('#0d1117')
    
    # Left Axis: Density
    ax1.set_xlabel('Time (s)', color='white')
    ax1.set_ylabel('Event Density (EPPS)', color='#00d4ff', fontweight='bold')
    ax1.semilogy(times_s, epps_vals, color='#00d4ff', linewidth=1.5, label='EPPS')
    ax1.tick_params(axis='y', labelcolor='#00d4ff', colors='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.grid(True, which='both', color='#30363d', linestyle='--', alpha=0.5)
    
    # Right Axis: K-value
    ax2 = ax1.twinx()
    ax2.set_ylabel('Filter Threshold (K)', color='#ff006e', fontweight='bold')
    ax2.step(times_s, k_vals, where='post', color='#ff006e', linewidth=2.0, label='K Value')
    ax2.set_yticks([2, 3, 4, 5])
    ax2.tick_params(axis='y', labelcolor='#ff006e', colors='white')
    
    # Shading Event-Storms (Where K > 2)
    storm_indices = k_vals > 2
    if np.any(storm_indices):
        # We fill blocks where condition is true
        # Using step logic for fill_between is tricky, so we just iterate for simplicity
        # or use a boolean mask with step property.
        # Simple approach: fill where K > 2 using step function logic
        ax2.fill_between(times_s, 0, 10, where=storm_indices, 
                         color='#ff006e', alpha=0.15, step='post', transform=ax2.get_xaxis_transform())

    # Styling
    ax1.set_facecolor('#0d1117')
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    plt.title('NET Adaptive Controller Trace', color='white', fontsize=14, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
        print(f"[NET] Adaptation trace saved → {save_path}")
    
    plt.close(fig)


# ============================================================================
# Benchmark Main
# ============================================================================

if __name__ == '__main__':
    from src.filter import compare, _generate_labeled_dataset
    
    print("="*60)
    print("  NET Phase 4: Adaptive Controller Benchmark")
    print("="*60)
    
    # 1. Generate Data
    print("\n[1/3] Generating labeled dataset...")
    events, signal_mask, sensor_size = _generate_labeled_dataset()
    duration_s = (events['t'][-1] - events['t'][0]) / 1e6
    
    # 2. Run Controller
    print("\n[2/3] Running Adaptive Loop (50ms chunks)...")
    controller = AdaptiveEventController(resolution=sensor_size, chunk_size_ms=50)
    
    t0 = time.perf_counter()
    filtered_events, logs = controller.process_stream(events)
    t1 = time.perf_counter()
    
    proc_time = (t1 - t0) * 1000
    print(f"       Total time: {proc_time:.2f} ms")
    
    # 3. Analyze
    print("\n[3/3] Analysis & Visualization...")
    
    # Reconstruct mask (approximation for metric func compatibility)
    # Since we strictly partitioned, we can match by timestamp logic if needed,
    # but simplest is to trust the output counts or just recalc overlap.
    # For accurate SNR 'compare', we need a boolean mask aligned to 'events'.
    # Because 'filtered_events' is a subset, we can map it back.
    
    # Efficient mask reconstruction:
    # filtered_t is a subset of events_t. Since both are sorted, standard inclusion check works.
    # Using numpy in1d is fast enough for 100k events.
    keep_mask = np.in1d(events['t'], filtered_events['t'])
    
    metrics = compare(events, keep_mask, sensor_size, duration_s, signal_mask, method_name="Adaptive NET")
    
    plot_path = os.path.join('results', 'adaptation_trace.png')
    plot_adaptation_trace(logs, plot_path)
    
    print("\n" + "="*60)
    print("  Phase 4 Complete.")
    print("="*60)
