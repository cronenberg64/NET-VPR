"""
NET: Neuromorphic Event Triage — Final Benchmark
================================================

Compares:
1. Static NET (K=2)
2. Adaptive NET (Controller)
3. Tonic Baseline (Denoise)

Metrics:
- Latency (ms)
- SNR (dB)
- Reduction (%)
- Signal Preservation (%)
- Energy Efficiency (Theoretical)

Outputs:
- Console table
- results/final_comparison.png
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from src.filter import SpatioTemporalPoissonFilter, tonic_denoise_baseline, compare, _generate_labeled_dataset
from src.controller import AdaptiveEventController

def run_benchmark():
    print("=" * 64)
    print("  NET: Neuromorphic Event Triage — Final Benchmark")
    print("=" * 64)

    # 1. Dataset
    print("\n[1/4] Generating standardized dataset...")
    events, signal_mask, sensor_size = _generate_labeled_dataset()
    duration_s = (events['t'][-1] - events['t'][0]) / 1e6
    n_total = len(events['t'])
    
    results = {}

    # ----------------------------------------------------------------
    # 2. Static NET (K=2)
    # ----------------------------------------------------------------
    print("\n[2/4] Benchmarking Static NET (K=2)...")
    net_static = SpatioTemporalPoissonFilter(radius=3, time_window_ms=5.0, k=2)
    net_static.warmup()
    
    # Timing
    t0 = time.perf_counter()
    static_filtered, static_mask = net_static.filter(events)
    t1 = time.perf_counter()
    static_time = (t1 - t0) * 1000
    
    results['Static NET'] = {
        'time_ms': static_time,
        'metrics': compare(events, static_mask, sensor_size, duration_s, signal_mask, method_name="Static NET")
    }

    # ----------------------------------------------------------------
    # 3. Adaptive NET (Controller)
    # ----------------------------------------------------------------
    print("\n[3/4] Benchmarking Adaptive NET (Synapse)...")
    controller = AdaptiveEventController(resolution=sensor_size, chunk_size_ms=50)
    
    t0 = time.perf_counter()
    adaptive_filtered_events, logs = controller.process_stream(events)
    t1 = time.perf_counter()
    adaptive_time = (t1 - t0) * 1000
    
    # Reconstruct mask
    adaptive_mask = np.in1d(events['t'], adaptive_filtered_events['t'])
    
    results['Adaptive NET'] = {
        'time_ms': adaptive_time,
        'metrics': compare(events, adaptive_mask, sensor_size, duration_s, signal_mask, method_name="Adaptive NET")
    }
    
    # Save trace
    from src.controller import plot_adaptation_trace
    plot_adaptation_trace(logs, os.path.join('results', 'adaptive_k_trace.png'))

    # ----------------------------------------------------------------
    # 4. Tonic Baseline
    # ----------------------------------------------------------------
    print("\n[4/4] Benchmarking Tonic Baseline...")
    t0 = time.perf_counter()
    tonic_filtered, tonic_mask = tonic_denoise_baseline(events, sensor_size)
    t1 = time.perf_counter()
    tonic_time = (t1 - t0) * 1000
    
    # Check if skipped
    tonic_skipped = (tonic_mask.all() and len(tonic_mask) == len(events['t']))
    if not tonic_skipped:
        results['Tonic'] = {
            'time_ms': tonic_time,
            'metrics': compare(events, tonic_mask, sensor_size, duration_s, signal_mask, method_name="Tonic")
        }
    else:
        results['Tonic'] = None
        print("      (Skipped due to missing dependency)")

    # Save to file because stdout is unreliable in agent environment
    summary_path = os.path.join('results', 'benchmark_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# Benchmark Results\n\n")
        methods = [k for k in results.keys() if results[k] is not None]
        for m in methods:
            metrics = results[m]['metrics']
            f.write(f"## {m}\n")
            f.write(f"- **Time**: {results[m]['time_ms']:.2f} ms\n")
            f.write(f"- **Events**: {metrics['n_raw']:,} → {metrics['n_filtered']:,}\n")
            f.write(f"- **Reduction**: {metrics['reduction_pct']:.2f}%\n")
            f.write(f"- **SNR Improvement**: {metrics['snr_improvement_db']:.2f} dB\n")
            f.write(f"- **Signal Kept**: {metrics['signal_kept_pct']:.2f}%\n")
            f.write(f"- **Noise Leaked**: {metrics['noise_leaked_pct']:.2f}%\n\n")
            
    print(f"\n[NET] Results saved to {summary_path}")
    
    # ----------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------
    plot_comparison(results)

def plot_comparison(results):
    print("\nGenerating final comparison figure...")
    
    methods = [k for k in results.keys() if results[k] is not None]
    latency = [results[m]['time_ms'] for m in methods]
    snr_imp = [results[m]['metrics']['snr_improvement_db'] for m in methods]
    
    # Energy Efficiency (Approximate metric: Events Processed / Time)
    throughput = [results[m]['metrics']['n_raw'] / (results[m]['time_ms']/1000) for m in methods]
    efficiency = [t / min(throughput) for t in throughput]
    
    x = np.arange(len(methods))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d1117')
    
    metrics_cfg = [
        (latency, 'Latency (ms)', '#ff006e', 'lower is better'),
        (snr_imp, 'SNR Improvement (dB)', '#00d4ff', 'higher is better'),
        (efficiency, 'Rel. Energy Eff. (x)', '#00ff9f', 'higher is better')
    ]
    
    for ax, (data, title, color, note) in zip(axes, metrics_cfg):
        ax.set_facecolor('#0d1117')
        bars = ax.bar(methods, data, color=color, alpha=0.8)
        ax.set_title(title, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        
        # Labels
        for rect in bars:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    f'{height:.1f}',
                    ha='center', va='bottom', color='white')
                    
    fig.suptitle('NET Framework vs. State-of-the-Art', color='white', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join('results', 'final_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n[NET] Final comparison saved → {save_path}")

if __name__ == '__main__':
    run_benchmark()
