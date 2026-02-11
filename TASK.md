# NET-VPR: Task Tracker

## Phase 1: Environment & Project Scaffolding
- [x] Directory structure (`src/`, `data/raw/`, `data/processed/`, `results/`, `scripts/`)
- [x] `requirements.txt` (numpy, h5py, matplotlib, scipy, tqdm, tonic, numba)
- [x] `.gitignore`
- [x] Research-grade `README.md` with Mermaid architecture diagram
- [x] `src/__init__.py`

## Phase 2: The "Nerve" — Data Generation
- [x] `src/generator.py` — `DVSSimulator` class
- [x] Clean diagonal-edge structured signal
- [x] Event-Storm noise injection (Poisson random spikes)
- [x] Structural noise injection (fixed-pixel flickering)
- [x] HDF5 save/load (`events/x,y,t,p`, gzip compressed)
- [x] Dark-themed scatter plot visualization
- [x] Verified: 69,511 events generated, HDF5 saved (310 KB), figure exported

## Phase 3: The "Cortex" — Spatio-Temporal Poisson Filter
- [ ] `src/filter.py` — `SpatioTemporalPoissonFilter` class
- [ ] `@numba.njit` accelerated filter loop
- [ ] EPPS + SNR metric functions
- [ ] `tonic.transforms.Denoise` baseline wrapper
- [ ] `compare()` — reduction ratio, EPPS, SNR output

## Phase 4: The "Synapse" — Adaptive Feedback Controller
- [ ] `src/controller.py` — `AdaptiveEventController` class
- [ ] Chunk-based event density calculation
- [ ] Dynamic `k` adjustment (proportional controller)
- [ ] `ControlLog` dataclass with per-chunk records

## Phase 5: The "Thesis" — Benchmarking & Visualization
- [ ] `scripts/benchmark.py` — full pipeline
- [ ] 4 publication-quality Matplotlib figures
- [ ] Recall@N theoretical metric
- [ ] Results export to `results/`

## Final Verification
- [ ] Full pipeline end-to-end (`python scripts/benchmark.py`)
- [ ] HDF5 structure matches LENS/Event-LAB format
- [ ] Filter demonstrably reduces noise while preserving signal
- [ ] Adaptive controller log shows dynamic `k` adjustment
- [ ] All 4 figures are publication-quality
