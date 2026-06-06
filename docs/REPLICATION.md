# Replication Notes

Practical details for replicating the benchmark. For the full experimental
methodology, refer to Chapter 3 of the thesis.

## Test machine

- CPU: AMD Ryzen 7 5800H (8 cores / 16 threads, 3.2-4.4 GHz)
- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB GDDR6)
- RAM: 32 GB DDR4-3200
- OS: Windows 10 Pro 22H2
- Storage: NVMe SSD (not a factor in benchmark timing)

## Test conditions

- All non-essential applications closed during benchmark runs.
- System idle for ~2 minutes before starting a batch.
- No user interaction during execution.
- Single-threaded execution, no parallel processing.
- Garbage collection between tests (handled by the pipeline).

## Exact library versions

The versions used for the thesis results are pinned in
`environments/environment_traditional.yml`. Key packages:

- Python 3.11, PyVista 0.46.4, fast-simplification 0.1.13
- Open3D 0.19.0, meshoptimizer 0.2.20a5, trimesh (latest)
- CGAL 6.1 (compiled from source, ~62 min build)
- NumPy 2.4.0, SciPy, pandas, matplotlib, seaborn

## Expected variability across machines

Should match exactly: geometric accuracy values (deterministic algorithms),
success/failure patterns, relative performance rankings.

Will vary: absolute execution times (~10% from hardware differences),
absolute memory usage (~15% from OS/Python overhead).

## What's not in the repository

The Arma Reforger production assets (8 meshes, provided by Bohemia
Interactive under NDA) are not included. The benchmark pipeline works
on any OBJ meshes. Stanford 3D Scanning Repository models are freely
available for replication of the scalability analysis.