# Mesh Simplification Benchmark

Research repository for the thesis *Comparative Analysis of Open-Source Mesh
Simplification Methods for Game Development* (MSc Game Technology, Breda
University of Applied Sciences, 2026).

Benchmarks four open-source mesh simplification methods --
[fast-simplification](https://github.com/pyvista/fast-simplification),
[Open3D](https://github.com/isl-org/Open3D),
[meshoptimizer](https://github.com/zeux/meshoptimizer), and
[CGAL](https://www.cgal.org) -- across three dimensions: geometric accuracy
(Hausdorff distance, RMSE), processing performance (time, memory), and
perceptual quality (pairwise comparison study, ~150 participants).

## Setup

Requires Python 3.12+.

```bash
# Option A: conda (recommended)
conda env create -f environments/environment_traditional.yml
conda activate mesh-simplification

# Option B: pip
pip install pyvista fast-simplification open3d meshoptimizer trimesh psutil \
            numpy scipy pandas matplotlib seaborn tqdm
conda install -c conda-forge rtree   # needed for geometric accuracy
```

To test the CGAL method, compile the C++ wrapper:

```bash
cd cgal_build && cmake .. && cmake --build . --config Release
```

See `environments/README.md` for details and neural method environments.

## Usage

Run the benchmark on a directory of OBJ files:

```bash
python mesh_simplifier_batch.py \
    -i ./test_meshes \
    -o ./results \
    --methods all \
    --reduction-levels 90 80 50 \
    --repetitions 3
```

This produces a `batch_report.json` with all metrics, simplified meshes
organized by asset/method, and a `benchmark_log.txt`.

Generate analysis outputs:

```bash
# Benchmark analysis (boxplots, tables, CSVs)
python analyze_benchmarks.py -i ./results/batch_report.json -o ./output

# Perceptual study analysis
python analyze_perceptual.py \
    -p ./data/participant_responses.json \
    -b ./data/batch_report.json \
    -o ./output

# Scalability analysis (Stanford models)
python analyze_scalability.py -i ./data/scalability_report.json -o ./output

# Error heatmaps (per-vertex colored meshes)
python generate_heatmaps.py --batch \
    --results-dir ./results --original-dir ./test_meshes \
    --output-dir ./figures --max-error 2.0

# Comparison charts
python generate_comparison_charts.py -i ./results/batch_report.json -o ./figures
```

Run `python <script>.py --help` for all options.

## Repository structure

```
.
|-- mesh_simplifier_batch.py        # Main benchmark pipeline
|-- mesh_simplifier.py              # Single-mesh simplifier (interactive use)
|-- cgal_simplify.cpp               # CGAL C++ wrapper
|-- CMakeLists.txt
|
|-- analyze_benchmarks.py           # Benchmark data analysis
|-- analyze_perceptual.py           # Perceptual study analysis
|-- analyze_scalability.py          # Scalability analysis (Stanford models)
|-- generate_comparison_charts.py   # Comparison charts from batch report
|-- generate_heatmaps.py           # Per-vertex error visualization
|
|-- data/                           # Raw data (JSON)
|   |-- batch_report.json           # Benchmark results
|   \-- participant_responses.json  # Perceptual study responses (anonymized)
|
|-- results/                        # Analysis outputs (figures, tables, CSVs)
|-- test_meshes/                    # Stanford Bunny + Utah Teapot for testing
|-- environments/                   # Conda environment files
|-- neural_methods/                 # Neural method investigation (excluded)
|   \-- DEVELOPMENT_LOG.md          # 40+ hours of documented attempts
\-- docs/
    \-- METHODOLOGY.md
```

## Data

The `data/` directory contains the raw research data in JSON format.

`batch_report.json` holds all benchmark results: per-asset, per-method,
per-reduction-level timings, memory usage, and geometric accuracy metrics
across three repetitions.

`participant_responses.json` holds anonymized responses from the perceptual
quality pairwise comparison study (~2800 judgments).

The Arma Reforger production assets used in the thesis are provided under NDA
by Bohemia Interactive and are not included. The benchmark pipeline can be run
on any OBJ meshes. Stanford 3D Scanning Repository models (bunny, dragon,
armadillo, lucy) are freely available at
http://graphics.stanford.edu/data/3Dscanrep/.

## Citation

```bibtex
@mastersthesis{gallo2026mesh,
    title   = {Comparative Analysis of Open-Source Mesh Simplification
               Methods for Game Development},
    author  = {Gallo, Lukas},
    school  = {Breda University of Applied Sciences},
    year    = {2026},
    type    = {Master's Thesis},
    address = {Breda, Netherlands}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

## License

GPL-3.0-or-later, due to CGAL's Surface_mesh_simplification package.
The three non-CGAL methods (fast-simplification, Open3D, meshoptimizer) are
MIT-licensed and can be used independently without GPL obligations.
See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

## Acknowledgments

- Supervisors: Ruben Tack, Robbie Storm
- Academic advisor: Thomas Buijtenweg
- Industry partner: Bohemia Interactive
- Test data: Stanford 3D Scanning Repository
- Libraries: fast-simplification, Open3D, meshoptimizer, CGAL teams
- Institution: Breda University of Applied Sciences