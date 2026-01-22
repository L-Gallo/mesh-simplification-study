# Mesh Simplification Benchmark Suite

> **Research Repository**: Master's Thesis - Comparing Mesh Simplification Methods for Game Development  
> Breda University of Applied Sciences | January 2026

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Comprehensive benchmark suite for comparing traditional geometric mesh simplification methods. Evaluates four open-source approaches (fast-simplification, Open3D, meshoptimizer, CGAL) across performance, accuracy, and stability metrics.

**Research Goal**: Provide evidence-based tool selection guidance for resource-constrained game developers.

**Scope Note**: This benchmark focuses on **traditional geometric methods** that are accessible to indie developers. Neural simplification methods (MeshCNN, Neural Mesh Simplifier) were excluded after extensive testing revealed fundamental scalability barriers (267× memory requirements, 750-edge architectural limits). See `neural_methods/DEVELOPMENT_LOG.md` for detailed exclusion rationale with empirical evidence.

**Key Features**:
- Multi-method batch processing with statistical analysis
- Geometric accuracy computation (Hausdorff distance, RMSE)
- Error visualization with unified color scales
- Stability analysis through repeated measurements
- Automated reporting with JSON output

## Quick Start

### Prerequisites

- Python 3.12+
- Windows/Linux/macOS
- ~2GB RAM for testing (scales with mesh size)

### Installation

```bash
# Option 1: Use conda environment file (recommended)
conda env create -f environments/environment_traditional.yml
conda activate mesh-simplification

# Option 2: Manual installation
conda create -n mesh-simplification python=3.12
conda activate mesh-simplification

# Install core dependencies
pip install pyvista fast-simplification open3d meshoptimizer trimesh psutil numpy matplotlib scipy

# Install rtree for geometric accuracy (required)
conda install -c conda-forge rtree
# OR on some systems:
# pip install rtree

# Optional: better visualizations
pip install seaborn tqdm
```

**Note on Neural Methods**: This repository includes environment files for excluded neural methods (`environment_meshcnn.yml`, `environment_neural.yml`) for research transparency. See `environments/README.md` for why these were excluded. Installing them is NOT recommended unless you want to verify the exclusion rationale.

**CGAL Setup** (optional, for testing CGAL method):
- Requires C++ compilation
- See `cgal_simplify.cpp` and `CMakeLists.txt`
- Pre-compiled Windows binary included (if available)

### Basic Usage

```bash
# 1. Run benchmark on test meshes
python mesh_simplifier_batch.py -i ./test_meshes -o ./results --methods all

# 2. Generate error visualizations
python generate_heatmaps.py --batch --results-dir ./results --original-dir ./test_meshes --output-dir ./figures --max-error 2.0

# 3. Create comparison charts
python generate_comparison_charts.py -i ./results/batch_report.json -o ./figures
```

## Repository Structure

```
mesh-simplification-benchmark/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Academic use license
├── .gitignore                         # Git exclusions
│
├── mesh_simplifier_batch.py          # Main benchmark script
├── generate_heatmaps.py              # Error visualization
├── generate_comparison_charts.py     # Statistical charts
│
├── cgal_simplify.cpp                 # CGAL wrapper (optional)
├── CMakeLists.txt                    # CGAL build config
│
├── environments/                      # Conda environment configs
│   ├── environment_traditional.yml   # Working methods (included)
│   ├── environment_meshcnn.yml       # Neural method (excluded)
│   ├── environment_neural.yml        # Neural method (excluded)
│   └── README.md                     # Exclusion rationale
│
├── neural_methods/                    # R&D - Implementation attempts
│   ├── README.md                     # Attempt documentation
│   ├── run_meshcnn.py                # Wrapper scripts (unsuccessful)
│   ├── run_neural_simp.py
│   └── prepare_mesh_for_meshcnn.py
│
├── test_meshes/                      # Example test data
│   ├── bunny.obj                     # Stanford Bunny
│   ├── teapot.obj                    # Utah Teapot
│   └── README.md                     # Data attribution
│
└── docs/                             # Additional documentation
    ├── METHODOLOGY.md                # Experimental protocol
    └── METRICS.md                    # Metric definitions
```

## Core Scripts

### 1. `mesh_simplifier_batch.py`

Batch process multiple meshes across different methods and reduction levels.

**Key Parameters**:
```bash
--input-dir, -i          # Directory with .obj files
--output-dir, -o         # Results directory
--methods, -m            # Methods to test (fast-simplification, open3d, meshoptimizer, cgal, all)
--reduction-levels       # Target reductions (default: 75 50 25)
--repetitions            # Stability testing runs (default: 3)
--compute-accuracy       # Enable geometric metrics (default: on)
```

**Example**:
```bash
python mesh_simplifier_batch.py \
    -i ./test_meshes \
    -o ./results \
    --methods all \
    --reduction-levels 75 50 25 \
    --repetitions 3
```

**Output**: 
- `batch_report.json`: Complete results with statistics
- `benchmark_log.txt`: Detailed execution log
- Simplified meshes in organized folder structure

### 2. `generate_heatmaps.py`

Create per-vertex error visualizations with consistent color scales.

**Critical for Thesis**: Always use `--max-error` or `--use-percentile-max` to ensure visual comparisons are valid.

**Example**:
```bash
# Batch mode with manual scale (recommended)
python generate_heatmaps.py \
    --batch \
    --results-dir ./results \
    --original-dir ./test_meshes \
    --output-dir ./figures \
    --max-error 2.0

# Automatic outlier filtering
python generate_heatmaps.py \
    --batch \
    --results-dir ./results \
    --original-dir ./test_meshes \
    --output-dir ./figures \
    --use-percentile-max
```

**Output**:
- `.ply` files with vertex colors (import into Blender/MeshLab)
- `.png` colorbar legends with statistics

### 3. `generate_comparison_charts.py`

Generate statistical comparison charts from benchmark results.

**Example**:
```bash
python generate_comparison_charts.py \
    -i ./results/batch_report.json \
    -o ./figures
```

**Output Charts**:
- Performance comparison (time, memory)
- Geometric accuracy comparison
- Success/stability rates
- Combined overview

## Input Requirements

- **Format**: OBJ files only (CGAL compatibility requirement)
- **Naming**: Descriptive filenames (e.g., `character_model.obj`)
- **Quality**: Clean manifold meshes recommended
- **Size**: Tested on meshes from 6K to 100K+ faces

## Output Format

### JSON Report Structure

```json
{
  "system": {
    "cpu": "AMD Ryzen 7 5800H",
    "gpu": "NVIDIA RTX 3060",
    "ram_gb": 32,
    "python_version": "3.12.0"
  },
  "configuration": {
    "methods": ["fast-simplification", "open3d", "meshoptimizer", "cgal"],
    "reduction_levels": [75, 50, 25],
    "repetitions": 3
  },
  "summary": {
    "overall_success_rate": 96.7,
    "overall_stability_rate": 88.3
  },
  "method_statistics": {
    "fast-simplification": {
      "success_rate": 100.0,
      "stability_rate": 93.3,
      "mean_time_ms": 234.5,
      "mean_hausdorff": 0.42
    }
  },
  "assets": {
    "bunny": {
      "methods": {
        "fast-simplification": {
          "75%": {
            "repetitions": [
              {"run": 1, "time_ms": 245.3, "hausdorff": 0.42},
              {"run": 2, "time_ms": 248.1, "hausdorff": 0.41},
              {"run": 3, "time_ms": 246.8, "hausdorff": 0.43}
            ],
            "statistics": {
              "mean_time_ms": 246.7,
              "cv_time": 0.57,
              "stable": true
            }
          }
        }
      }
    }
  }
}
```

## Pilot Test Results

Benchmark on Stanford Bunny (69K faces) and Utah Teapot (6K faces):

| Method | Mean Time (ms) | Mean Hausdorff (%) | Success Rate | Stability |
|--------|----------------|-------------------|--------------|-----------|
| meshoptimizer | 11.4 | 0.401 | 100% | 100% |
| fast-simplification | 31.6 | 1.427 | 100% | 100% |
| Open3D | 192.7 | 0.345 | 100% | 66.7% |
| CGAL | 723.7 | 0.370 | 100% | 100% |

**Key Finding**: meshoptimizer achieves 63× faster execution than CGAL with only 7% higher geometric error.

## Known Issues

### Heatmap Visualization

**Issue**: All heatmaps appear uniformly blue  
**Cause**: Unified scale maximum set too high (often due to one outlier)  
**Solution**: Run with `--use-percentile-max` or manually set `--max-error` based on debug output

### Open3D Instability

**Issue**: High coefficient of variation (CV > 15%) at 50% reduction  
**Cause**: Adaptive algorithm behavior (undocumented in API)  
**Impact**: Variable execution time, but 100% success rate maintained

### Windows Encoding

**Issue**: UnicodeEncodeError in log files  
**Solution**: Ensure latest script version (all Unicode replaced with ASCII)

## Metrics Definitions

**Hausdorff Distance**: Maximum distance from any point on simplified mesh to nearest point on original mesh. Measures worst-case error.

**RMSE (Root Mean Square Error)**: Average per-vertex error. Measures typical error magnitude.

**Coefficient of Variation (CV)**: (std / mean) × 100. Tests with CV > 15% flagged as unstable.

**Success Rate**: Percentage of tests that complete without errors.

**Stability Rate**: Percentage of tests with CV ≤ 15%.

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@mastersthesis{gallo2026mesh,
  title={Comparing Mesh Simplification Methods for Independent Game Developers},
  author={Gallo, Lukas},
  school={Breda University of Applied Sciences},
  year={2026},
  type={Master's Thesis},
  address={Breda, Netherlands}
}
```

Or click **"Cite this repository"** on GitHub (generated from [CITATION.cff](CITATION.cff)).

## License

This project is licensed under **GPL-3.0-or-later** due to dependencies on CGAL's Surface_mesh_simplification package.

### What This Means

**For Academic Research**:
- ✅ Free to use, modify, and distribute for research and education
- ✅ Perfect for reproducibility and citation
- ✅ Standard license for computational geometry research

**For Indie Game Developers**:
- ✅ Can use in **open-source games** (GPL-compatible)
- ✅ Can use **Open3D, meshoptimizer, fast-simplification** methods (MIT-licensed) without GPL restrictions
- ⚠️ Using CGAL component in **closed-source games** requires GPL compliance (source disclosure) or commercial CGAL license

**For Commercial Studios**:
- Three methods (Open3D, meshoptimizer, fast-simplification) are MIT-licensed and commercial-friendly
- CGAL component requires either: (1) GPL compliance with source disclosure, or (2) commercial license from GeometryFactory
- Or implement alternative without CGAL

### Component Licenses

| Component | Underlying Library License | Project License |
|-----------|---------------------------|-----------------|
| Open3D integration | MIT | GPL-3.0-or-later* |
| meshoptimizer integration | MIT | GPL-3.0-or-later* |
| fast-simplification integration | MIT | GPL-3.0-or-later* |
| cgal_simplify.cpp | GPL-3.0-or-later (CGAL) | GPL-3.0-or-later |

*Note: While the underlying libraries are permissively licensed, they're distributed as part of this GPL-licensed project.

See [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) for complete dependency information.

### Why GPL?

This project uses CGAL's `Surface_mesh_simplification` package, which is GPL-licensed. CGAL uses dual licensing:
- **Kernel/Foundation** (LGPL): Basic geometric types
- **Algorithms** (GPL): Surface processing, simplification, Boolean operations

Our mesh simplification implementation requires the GPL-licensed algorithms, making the combined work GPL-3.0-or-later.

## Contact & Support

**Author**: Lukas Gallo  
**Institution**: Breda University of Applied Sciences  
**Program**: Master of Game Technology  

**Issues**: Please open a GitHub issue for technical problems.

**Questions**: Contact via institutional email or thesis supervisor.

## Reproducibility Checklist

For researchers attempting to replicate this benchmark:

- [ ] Python 3.12 environment set up
- [ ] All dependencies installed from `requirements.txt`
- [ ] Relevant conda environment with all dependencies installed from `environments/`
- [ ] rtree installed via conda
- [ ] CGAL compiled (if testing CGAL method)
- [ ] Test meshes in OBJ format
- [ ] Run pilot test on 2-3 meshes first
- [ ] Verify `batch_report.json` format matches documentation
- [ ] Check heatmap visualizations render correctly

## Acknowledgments

- Test meshes: Stanford 3D Scanning Repository
- Simplification libraries: fast-simplification, Open3D, meshoptimizer, CGAL development teams
- Supervisor: Ruben Tack, Robbie Storm
- Institution: Breda University of Applied Sciences

---

**Version**: 2.0.0  
**Last Updated**: January 16, 2026  
**Thesis Status**: Block C - Data Collection Phase
