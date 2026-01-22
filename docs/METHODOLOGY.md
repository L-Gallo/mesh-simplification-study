# Experimental Methodology

This document describes the experimental protocol for the mesh simplification benchmark suite to enable exact replication.

## Hardware & Software Configuration

### Test System Specifications

**CPU**: AMD Ryzen 7 5800H
- Cores: 8 physical / 16 threads
- Base Clock: 3.2 GHz
- Boost Clock: 4.4 GHz

**GPU**: NVIDIA GeForce RTX 3060 (Laptop)
- VRAM: 6GB GDDR6
- CUDA Cores: 3840

**RAM**: 32GB DDR4-3200 MHz

**Storage**: NVMe SSD (read/write speeds do not affect benchmark)

**Operating System**: Windows 11 Pro (Build [specify exact build])

### Software Versions

**Python Environment**:
- Python: 3.12.0
- Conda: [version]

**Core Libraries** (from `requirements.txt`):
- pyvista: 0.43.0
- fast-simplification: 0.1.0
- open3d: 0.18.0
- meshoptimizer: 0.20.0
- trimesh: 4.0.0
- numpy: 1.26.0
- rtree: 1.2.0

**CGAL** (optional):
- Version: 5.6
- Compiler: MSVC 2022 / GCC 11+ (Linux)
- CMake: 3.20+

### Tool Selection Protocol

**Phase 1: Literature Review**
- Identified 8 candidate methods (4 traditional, 4 neural)
- Traditional: fast-simplification, Open3D, meshoptimizer, CGAL
- Neural: MeshCNN, Neural Mesh Simplifier, others

**Phase 2: Accessibility Screening**
- Installation time measurement
- Hardware requirement assessment
- Documentation quality evaluation
- Successful execution on test meshes

**Phase 3: Pilot Testing**
- Attempted implementation of all 8 methods
- Measured successful execution rates
- Documented barriers encountered
- **Result**: 4/4 traditional methods succeeded, 0/4 neural methods succeeded

**Phase 4: Evidence-Based Selection**
- **Included**: 4 traditional methods (100% success rate on pilot assets)
- **Excluded**: 4 neural methods (0% success rate on production-scale assets)

**Exclusion Criteria**:
A method was excluded if it met ANY of:
1. Scale incompatibility (cannot handle production mesh complexity)
2. Hardware inaccessibility (requires >50× available GPU memory)
3. Time infeasibility (>20 hours implementation with no success)
4. Zero success rate after reasonable adaptation effort

**Documentation**: See `environments/README.md` for detailed exclusion rationale with empirical evidence (267× memory requirements, 750-edge architectural limits, 26-hour implementation attempts).

---

## Test Protocol

### Phase 1: Pilot Testing

**Purpose**: Validate methodology stability and establish baseline metrics.

**Test Assets**:
1. Stanford Bunny (69,451 faces)
2. Utah Teapot (6,320 faces)

**Rationale**: Public domain meshes with known geometry for reproducibility.

**Test Configuration**:
- Methods: All 4 (fast-simplification, Open3D, meshoptimizer, CGAL)
- Reduction Levels: 90%, 80%, 50%
- Repetitions: 3 per configuration
- Total Tests: 2 assets × 4 methods × 3 levels × 3 reps = 72 tests

**Execution**:
```bash
python mesh_simplifier_batch.py \
    -i ./pilot_meshes \
    -o ./pilot_results \
    --methods all \
    --reduction-levels 90 80 50 \
    --repetitions 3 \
    --compute-accuracy
```

### Phase 2: Production Testing

**Test Assets**: 
- [Number] game production assets from Bohemia Interactive
- Provided under NDA (not included in public repository)
- Range: [min-max] triangle count
- Variety: Characters, environment props, vehicles

**Test Configuration**:
- Methods: All 4
- Reduction Levels: 90%, 75%, 50%, 25%
- Repetitions: 3 per configuration

### Test Execution Protocol

1. **Pre-test Validation**:
   - Verify mesh is manifold (no holes, non-manifold edges)
   - Confirm OBJ format with valid vertex normals
   - Check file integrity (not corrupted)

2. **Test Execution**:
   - Single-threaded execution (no parallel processing)
   - Close all non-essential applications
   - System idle for 2 minutes before test start
   - No user interaction during test
   - Room temperature: 20-25°C (thermal stability)

3. **Per-Test Procedure**:
   ```python
   # 1. Load original mesh
   original_mesh = load_mesh(input_path)
   
   # 2. Record start time and memory
   start_time = time.perf_counter()
   start_memory = process.memory_info().rss
   
   # 3. Simplify mesh
   simplified_mesh = method.simplify(original_mesh, target_reduction)
   
   # 4. Record end time and memory
   end_time = time.perf_counter()
   end_memory = process.memory_info().rss
   
   # 5. Save simplified mesh
   save_mesh(simplified_mesh, output_path)
   
   # 6. Compute geometric accuracy (Run 1 only)
   if run == 1:
       hausdorff = compute_hausdorff_distance(original_mesh, simplified_mesh)
       rmse = compute_rmse(original_mesh, simplified_mesh)
   
   # 7. Log results
   log_result(test_id, execution_time, memory_delta, hausdorff, rmse)
   ```

4. **Error Handling**:
   - Catch all exceptions
   - Log error type and message
   - Mark test as failed
   - Continue to next test (don't abort batch)

5. **Post-test Cleanup**:
   - Garbage collection between tests
   - 1-second cooldown period
   - Memory validation (check for leaks)

## Metrics

### Performance Metrics

**Execution Time** (milliseconds):
- Measured using `time.perf_counter()` (nanosecond precision)
- Excludes file I/O operations
- Includes only simplification algorithm execution

**Memory Usage** (megabytes):
- Peak resident set size during simplification
- Measured using `psutil.Process().memory_info().rss`
- Delta from baseline (memory before - memory after)

### Geometric Accuracy Metrics

**Hausdorff Distance** (percentage):
- One-directional: Original → Simplified
- Computed on first repetition only (deterministic)
- Implementation: `trimesh.proximity.longest_ray()`
- Normalized by bounding box diagonal
- Reported as percentage of model size

**RMSE (Root Mean Square Error)** (percentage):
- Per-vertex distance averaged across all vertices
- Computed on first repetition only
- Implementation: `trimesh.proximity.closest_point()`
- Normalized by bounding box diagonal
- Reported as percentage of model size

### Stability Metrics

**Coefficient of Variation (CV)**:
- Formula: CV = (std / mean) × 100
- Computed separately for execution time and memory
- Threshold: CV > 15% flagged as unstable
- Based on 3 repetitions

**Success Rate**:
- Percentage of tests that complete without errors
- Computed per method, per reduction level
- Formula: (successful_tests / total_tests) × 100

**Stability Rate**:
- Percentage of successful tests with CV ≤ 15%
- Formula: (stable_tests / successful_tests) × 100

## Data Collection

### Raw Data Format

All results stored in `batch_report.json` with structure:

```json
{
  "system": {...},
  "configuration": {...},
  "assets": {
    "asset_name": {
      "original_metrics": {
        "vertices": int,
        "faces": int,
        "edges": int
      },
      "methods": {
        "method_name": {
          "reduction_level": {
            "repetitions": [
              {
                "run": int,
                "time_ms": float,
                "memory_mb": float,
                "hausdorff": float (run 1 only),
                "rmse": float (run 1 only),
                "success": bool
              }
            ],
            "statistics": {
              "mean_time_ms": float,
              "std_time_ms": float,
              "cv_time": float,
              "stable": bool
            }
          }
        }
      }
    }
  }
}
```

### Statistical Analysis

**Aggregation**:
- Mean and standard deviation computed across all 3 repetitions
- Geometric accuracy uses run 1 value only (no averaging)
- Success rate aggregated across all assets and levels

**Outlier Detection**:
- Z-score > 2.0 flagged for review
- CV > 15% flagged as unstable
- Manual inspection of flagged tests

**Comparison Method**:
- Paired comparisons (same asset, same reduction level)
- Non-parametric tests (no normality assumption)
- Statistical significance: p < 0.05

## Reproducibility Requirements

To replicate this benchmark:

1. **Hardware**: Similar performance tier (mid-range gaming laptop, ~2022-2024)
2. **Software**: Exact library versions from `requirements.txt`
3. **Assets**: Public domain test meshes (Stanford, Utah) for validation
4. **Protocol**: Follow test execution protocol exactly
5. **Environment**: Isolated conda environment, no background processes

**Expected Variability**:
- Execution time: ±10% due to hardware differences
- Memory usage: ±15% due to OS/Python overhead
- Geometric accuracy: ±0.01% (deterministic algorithms)

**Not Expected to Match Exactly**:
- Absolute execution times (hardware-dependent)
- Absolute memory usage (OS-dependent)

**Expected to Match Exactly**:
- Success/failure patterns for each method
- Geometric accuracy values (deterministic)
- Relative performance rankings between methods

## Limitations & Threats to Validity

### Internal Validity
- Limited test assets in pilot (n=2)
- Small repetition count (n=3) for stability
- Single hardware configuration

### External Validity
- Pilot meshes are academic models, not game assets
- Production assets under NDA (not publicly available)
- Results may not generalize to other mesh types

### Construct Validity
- Geometric metrics may not fully capture perceptual quality
- Execution time affected by implementation quality, not just algorithm
- Memory usage includes Python overhead

### Reliability
- 3 repetitions may not capture full stability range
- Tests run sequentially (no control for system state changes)
- No cross-platform validation

## Future Work

- Cross-platform testing (Linux, macOS)
- Larger repetition counts (n=10+)
- Hardware sensitivity analysis (different GPUs, CPUs)
- Perceptual quality validation with human participants
- Comparison with commercial tools (if licenses available)

---

**Document Version**: 1.0  
**Last Updated**: January 16, 2026  
**Author**: Bludimir  
**Status**: Block B - Active Data Collection
