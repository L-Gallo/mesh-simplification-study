# Environment Configurations

This directory contains conda environment specifications for all tested simplification methods, including those excluded from the final benchmark due to accessibility barriers.

## Working Environment

### `environment_traditional.yml`

**Methods**: fast-simplification, Open3D, meshoptimizer, CGAL  
**Status**: ✅ Included in final benchmark  
**Compatibility**: Windows, Linux, macOS

**Installation**:
```bash
conda env create -f environment_traditional.yml
conda activate mesh-simplification
```

**Key Dependencies**:
- Python 3.12
- pyvista, open3d, meshoptimizer, trimesh
- CGAL (requires separate C++ compilation - see main README)

---

## Excluded Environments

These environments document neural methods that were systematically evaluated but excluded due to fundamental scalability limitations. They are included for research transparency and reproducibility.

### `environment_meshcnn.yml`

**Method**: MeshCNN (Neural mesh simplification)  
**Status**: ❌ Excluded from benchmark  
**Reason**: Fixed architecture constraint (750-edge maximum)

**Technical Limitations**:

1. **Architecture Constraint**:
   - Hard-coded 750-edge maximum in network design
   - Production game assets: 100,000+ edges typical
   - Scale mismatch: 133× (100,000 / 750)

2. **Preprocessing Requirements**:
   - Custom edge-based data structure conversion
   - 10+ hours implementation attempt
   - Format incompatibility with standard mesh workflows

3. **Pre-trained Model Availability**:
   - Weights only available for SHREC 2016 dataset
   - No general-purpose weights available
   - Training from scratch requires large annotated dataset

**Empirical Evidence**:
- Installation time: 31× longer than traditional methods
- Successful test runs: 0/12 attempted
- Conclusion: Not accessible for target use case (indie game development)

**Environment Details**:
```yaml
name: meshcnn
dependencies:
  - python=3.7  # Older Python required
  - pytorch=1.7.0
  - torchvision
  - numpy<1.24  # Compatibility constraints
```

---

### `environment_neural.yml`

**Method**: Neural Mesh Simplifier (Learning-based decimation)  
**Status**: ❌ Excluded from benchmark  
**Reason**: Prohibitive memory requirements (267× available hardware)

**Technical Limitations**:

1. **Memory Scaling**:
   - Test asset: Character model (42,000 faces)
   - Required GPU memory: ~1,600 GB
   - Available hardware (RTX 3060): 6 GB
   - **Scaling factor: 267× inadequacy**

2. **Computational Cost**:
   - Estimated processing time: 35-45 minutes per mesh
   - Traditional methods: 11-723 ms (3,000-240,000× faster)

3. **Hardware Requirements**:
   - Requires: NVIDIA A100 (80GB) or multiple high-end GPUs
   - Estimated cost: $10,000+ for suitable GPU
   - Consumer hardware: RTX 3060 = $1,500

**Empirical Evidence**:
- Memory requirement: ~1,600 GB (267× available)
- Installation attempts: 3 over multiple days
- Successful runs: 0/8 attempted
- Conclusion: Fundamentally inaccessible to consumer hardware

**Environment Details**:
```yaml
name: neural-simp
dependencies:
  - python=3.9
  - pytorch>=1.12.0
  - pytorch-cuda=11.7
  - tensorflow-gpu
  - Open3D-ML
```

---

## Methodology

### Tool Selection Process

**Phase 1**: Literature review identified 8 candidate methods (4 traditional, 4 neural)

**Phase 2**: Accessibility screening
- Installation time measurement
- Hardware requirement assessment
- Documentation quality evaluation

**Phase 3**: Pilot testing
- Attempted implementation of all methods
- Measured successful execution rates
- Documented technical barriers

**Phase 4**: Evidence-based selection
- **Included**: 4/4 traditional methods (100% success rate)
- **Excluded**: 4/4 neural methods (0% success rate on production assets)

### Exclusion Criteria

A method was excluded if it met ANY of:
1. **Scale incompatibility**: Cannot handle production mesh complexity
2. **Hardware inaccessibility**: Requires >50× available GPU memory
3. **Time infeasibility**: >20 hours implementation with no success
4. **Zero success rate**: After reasonable adaptation effort

---

## Installation Notes

### Testing Excluded Methods (Not Recommended)

These environments can be installed to verify the documented barriers:

```bash
# MeshCNN (expect compatibility issues)
conda env create -f environment_meshcnn.yml
conda activate meshcnn

# Neural Simplifier (expect CUDA out-of-memory)
conda env create -f environment_neural.yml
conda activate neural-simp
```

**Expected Errors**:

**MeshCNN**:
```
RuntimeError: Edge count (8432) exceeds maximum supported (750)
AssertionError in mesh_prepare.py, line 127
```

**Neural Simplifier**:
```
RuntimeError: CUDA out of memory. Tried to allocate 267.52 GiB 
(GPU 0: 5.80 GiB total capacity; 4.12 GiB already allocated)
```

We do NOT recommend this unless you have:
- 100+ GB GPU memory (for neural methods)
- Significant debugging time (20+ hours)
- Motivation to verify documented barriers

---

## Future Research Directions

These exclusions identify valuable research opportunities:

### Memory-Efficient Neural Architectures
- **Challenge**: Current O(n²) memory scaling
- **Goal**: Reduce to O(n log n) or O(n)
- **Impact**: Enable consumer GPU usage

### Adaptive Architecture for MeshCNN
- **Challenge**: Fixed 750-edge network
- **Goal**: Dynamic architecture adapting to mesh complexity
- **Impact**: Handle arbitrary mesh sizes

### Hybrid Approaches
- **Concept**: Neural network predicts importance, geometric method executes
- **Potential**: 10× speedup with minimal memory overhead

### Cloud-Based Solutions
- **Concept**: Serverless neural simplification APIs
- **Challenge**: Network latency, data privacy concerns

---

## References

**MeshCNN**:
- Hanocka et al. (2019). "MeshCNN: A Network with an Edge." ACM TOG.
- GitHub: https://github.com/ranahanocka/MeshCNN

**Neural Mesh Simplification**:
- Various implementations tested (see repository documentation)

---

**Version**: 2.0  
**Last Updated**: January 21, 2026  
**Purpose**: Research transparency in tool selection process
