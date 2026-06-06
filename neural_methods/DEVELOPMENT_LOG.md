# Neural Mesh Simplifier - Development Log

Investigation of Potamias et al. (2022) for the thesis benchmark.
30+ hours of optimization attempts on consumer hardware (RTX 3060, 6 GB VRAM).
All attempts failed due to O(n^2) memory scaling in the architecture.

Author: Lukas Gallo

## Summary

| Category | Changes | Lines | Effect |
|----------|---------|-------|--------|
| Configuration | 5 parameter adjustments | 6 | ~40% memory reduction |
| Model architecture | Sparse matrix refactor | 83 | ~25% memory reduction |
| Training pipeline | Memory management | 48 | ~5% improvement |
| **Total** | **3 files modified** | **137** | **Still 142x over budget** |


## 1. Configuration changes (configs/local.yaml)

**Reduced feature dimensions** (128 -> 32):
Memory dropped ~30-35%, but still needed ~1100 GB for a 42K-face mesh.

**Reduced batch size** (2 -> 1):
~10% memory reduction. Still ~990 GB required.

**Reduced epochs** (20 -> 2):
No memory impact, just faster iteration for testing.

**Disabled loss components** (lambda_e: 0.1, lambda_o: 0.0):
Negligible impact (<5%). The bottleneck is in the forward pass, not loss computation.


## 2. Model architecture changes (neural_mesh_simplification.py)

**Device movement fix** (line 42-43):
Added `data = data.to(self.device)` to prevent CPU/GPU transfer errors.
Fixed device mismatches but no memory savings.

**Sparse adjacency matrix** (major refactor, 83 lines):

The original creates a dense n x n adjacency matrix:
```python
# Original: O(n^2) memory
adj_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)
# 42,000^2 x 4 bytes = ~7 GB per matrix
```

Replaced with sparse representation:
```python
# Modified: O(edges) memory
adj_matrix = torch.sparse_coo_tensor(
    edge_indices, edge_values, (num_nodes, num_nodes), device=self.device
)
# ~126,000 x 4 bytes = ~0.5 MB
```

Result: ~25% memory reduction in this function, but downstream operations
(attention layers, triangle generation) still require dense intermediates.
Making one operation sparse does not fix O(n^2) scaling elsewhere.

**CPU-based triangle generation**:
Moved edge dictionary construction to CPU (32 GB RAM vs 6 GB VRAM).
Memory problem solved for this function, but 40-60x slower than GPU.
Estimated per-mesh time: 35-45 minutes (vs 11 ms for meshoptimizer).

| Approach | GPU memory | Speed | Outcome |
|----------|-----------|-------|---------|
| Original dense | ~7 GB/op | Fast | OOM |
| Sparse GPU | ~2 GB/op | Fast | Still OOM |
| CPU dictionary | ~500 MB (CPU) | 40x slower | Too slow |


## 3. Training pipeline changes (trainer.py)

**GPU memory cap** (line 220):
`torch.cuda.set_per_process_memory_fraction(0.6, 0)` -- crashes earlier
with less memory available. Does not reduce model requirements.

**Disabled validation** (lines 228-235):
Saves ~5% memory. Training still fails on the forward pass.

**Debug logging** (lines 278-306):
Added print statements throughout the training loop. Confirmed failure
point: `generate_candidate_triangles()` in the forward pass, before
backward pass. Rules out gradient accumulation as the issue.

**Aggressive cleanup** (lines 292-305):
`del batch; del output; del loss; torch.cuda.empty_cache()` after each
batch. Negligible impact since the OOM happens within a single forward pass.


## Test results

Hardware: RTX 3060 (6 GB VRAM), Ryzen 7 5800H, 32 GB RAM.
Test asset: character model, 42,000 faces, ~126,000 edges.

| Configuration | Est. memory required | Ratio to available |
|--------------|---------------------|-------------------|
| Original (unmodified) | ~1600 GB | 267x |
| After config changes | ~1100 GB | 184x |
| After sparse matrix | ~850 GB | 142x |
| CPU offload | ~3.5 GB (40x slower) | Fits, unusable speed |

8 test runs, 0 successful. Typical error:
```
RuntimeError: CUDA out of memory. Tried to allocate 142.33 GiB
(GPU 0: 5.80 GiB total capacity; 4.12 GiB already allocated)
```

Farthest progress: model loaded, mesh preprocessed, crashed on first
forward pass. Never reached backward pass or optimization.


## Files modified

```
neural-mesh-simplification/
  configs/local.yaml                             (6 lines)
  src/neural_mesh_simplification/models/
    neural_mesh_simplification.py                (83 lines)
  src/neural_mesh_simplification/trainer/
    trainer.py                                   (48 lines)
```

Full diff: `my_neural_modifications.patch` (apply with `git apply`).