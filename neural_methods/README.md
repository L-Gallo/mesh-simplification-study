# Neural Methods Investigation

This directory documents ~40 hours of attempts to implement neural mesh
simplification methods for the thesis benchmark. Both methods were excluded
after failing to run on production-scale meshes with consumer hardware.

MeshCNN (Hanocka et al., 2019) has a hard-coded 750-edge architecture
limit. Production game assets have 100K+ edges.

Neural Mesh Simplifier (Potamias et al., 2022) requires ~850-1600 GB GPU
memory for a 42K-face mesh. An RTX 3060 has 6 GB. After 30+ hours of
optimization (sparse matrices, reduced feature dimensions, CPU offloading),
the requirement dropped to ~850 GB -- still 142x over budget.

The exclusion is documented as an empirical finding in the thesis, not a
convenience decision. Independent confirmation from Martin Normark, who
encountered the same gradient flow blockage in the face classifier.

## Files

```
DEVELOPMENT_LOG.md              30+ hours of optimization attempts on
                                Neural Mesh Simplifier, with memory
                                measurements at each stage

my_neural_modifications.patch   137 lines changed across 3 files
                                (configs/local.yaml, neural_mesh_simplification.py,
                                trainer.py). Apply with: git apply

run_neural_simp.py              Wrapper script for Neural Mesh Simplifier
run_meshcnn.py                  Wrapper script for MeshCNN
prepare_mesh_for_meshcnn.py     Preprocessing for MeshCNN format
```

## Reproducing

```bash
# Clone the original repository
git clone https://github.com/martinnormark/neural-mesh-simplification/
cd neural-mesh-simplification

# Apply modifications
git apply ../my_neural_modifications.patch

# Setup environment
conda env create -f ../environments/environment_neural.yml
conda activate neural-simp

# Run (expect CUDA out-of-memory on consumer GPUs)
python train.py --config configs/local.yaml
```