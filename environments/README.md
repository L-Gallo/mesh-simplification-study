# Environment Configurations

## environment_traditional.yml

The working environment for all four benchmark methods.

```bash
conda env create -f environment_traditional.yml
conda activate mesh-simplification
```

Includes Python 3.12, PyVista, Open3D, meshoptimizer, trimesh, and analysis
dependencies. CGAL requires a separate C++ build (see main README).

## Neural method environments (excluded)

`environment_meshcnn.yml` and `environment_neural.yml` are included for
research transparency. Both neural methods were excluded from the final
benchmark after extensive testing revealed fundamental scalability barriers.

Installing these is not recommended unless you want to independently verify
the exclusion. Expect compatibility issues (older Python/PyTorch versions)
and out-of-memory errors on consumer hardware.

See `neural_methods/DEVELOPMENT_LOG.md` for the full investigation with
empirical evidence.