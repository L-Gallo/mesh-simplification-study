# Test Meshes

Small public-domain models for pipeline verification and quick benchmarks.

## Files

**bunny.obj** -- Stanford Bunny (34,817 vertices, 69,451 faces)
Source: Stanford 3D Scanning Repository
http://graphics.stanford.edu/data/3Dscanrep/

**teapot.obj** -- Utah Teapot (3,241 vertices, 6,320 faces)
Source: Martin Newell, University of Utah, 1975

Both are OBJ format with vertex normals, validated as manifold.

## Full test suite

The thesis also benchmarks on Stanford Dragon, Armadillo, and Lucy for
scalability analysis. These are too large for the repository but can be
downloaded from the Stanford 3D Scanning Repository link above.

Production game assets (Bohemia Interactive, Arma Reforger) are under NDA
and not included. The pipeline works on any OBJ mesh.