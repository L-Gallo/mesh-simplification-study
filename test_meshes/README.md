# Test Meshes

This directory contains public domain 3D models used for methodology validation.

## Included Assets

### Stanford Bunny
- **File**: `bunny.obj`
- **Source**: Stanford 3D Scanning Repository
- **Vertices**: ~35,000 (original: 69,451 faces after conversion)
- **License**: Public Domain
- **Citation**: 
  ```
  Stanford Computer Graphics Laboratory
  Stanford Bunny
  Available at: http://graphics.stanford.edu/data/3Dscanrep/
  ```

### Utah Teapot
- **File**: `teapot.obj`
- **Source**: University of Utah
- **Vertices**: ~3,000 (original: 6,320 faces)
- **License**: Public Domain
- **Citation**:
  ```
  Martin Newell (1975)
  Utah Teapot
  University of Utah Computer Science Department
  ```

## Usage

These meshes are used for:
1. **Pilot testing**: Validate methodology before production testing
2. **Reproducibility**: Other researchers can replicate results
3. **Benchmarking**: Standard test cases for comparison

## Download Original Sources

If you need higher-resolution versions or other formats:

**Stanford Repository**:
- URL: http://graphics.stanford.edu/data/3Dscanrep/
- Other models: Bunny, Dragon, Happy Buddha, Armadillo

**Utah Teapot**:
- URL: https://en.wikipedia.org/wiki/Utah_teapot
- Available in many graphics packages by default

## Mesh Properties

| Model | Format | Vertices | Faces | Manifold | Has Normals | File Size |
|-------|--------|----------|-------|----------|-------------|-----------|
| Bunny | OBJ | 34,817 | 69,451 | Yes | Yes | ~3.2 MB |
| Teapot | OBJ | 3,241 | 6,320 | Yes | Yes | ~290 KB |

## Processing Notes

All meshes in this directory have been:
- Converted to OBJ format (CGAL compatibility)
- Validated for manifold geometry
- Cleaned of non-manifold edges
- Normalized to unit bounding box (optional)

## Adding Your Own Test Meshes

To add custom test meshes:

1. **Format**: Must be `.obj` files
2. **Quality**: 
   - Manifold geometry (no holes, no non-manifold edges)
   - Valid vertex normals
   - Triangulated faces
3. **Size**: <10MB per file recommended
4. **License**: Ensure you have rights to redistribute

**Validation**:
```bash
# Check mesh with Blender
blender --background --python - <<EOF
import bpy
bpy.ops.import_scene.obj(filepath="your_mesh.obj")
mesh = bpy.context.active_object.data
print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces: {len(mesh.polygons)}")
EOF
```

## Production Assets (Not Included)

Production game assets from Bohemia Interactive are used in the full thesis research but are **not included** in this repository due to NDA restrictions.

Properties of production dataset:
- Count: [X] meshes
- Range: [min]-[max] triangle count
- Types: Characters, props, vehicles, environments
- Source: Bohemia Interactive (under NDA)

## Attribution Requirements

If you use these meshes in your research:

**For Stanford Bunny**:
```
The Stanford Bunny model is courtesy of the Stanford Computer Graphics Laboratory.
```

**For Utah Teapot**:
```
The Utah Teapot was created by Martin Newell at the University of Utah in 1975.
```

## File Format Details

All OBJ files include:
- Vertex positions (`v x y z`)
- Vertex normals (`vn x y z`)
- Face indices (`f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3`)

**Note**: Texture coordinates are optional and may not be present.

---

**Directory Version**: 1.0  
**Last Updated**: January 16, 2026  
**Maintainer**: Lukas Gallo
