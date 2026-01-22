# save as: prepare_mesh_for_meshcnn.py
import open3d as o3d
import numpy as np
from pathlib import Path

def prepare_mesh_for_meshcnn(input_path, output_path):
    """
    Convert any mesh to MeshCNN-compatible OBJ format.
    
    Requirements:
    - Manifold (each edge shared by exactly 2 faces)
    - Triangulated
    - Watertight
    - OBJ format
    """
    print(f"Loading mesh: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    # Step 1: Basic cleanup
    print("Removing duplicates...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    # Step 2: Check manifold status
    is_edge_manifold = mesh.is_edge_manifold()
    is_vertex_manifold = mesh.is_vertex_manifold()
    is_watertight = mesh.is_watertight()
    
    print(f"Edge manifold: {is_edge_manifold}")
    print(f"Vertex manifold: {is_vertex_manifold}")
    print(f"Watertight: {is_watertight}")
    
    # Step 3: Attempt repairs
    if not is_edge_manifold:
        print("WARNING: Mesh is not edge-manifold. Attempting repair...")
        mesh.remove_non_manifold_edges()
    
    # Step 4: Scale to reasonable size (MeshCNN works better with normalized meshes)
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    mesh.translate(-center)
    
    max_dist = np.linalg.norm(vertices - center, axis=1).max()
    mesh.scale(1.0 / max_dist, center=np.array([0, 0, 0]))
    
    # Step 5: Compute normals (required for MeshCNN features)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # Step 6: Save as OBJ
    print(f"Saving to: {output_path}")
    success = o3d.io.write_triangle_mesh(
        output_path, 
        mesh, 
        write_ascii=True,  # MeshCNN prefers ASCII
        write_vertex_normals=True,
        write_vertex_colors=False,
        write_triangle_uvs=False
    )
    
    if success:
        print(f"✓ Mesh prepared successfully!")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.triangles)}")
    else:
        print("✗ Failed to save mesh")
    
    return success

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python prepare_mesh_for_meshcnn.py input.ply output.obj")
        sys.exit(1)
    
    prepare_mesh_for_meshcnn(sys.argv[1], sys.argv[2])