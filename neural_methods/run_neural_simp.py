"""
Batch processing script for Neural Mesh Simplification
Processes multiple meshes at multiple reduction levels with performance profiling

Script location: Root folder (e.g., O:\BUas Master\Scripting\)
Neural simplification: Root/neural-mesh-simplification/
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path

# Add neural-mesh-simplification to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
neural_simp_dir = os.path.join(script_dir, "neural-mesh-simplification")
sys.path.insert(0, os.path.join(neural_simp_dir, "src"))

import trimesh
from neural_mesh_simplification import NeuralMeshSimplifier
from neural_mesh_simplification.data.dataset import load_mesh


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_mesh_stats(mesh):
    """Get basic mesh statistics"""
    if isinstance(mesh, trimesh.Trimesh):
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges)
        }
    return None


def process_single_mesh(input_path, output_dir, simplifiers, target_percentages, algorithm_name="neural-simp"):
    """
    Process a single mesh at multiple reduction levels
    
    Args:
        input_path: Path to input mesh file
        output_dir: Directory to save simplified meshes
        simplifiers: List of NeuralMeshSimplifier instances (one per target ratio)
        target_percentages: List of target percentage reductions
        algorithm_name: Name to use in output filename
    """
    mesh_name = Path(input_path).stem
    print(f"\n{'='*80}")
    print(f"Processing: {mesh_name}")
    print(f"{'='*80}")
    
    # Load original mesh
    try:
        original_mesh = load_mesh(input_path)
    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        return
    
    # Handle Scene vs Trimesh
    if isinstance(original_mesh, trimesh.Scene):
        # For scenes, process the first geometry
        if len(original_mesh.geometry) > 0:
            original_mesh = list(original_mesh.geometry.values())[0]
        else:
            print(f"❌ Scene contains no geometry")
            return
    
    if not isinstance(original_mesh, trimesh.Trimesh):
        print(f"❌ Invalid mesh type: {type(original_mesh)}")
        return
    
    # Get original stats
    orig_stats = get_mesh_stats(original_mesh)
    print(f"Original mesh: {orig_stats['vertices']:,} vertices, {orig_stats['faces']:,} faces")
    
    # Process at each reduction level
    results = []
    for simplifier, target_pct in zip(simplifiers, target_percentages):
        print(f"\n  → Simplifying to {target_pct}% reduction...")
        
        start_time = time.time()
        try:
            simplified_mesh = simplifier.simplify(original_mesh)
            elapsed_time = time.time() - start_time
            
            # Get simplified stats
            simp_stats = get_mesh_stats(simplified_mesh)
            actual_reduction = 100 * (1 - simp_stats['vertices'] / orig_stats['vertices'])
            
            # Save simplified mesh
            output_filename = f"{mesh_name}_{algorithm_name}_{target_pct}percent.obj"
            output_path = os.path.join(output_dir, output_filename)
            simplified_mesh.export(output_path)
            
            # Store results
            results.append({
                'target': target_pct,
                'vertices': simp_stats['vertices'],
                'faces': simp_stats['faces'],
                'actual_reduction': actual_reduction,
                'time': elapsed_time
            })
            
            print(f"    ✓ Saved: {output_filename}")
            print(f"    ✓ Result: {simp_stats['vertices']:,} vertices ({actual_reduction:.1f}% reduction)")
            print(f"    ✓ Time: {elapsed_time:.2f}s")
            
        except Exception as e:
            print(f"    ❌ Error during simplification: {e}")
            continue
    
    # Print summary for this mesh
    if results:
        print(f"\n  Summary for {mesh_name}:")
        print(f"  {'Target':<10} {'Vertices':<12} {'Faces':<12} {'Reduction':<12} {'Time (s)':<10}")
        print(f"  {'-'*60}")
        for r in results:
            print(f"  {r['target']}%{'':<7} {r['vertices']:<12,} {r['faces']:<12,} {r['actual_reduction']:<11.1f}% {r['time']:<10.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process meshes with Neural Mesh Simplification"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input .obj mesh files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./simplified_meshes",
        help="Directory to save simplified meshes (default: ./simplified_meshes)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: neural-mesh-simplification/configs/default.yaml)"
    )
    parser.add_argument(
        "--target-percentages",
        type=int,
        nargs="+",
        default=[75, 50, 25],
        help="Target reduction percentages (default: 75 50 25)"
    )
    parser.add_argument(
        "--algorithm-name",
        type=str,
        default="neural-simp",
        help="Algorithm name for output filenames (default: neural-simp)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}")
    
    # Set default config path if not provided
    if args.config is None:
        args.config = os.path.join(neural_simp_dir, "configs", "default.yaml")
    
    # Load config
    try:
        config = load_config(args.config)
        print(f"✓ Loaded config: {args.config}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Convert percentages to ratios (e.g., 75% -> 0.75 remaining = 0.25 target_ratio)
    target_ratios = [(100 - pct) / 100 for pct in args.target_percentages]
    
    # Create simplifier instances for each target ratio
    print(f"\n✓ Creating {len(target_ratios)} simplifier instance(s)...")
    simplifiers = []
    for ratio, pct in zip(target_ratios, args.target_percentages):
        # Create a copy of config and update target_ratio
        model_config = config["model"].copy()
        model_config["target_ratio"] = ratio
        
        simplifier = NeuralMeshSimplifier(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            edge_hidden_dim=model_config["edge_hidden_dim"],
            num_layers=model_config["num_layers"],
            k=model_config["k"],
            edge_k=model_config["edge_k"],
            target_ratio=model_config["target_ratio"],
        )
        simplifiers.append(simplifier)
        print(f"  • {pct}% reduction (target_ratio={ratio:.2f})")
    
    # Find all .obj files
    mesh_files = sorted([
        f for f in os.listdir(args.input_dir) 
        if f.lower().endswith('.obj')
    ])
    
    if not mesh_files:
        print(f"\n❌ No .obj files found in {args.input_dir}")
        return
    
    print(f"\n✓ Found {len(mesh_files)} mesh file(s)")
    
    # Process each mesh
    total_start = time.time()
    for mesh_file in mesh_files:
        input_path = os.path.join(args.input_dir, mesh_file)
        process_single_mesh(
            input_path,
            args.output_dir,
            simplifiers,
            args.target_percentages,
            args.algorithm_name
        )
    
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"✓ All meshes processed in {total_time:.2f}s")
    print(f"✓ Simplified meshes saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()