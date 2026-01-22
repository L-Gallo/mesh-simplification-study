#!/usr/bin/env python3
"""
Mesh Error Heatmap Generator
=============================
Generates colored 3D meshes showing per-vertex geometric error distribution.
Useful for visual comparison of simplification methods.

Outputs:
- Colored .ply mesh with vertex colors (blue=low error, red=high error)
- Colorbar legend as .png image

Author: Master's Thesis Visualization Tool
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial import cKDTree
from typing import Optional

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh not installed. Install with: pip install trimesh")
    sys.exit(1)


def generate_error_heatmap(
    original_path: Path,
    simplified_path: Path,
    output_path: Path,
    colormap: str = 'coolwarm',
    percentile: float = 95.0,
    show_stats: bool = True,
    fixed_max_error: Optional[float] = None
) -> None:
    """
    Generate colored mesh showing per-vertex error distribution.
    
    Args:
        original_path: Path to original mesh
        simplified_path: Path to simplified mesh
        output_path: Path to save colored mesh (.ply)
        colormap: Matplotlib colormap name (default: 'coolwarm')
        percentile: Percentile for error normalization (default: 95.0)
        show_stats: Whether to print error statistics
        fixed_max_error: If provided, use this as max error for color scaling (enables comparison)
    """
    print(f"Loading meshes...")
    print(f"  Original: {original_path}")
    print(f"  Simplified: {simplified_path}")
    
    # Load meshes
    try:
        mesh_orig = trimesh.load(str(original_path), force='mesh')
        mesh_simp = trimesh.load(str(simplified_path), force='mesh')
    except Exception as e:
        print(f"ERROR: Failed to load meshes: {e}")
        sys.exit(1)
    
    # Handle Scene objects
    if isinstance(mesh_orig, trimesh.Scene):
        mesh_orig = list(mesh_orig.geometry.values())[0]
    if isinstance(mesh_simp, trimesh.Scene):
        mesh_simp = list(mesh_simp.geometry.values())[0]
    
    print(f"  Original: {len(mesh_orig.vertices):,} vertices, {len(mesh_orig.faces):,} faces")
    print(f"  Simplified: {len(mesh_simp.vertices):,} vertices, {len(mesh_simp.faces):,} faces")
    
    # Diagnostic: Check if simplified vertices are subset of original
    # This would explain zero distances
    print(f"\nDiagnostic checks:")
    orig_vertex_set = set(map(tuple, mesh_orig.vertices.round(6)))
    simp_vertex_set = set(map(tuple, mesh_simp.vertices.round(6)))
    subset_count = len(simp_vertex_set.intersection(orig_vertex_set))
    print(f"  Simplified vertices that match original: {subset_count}/{len(mesh_simp.vertices)} ({subset_count/len(mesh_simp.vertices)*100:.1f}%)")
    
    # Compute per-vertex distances
    print(f"\nComputing per-vertex errors...")
    
    # Method 1: Simplified vertices -> Original mesh
    query_orig = trimesh.proximity.ProximityQuery(mesh_orig)
    distances_simp_to_orig = np.abs(query_orig.signed_distance(mesh_simp.vertices))
    print(f"  Simplified->Original: min={distances_simp_to_orig.min():.6f}, max={distances_simp_to_orig.max():.6f}, mean={distances_simp_to_orig.mean():.6f}")
    
    # Method 2: Sample original surface -> Simplified mesh (this captures loss of detail)
    query_simp = trimesh.proximity.ProximityQuery(mesh_simp)
    # Sample points from original mesh surface
    sample_points, _ = trimesh.sample.sample_surface(mesh_orig, 10000)
    distances_orig_to_simp = np.abs(query_simp.signed_distance(sample_points))
    print(f"  Original->Simplified: min={distances_orig_to_simp.min():.6f}, max={distances_orig_to_simp.max():.6f}, mean={distances_orig_to_simp.mean():.6f}")
    
    # Use the max of both directions (two-way Hausdorff concept)
    # But for visualization, we color simplified mesh vertices, so interpolate the sampled errors
    if distances_orig_to_simp.max() > distances_simp_to_orig.max():
        print(f"  Using Original->Simplified distances (captures surface detail loss)")
        # Map sampled point errors back to simplified mesh vertices via nearest neighbor
        tree = cKDTree(sample_points)
        _, indices = tree.query(mesh_simp.vertices, k=3)  # 3 nearest sampled points
        # Average error from 3 nearest sampled points
        distances = np.mean(distances_orig_to_simp[indices], axis=1)
    else:
        print(f"  Using Simplified->Original distances")
        distances = distances_simp_to_orig
    
    # Debug: Check if distances are computed
    print(f"  Distance computation: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}")
    
    # Compute bounding box diagonal for normalization
    bb_diagonal = np.linalg.norm(mesh_orig.bounds[1] - mesh_orig.bounds[0])
    print(f"  Bounding box diagonal: {bb_diagonal:.6f}")
    
    distances_normalized = (distances / bb_diagonal) * 100  # Percentage
    
    # Statistics
    mean_error = np.mean(distances_normalized)
    std_error = np.std(distances_normalized)
    max_error = np.max(distances_normalized)
    median_error = np.median(distances_normalized)
    
    if show_stats:
        print(f"\nError Statistics (% of bounding box diagonal):")
        print(f"  Mean:   {mean_error:.4f}%")
        print(f"  Median: {median_error:.4f}%")
        print(f"  Std:    {std_error:.4f}%")
        print(f"  Max:    {max_error:.4f}%")
        print(f"  Min:    {np.min(distances_normalized):.4f}%")
    
    # Handle edge case: all errors are zero or extremely small
    if max_error < 1e-6:
        print(f"\n[WARNING] Errors are extremely small (max={max_error:.8f}%)")
        print(f"[WARNING] This usually means:")
        print(f"  - Meshes are identical (no simplification occurred)")
        print(f"  - Coordinate system mismatch")
        print(f"  - Mesh loading issue")
        print(f"[WARNING] Generating heatmap with uniform coloring...")
        colors = np.tile([0.0, 0.0, 1.0], (len(mesh_simp.vertices), 1))  # All blue
        percentile_value = fixed_max_error if fixed_max_error else 0.0
    else:
        # Determine scale for color mapping
        if fixed_max_error is not None:
            # Use provided global scale for unified comparison
            percentile_value = fixed_max_error
            print(f"\n[UNIFIED SCALE] Using fixed maximum: {percentile_value:.4f}%")
            print(f"  This mesh's actual max: {max_error:.4f}%")
        else:
            # Use percentile from this mesh only
            percentile_value = np.percentile(distances_normalized, percentile)
            
            # Avoid division by zero
            if percentile_value < 1e-6:
                print(f"\n[WARNING] Percentile value too small ({percentile_value:.8f}%), using max instead")
                percentile_value = max_error if max_error > 1e-6 else 1.0
            
            print(f"\nColor mapping:")
            print(f"  Using {percentile}th percentile: {percentile_value:.4f}%")
        
        normalized = np.clip(distances_normalized / percentile_value, 0, 1)
        
        print(f"  Colormap: {colormap}")
        print(f"  Blue (low) = 0.000% error")
        print(f"  Red (high) = {percentile_value:.4f}% error")
        
        # Apply colormap
        try:
            colormap_func = plt.get_cmap(colormap)
        except:
            colormap_func = cm.get_cmap(colormap)
        colors = colormap_func(normalized)[:, :3]  # RGB only, drop alpha
    
    # Assign colors to mesh and save
    mesh_simp.visual.vertex_colors = (colors * 255).astype(np.uint8)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save colored mesh
    print(f"\nSaving colored mesh to: {output_path}")
    try:
        mesh_simp.export(output_path)
        print(f"[OK] Mesh saved successfully")
    except Exception as e:
        print(f"ERROR: Failed to save mesh: {e}")
        sys.exit(1)
    
    # Generate colorbar legend
    colorbar_path = output_path.with_suffix('.png')
    print(f"Saving colorbar legend to: {colorbar_path}")
    
    # Ensure directory exists
    colorbar_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract asset and method info from filename
    # Example: bunny_fast-_90pct_run1_heatmap.ply -> bunny, fast-simplification, 90%
    filename_parts = output_path.stem.replace('_heatmap', '').split('_')
    asset_name = filename_parts[0] if filename_parts else "Unknown"
    method_name = filename_parts[1] if len(filename_parts) > 1 else "Unknown"
    reduction = filename_parts[2] if len(filename_parts) > 2 else "Unknown"
    
    # Create wider figure to accommodate stats
    fig = plt.figure(figsize=(6, 6))
    
    # Add title with asset and method info
    fig.suptitle(f'{asset_name.capitalize()} - {method_name} - {reduction}', 
                 fontsize=12, fontweight='bold', y=0.95)
    
    # Colorbar on left side
    ax_cbar = fig.add_axes([0.1, 0.3, 0.15, 0.5])  # [left, bottom, width, height]
    
    # Create colorbar
    norm = plt.Normalize(vmin=0, vmax=percentile_value)
    try:
        cmap_obj = plt.get_cmap(colormap)
    except:
        cmap_obj = cm.get_cmap(colormap)
    
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Geometric Error\n(% of bounding box diagonal)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    # Add statistics on right side with better formatting
    scale_info = "[UNIFIED SCALE]" if fixed_max_error is not None else "[Individual Scale]"
    
    stats_text = (
        f"{scale_info}\n\n"
        f"Error Statistics:\n\n"
        f"Mean:   {mean_error:.4f}%\n"
        f"Median: {median_error:.4f}%\n"
        f"Std:    {std_error:.4f}%\n"
        f"Max:    {max_error:.4f}%\n"
        f"Min:    {np.min(distances_normalized):.4f}%\n\n"
        f"Color Scale:\n"
        f"Blue  = 0.000%\n"
        f"Red   = {percentile_value:.4f}%+"  # Added "+" here!
    )
    if fixed_max_error is None:
        stats_text += f"\n({percentile}th percentile)"
    
    fig.text(0.45, 0.5, stats_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(colorbar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Colorbar saved successfully")
    print(f"\n{'='*60}")
    print(f"Heatmap generation complete!")
    print(f"{'='*60}")


def batch_generate_heatmaps(
    results_dir: Path,
    original_meshes_dir: Path,
    output_dir: Path,
    pattern: str = "*_run1.obj",
    colormap: str = 'coolwarm',
    unified_scale: bool = True,
    manual_max_error: Optional[float] = None,
    use_percentile_max: bool = False
) -> None:
    """
    Generate heatmaps for multiple simplified meshes.
    
    Args:
        results_dir: Directory containing simplified meshes
        original_meshes_dir: Directory containing original meshes
        output_dir: Directory to save heatmaps
        pattern: Glob pattern for simplified meshes (default: *_run1.obj)
        colormap: Colormap to use
        unified_scale: If True, use same color scale across all meshes (recommended for comparison)
        manual_max_error: If provided, use this value as max (overrides auto-computation)
        use_percentile_max: If True, use 99th percentile instead of absolute max
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    simplified_files = list(results_dir.rglob(pattern))
    
    if not simplified_files:
        print(f"ERROR: No files matching '{pattern}' found in {results_dir}")
        sys.exit(1)
    
    print(f"Found {len(simplified_files)} simplified meshes to process")
    
    # FIRST PASS: Compute global max error if unified_scale is True
    global_max_error = 0.0
    error_records = []  # Track which file produces max error
    all_errors = []  # Collect all individual errors for percentile
    
    # If manual max provided, use it directly
    if manual_max_error is not None:
        global_max_error = manual_max_error
        print(f"\n[MANUAL SCALE] Using manually specified maximum: {global_max_error:.4f}%")
        print(f"  All heatmaps will use scale: 0.000% (blue) to {global_max_error:.4f}% (red)")
    elif unified_scale:
        print(f"\n[UNIFIED SCALE MODE] Computing global error range across all meshes...")
        
        for simp_file in simplified_files:
            # Extract asset name
            parts = simp_file.stem.split('_')
            asset_name = parts[0]
            original_file = original_meshes_dir / f"{asset_name}.obj"
            
            if not original_file.exists():
                continue
            
            try:
                # Quick error computation
                mesh_orig = trimesh.load(str(original_file), force='mesh')
                mesh_simp = trimesh.load(str(simp_file), force='mesh')
                
                if isinstance(mesh_orig, trimesh.Scene):
                    mesh_orig = list(mesh_orig.geometry.values())[0]
                if isinstance(mesh_simp, trimesh.Scene):
                    mesh_simp = list(mesh_simp.geometry.values())[0]
                
                # Compute distances both ways
                query_orig = trimesh.proximity.ProximityQuery(mesh_orig)
                distances_1 = np.abs(query_orig.signed_distance(mesh_simp.vertices))
                
                query_simp = trimesh.proximity.ProximityQuery(mesh_simp)
                sample_points, _ = trimesh.sample.sample_surface(mesh_orig, 10000)
                distances_2 = np.abs(query_simp.signed_distance(sample_points))
                
                # Normalize
                bb_diagonal = np.linalg.norm(mesh_orig.bounds[1] - mesh_orig.bounds[0])
                max_dist_1 = (distances_1.max() / bb_diagonal) * 100
                max_dist_2 = (distances_2.max() / bb_diagonal) * 100
                
                # Collect all errors if using percentile mode
                if use_percentile_max:
                    errors_1_norm = (distances_1 / bb_diagonal) * 100
                    errors_2_norm = (distances_2 / bb_diagonal) * 100
                    all_errors.extend(errors_1_norm.tolist())
                    all_errors.extend(errors_2_norm.tolist())
                
                mesh_max = max(max_dist_1, max_dist_2)
                
                # Track for debugging
                error_records.append({
                    'file': simp_file.name,
                    'max_error': mesh_max,
                    'max_dist_1': max_dist_1,
                    'max_dist_2': max_dist_2
                })
                
                global_max_error = max(global_max_error, mesh_max)
                
            except Exception as e:
                print(f"  Warning: Could not compute errors for {simp_file.name}: {e}")
                continue
        
        # Use percentile instead of max if requested
        if use_percentile_max and all_errors:
            percentile_value = np.percentile(all_errors, 99.0)
            print(f"\n  Using 99th percentile instead of absolute max:")
            print(f"    Absolute max: {global_max_error:.4f}%")
            print(f"    99th percentile: {percentile_value:.4f}%")
            global_max_error = percentile_value
        
        # Sort and show top 5 error contributors
        error_records.sort(key=lambda x: x['max_error'], reverse=True)
        print(f"\n  Top 5 highest-error meshes:")
        for i, record in enumerate(error_records[:5], 1):
            print(f"    {i}. {record['file']}: {record['max_error']:.4f}% (simp->orig: {record['max_dist_1']:.4f}%, orig->simp: {record['max_dist_2']:.4f}%)")
        
        print(f"\n  Global maximum error: {global_max_error:.4f}%")
        print(f"  All heatmaps will use scale: 0.000% (blue) to {global_max_error:.4f}% (red)")
    
    print(f"\n{'='*60}\n")
    
    # SECOND PASS: Generate heatmaps with unified scale
    for idx, simp_file in enumerate(simplified_files, 1):
        print(f"\n[{idx}/{len(simplified_files)}] Processing: {simp_file.name}")
        print("-" * 60)
        
        # Extract asset name from filename
        parts = simp_file.stem.split('_')
        asset_name = parts[0]
        
        # Find original mesh
        original_file = original_meshes_dir / f"{asset_name}.obj"
        
        if not original_file.exists():
            print(f"[WARNING] Original mesh not found: {original_file}")
            print(f"  Skipping...")
            continue
        
        # Generate output filename
        output_filename = f"{simp_file.stem}_heatmap.ply"
        output_path = output_dir / output_filename
        
        # Generate heatmap with unified scale
        try:
            if unified_scale:
                generate_error_heatmap(
                    original_path=original_file,
                    simplified_path=simp_file,
                    output_path=output_path,
                    colormap=colormap,
                    show_stats=True,
                    fixed_max_error=global_max_error  # Pass global scale
                )
            else:
                generate_error_heatmap(
                    original_path=original_file,
                    simplified_path=simp_file,
                    output_path=output_path,
                    colormap=colormap,
                    show_stats=True
                )
        except Exception as e:
            print(f"[ERROR] Failed to generate heatmap: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Heatmaps saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate error heatmaps for simplified meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate heatmap for single mesh
  python generate_heatmaps.py \\
      --original ./meshes/bunny.obj \\
      --simplified ./results/bunny/fast-simplification/bunny_fs_75pct_run1.obj \\
      --output ./figures/bunny_fs_heatmap.ply

  # Batch generate heatmaps for all run1 meshes (unified scale for comparison)
  python generate_heatmaps.py \\
      --batch \\
      --results-dir ./results \\
      --original-dir ./meshes \\
      --output-dir ./figures/heatmaps

  # Batch with individual scales per mesh
  python generate_heatmaps.py \\
      --batch \\
      --results-dir ./results \\
      --original-dir ./meshes \\
      --output-dir ./figures/heatmaps \\
      --no-unified-scale

  # Use different colormap
  python generate_heatmaps.py \\
      --original ./meshes/bunny.obj \\
      --simplified ./results/bunny_simplified.obj \\
      --output ./bunny_heatmap.ply \\
      --colormap viridis

Available colormaps: coolwarm (default), viridis, plasma, inferno, jet, RdYlGn

IMPORTANT: For comparing methods, always use --unified-scale (default in batch mode).
This ensures all meshes use the same color scale, making visual comparison valid.
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process multiple meshes"
    )
    mode_group.add_argument(
        "--original",
        type=str,
        help="Single mode: path to original mesh"
    )
    
    # Single mode arguments
    parser.add_argument(
        "--simplified",
        type=str,
        help="Single mode: path to simplified mesh"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Single mode: output path for colored mesh (.ply)"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Batch mode: directory containing simplified meshes"
    )
    
    parser.add_argument(
        "--original-dir",
        type=str,
        help="Batch mode: directory containing original meshes"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Batch mode: directory to save heatmaps"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_run1.obj",
        help="Batch mode: glob pattern for simplified meshes (default: *_run1.obj)"
    )
    
    parser.add_argument(
        "--unified-scale",
        action="store_true",
        default=True,
        help="Batch mode: use same color scale across all meshes for comparison (default: enabled)"
    )
    
    parser.add_argument(
        "--no-unified-scale",
        action="store_false",
        dest="unified_scale",
        help="Batch mode: use individual scale for each mesh"
    )
    
    parser.add_argument(
        "--max-error",
        type=float,
        default=None,
        help="Batch mode: manually set maximum error for color scale (e.g., --max-error 1.0 for 0-1%% range)"
    )
    
    parser.add_argument(
        "--use-percentile-max",
        action="store_true",
        help="Batch mode: use 99th percentile of all errors as max (reduces outlier impact)"
    )
    
    # Common arguments
    parser.add_argument(
        "--colormap",
        type=str,
        default="coolwarm",
        choices=["coolwarm", "viridis", "plasma", "inferno", "jet", "RdYlGn", "RdYlGn_r"],
        help="Colormap for error visualization (default: coolwarm)"
    )
    
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile for error normalization (default: 95.0)"
    )
    
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't print error statistics"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch mode
            if not args.results_dir or not args.original_dir or not args.output_dir:
                parser.error("Batch mode requires --results-dir, --original-dir, and --output-dir")
            
            batch_generate_heatmaps(
                results_dir=Path(args.results_dir),
                original_meshes_dir=Path(args.original_dir),
                output_dir=Path(args.output_dir),
                pattern=args.pattern,
                colormap=args.colormap,
                unified_scale=args.unified_scale,
                manual_max_error=args.max_error,
                use_percentile_max=args.use_percentile_max
            )
        else:
            # Single mode
            if not args.simplified or not args.output:
                parser.error("Single mode requires --original, --simplified, and --output")
            
            generate_error_heatmap(
                original_path=Path(args.original),
                simplified_path=Path(args.simplified),
                output_path=Path(args.output),
                colormap=args.colormap,
                percentile=args.percentile,
                show_stats=not args.no_stats
            )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()