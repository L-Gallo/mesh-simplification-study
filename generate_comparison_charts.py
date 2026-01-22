#!/usr/bin/env python3
"""
Mesh Simplification Comparison Chart Generator
===============================================
Generates statistical comparison charts from batch benchmark results.

Outputs:
- Hausdorff distance heatmap (methods x reduction levels)
- RMSE heatmap
- Performance comparison charts (time, memory)
- Success rate comparison

Author: Master's Thesis Visualization Tool
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not installed. Charts will use basic matplotlib.")


def load_batch_report(report_path: Path) -> Dict:
    """Load batch report JSON."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load report: {e}")
        sys.exit(1)


def extract_metric_matrix(
    report: Dict,
    metric_type: str,
    reduction_levels: List[str]
) -> np.ndarray:
    """
    Extract metric values into a matrix (methods x reductions).
    
    Args:
        report: Batch report dictionary
        metric_type: 'hausdorff', 'rmse', 'time', or 'memory'
        reduction_levels: List of reduction level strings (e.g., ['25%', '50%', '75%'])
    
    Returns:
        2D numpy array [methods, reductions]
    """
    method_stats = report['method_statistics']
    methods = sorted(method_stats.keys())
    
    matrix = np.zeros((len(methods), len(reduction_levels)))
    
    metric_map = {
        'hausdorff': 'mean_hausdorff',
        'rmse': 'mean_rmse',
        'time': 'mean_time_ms',
        'memory': 'mean_memory_mb'
    }
    
    for i, method in enumerate(methods):
        stats = method_stats[method]
        
        if metric_type in ['hausdorff', 'rmse']:
            # Single aggregate value for geometric accuracy
            value = stats.get(metric_map[metric_type], 0)
            matrix[i, :] = value
        elif metric_type in ['time', 'memory']:
            # Single aggregate value for performance
            value = stats.get(metric_map[metric_type], 0)
            matrix[i, :] = value
    
    return matrix, methods


def generate_geometric_accuracy_heatmap(
    report: Dict,
    output_dir: Path,
    reduction_levels: List[str]
) -> None:
    """Generate heatmaps for Hausdorff distance and RMSE."""
    print("\nGenerating geometric accuracy heatmaps...")
    
    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid")
    
    # Extract data
    hausdorff_matrix, methods = extract_metric_matrix(report, 'hausdorff', reduction_levels)
    rmse_matrix, _ = extract_metric_matrix(report, 'rmse', reduction_levels)
    
    # Check if we have data
    if np.all(hausdorff_matrix == 0):
        print("  [WARNING] No geometric accuracy data found in report")
        return
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hausdorff Distance Heatmap
    ax1 = axes[0]
    if SEABORN_AVAILABLE:
        sns.heatmap(
            hausdorff_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',  # Red=bad, Green=good
            xticklabels=reduction_levels,
            yticklabels=methods,
            cbar_kws={'label': 'Hausdorff Distance (%)'},
            ax=ax1
        )
    else:
        im1 = ax1.imshow(hausdorff_matrix, cmap='RdYlGn_r', aspect='auto')
        ax1.set_xticks(range(len(reduction_levels)))
        ax1.set_xticklabels(reduction_levels)
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(methods)
        plt.colorbar(im1, ax=ax1, label='Hausdorff Distance (%)')
        
        # Add annotations
        for i in range(len(methods)):
            for j in range(len(reduction_levels)):
                text = ax1.text(j, i, f'{hausdorff_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black")
    
    ax1.set_title('Mean Hausdorff Distance\n(lower is better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Reduction Level', fontsize=10)
    ax1.set_ylabel('Method', fontsize=10)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    
    # RMSE Heatmap
    ax2 = axes[1]
    if SEABORN_AVAILABLE:
        sns.heatmap(
            rmse_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            xticklabels=reduction_levels,
            yticklabels=methods,
            cbar_kws={'label': 'RMSE (%)'},
            ax=ax2
        )
    else:
        im2 = ax2.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')
        ax2.set_xticks(range(len(reduction_levels)))
        ax2.set_xticklabels(reduction_levels)
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(methods)
        plt.colorbar(im2, ax=ax2, label='RMSE (%)')
        
        for i in range(len(methods)):
            for j in range(len(reduction_levels)):
                text = ax2.text(j, i, f'{rmse_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black")
    
    ax2.set_title('Mean Root Mean Squared Error\n(lower is better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Reduction Level', fontsize=10)
    ax2.set_ylabel('Method', fontsize=10)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'geometric_accuracy_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


def generate_performance_comparison(
    report: Dict,
    output_dir: Path
) -> None:
    """Generate performance comparison charts (time and memory)."""
    print("\nGenerating performance comparison charts...")
    
    method_stats = report['method_statistics']
    methods = sorted(method_stats.keys())
    
    # Extract data
    times = []
    time_stds = []
    memories = []
    memory_stds = []
    
    for method in methods:
        stats = method_stats[method]
        times.append(stats['mean_time_ms'])
        time_stds.append(stats['std_time_ms'])
        memories.append(stats['mean_memory_mb'])
        memory_stds.append(stats['std_memory_mb'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Execution Time
    ax1 = axes[0]
    bars1 = ax1.bar(methods, times, yerr=time_stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_ylabel('Mean Execution Time (ms)', fontsize=11)
    ax1.set_title('Mean Execution Time\n(lower is better)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15, labelsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time, std in zip(bars1, times, time_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{time:.1f}ms',
                ha='center', va='bottom', fontsize=8)
    
    # Memory Usage
    ax2 = axes[1]
    bars2 = ax2.bar(methods, memories, yerr=memory_stds, capsize=5, alpha=0.7, color='coral')
    ax2.set_ylabel('Mean Peak Memory (MB)', fontsize=11)
    ax2.set_title('Mean Peak Memory Usage\n(lower is better)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15, labelsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mem, std in zip(bars2, memories, memory_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mem:.1f}MB',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


def generate_success_rate_chart(
    report: Dict,
    output_dir: Path
) -> None:
    """Generate success rate and stability comparison chart."""
    print("\nGenerating success rate comparison...")
    
    method_stats = report['method_statistics']
    methods = sorted(method_stats.keys())
    
    # Extract data
    success_rates = []
    stability_rates = []
    
    for method in methods:
        stats = method_stats[method]
        success_rates.append(stats['success_rate'])
        stability_rates.append(stats['stability_rate'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate', 
                   alpha=0.8, color='limegreen')
    bars2 = ax.bar(x + width/2, stability_rates, width, label='Stability Rate',
                   alpha=0.8, color='gold')
    
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('Success Rate and Stability Comparison\n(higher is better)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = output_dir / 'success_rate_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


def generate_failure_breakdown(
    report: Dict,
    output_dir: Path
) -> None:
    """Generate stacked bar chart showing failure types per method."""
    print("\nGenerating failure breakdown chart...")
    
    method_stats = report['method_statistics']
    methods = sorted(method_stats.keys())
    
    # Extract data
    crashes = []
    timeouts = []
    invalid_geo = []
    
    for method in methods:
        stats = method_stats[method]
        crashes.append(stats['crashes'])
        timeouts.append(stats['timeouts'])
        invalid_geo.append(stats['invalid_geometry'])
    
    # Check if there are any failures
    if sum(crashes + timeouts + invalid_geo) == 0:
        print("  [OK] No failures to visualize (all tests successful)")
        return
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.6
    
    p1 = ax.bar(x, crashes, width, label='Crashes', color='firebrick', alpha=0.8)
    p2 = ax.bar(x, timeouts, width, bottom=crashes, label='Timeouts', 
                color='darkorange', alpha=0.8)
    p3 = ax.bar(x, invalid_geo, width, 
                bottom=np.array(crashes) + np.array(timeouts),
                label='Invalid Geometry', color='gold', alpha=0.8)
    
    ax.set_ylabel('Number of Failures', fontsize=11)
    ax.set_title('Failure Type Breakdown by Method\n(lower is better)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'failure_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


def generate_combined_overview(
    report: Dict,
    output_dir: Path
) -> None:
    """Generate comprehensive overview with multiple metrics."""
    print("\nGenerating combined overview chart...")
    
    method_stats = report['method_statistics']
    methods = sorted(method_stats.keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # 1. Success & Stability (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    success_rates = [method_stats[m]['success_rate'] for m in methods]
    stability_rates = [method_stats[m]['stability_rate'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    ax1.bar(x - width/2, success_rates, width, label='Success', color='limegreen', alpha=0.7)
    ax1.bar(x + width/2, stability_rates, width, label='Stability', color='gold', alpha=0.7)
    ax1.set_ylabel('Rate (%)', fontsize=10)
    ax1.set_title('Success & Stability Rates', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Execution Time (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    times = [method_stats[m]['mean_time_ms'] for m in methods]
    ax2.bar(methods, times, color='steelblue', alpha=0.7)
    ax2.set_ylabel('Time (ms)', fontsize=10)
    ax2.set_title('Mean Execution Time', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15, labelsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Memory Usage (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    memories = [method_stats[m]['mean_memory_mb'] for m in methods]
    ax3.bar(methods, memories, color='coral', alpha=0.7)
    ax3.set_ylabel('Memory (MB)', fontsize=10)
    ax3.set_title('Mean Peak Memory Usage', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15, labelsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Geometric Accuracy (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    hausdorffs = [method_stats[m]['mean_hausdorff'] for m in methods]
    rmses = [method_stats[m]['mean_rmse'] for m in methods]
    
    if any(h > 0 for h in hausdorffs):
        x = np.arange(len(methods))
        width = 0.35
        ax4.bar(x - width/2, hausdorffs, width, label='Hausdorff', color='purple', alpha=0.7)
        ax4.bar(x + width/2, rmses, width, label='RMSE', color='teal', alpha=0.7)
        ax4.set_ylabel('Error (%)', fontsize=10)
        ax4.set_title('Mean Geometric Accuracy', fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
        ax4.legend(fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No geometric accuracy data available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Mean Geometric Accuracy', fontsize=11, fontweight='bold')
    
    # Overall title with proper spacing
    fig.suptitle('Mesh Simplification Methods - Comprehensive Comparison', 
                 fontsize=14, fontweight='bold', y=0.99)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = output_dir / 'combined_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


def print_report_summary(report: Dict) -> None:
    """Print text summary of the report."""
    print("\n" + "="*80)
    print("BATCH REPORT SUMMARY")
    print("="*80)
    
    summary = report['summary']
    print(f"\nTotal Assets: {summary['total_assets']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Execution Time: {summary['total_execution_time_seconds']:.1f}s ({summary['total_execution_time_seconds']/60:.1f} minutes)")
    
    print(f"\nOverall Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"Overall Stability Rate: {summary['overall_stability_rate']:.1f}%")
    print(f"Total Failures: {summary['total_failures']}")
    print(f"Total Unstable: {summary['total_unstable']}")
    
    print("\nMethod Statistics:")
    print("-"*80)
    print(f"{'Method':<25} {'Success':<12} {'Stability':<12} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-"*80)
    
    method_stats = report['method_statistics']
    for method, stats in sorted(method_stats.items()):
        print(
            f"{method:<25} "
            f"{stats['success_rate']:>6.1f}%{'':<5} "
            f"{stats['stability_rate']:>6.1f}%{'':<5} "
            f"{stats['mean_time_ms']:>8.1f}{'':<4} "
            f"{stats['mean_memory_mb']:>8.1f}"
        )
    
    print("\n" + "="*80)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from batch benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all charts
  python generate_comparison_charts.py -i ./results/batch_report.json -o ./figures

  # Generate specific charts only
  python generate_comparison_charts.py -i ./results/batch_report.json -o ./figures \\
      --charts accuracy performance success

  # Specify custom reduction levels for heatmaps
  python generate_comparison_charts.py -i ./results/batch_report.json -o ./figures \\
      --reduction-levels 90% 75% 50% 25%
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to batch_report.json"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./figures",
        help="Output directory for charts (default: ./figures)"
    )
    
    parser.add_argument(
        "--charts",
        nargs="+",
        choices=["accuracy", "performance", "success", "failures", "overview", "all"],
        default=["all"],
        help="Which charts to generate (default: all)"
    )
    
    parser.add_argument(
        "--reduction-levels",
        nargs="+",
        default=["25%", "50%", "75%"],
        help="Reduction levels for heatmaps (default: 25%% 50%% 75%%)"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print text summary"
    )
    
    args = parser.parse_args()
    
    # Load report
    report_path = Path(args.input)
    if not report_path.exists():
        print(f"ERROR: Report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading report: {report_path}")
    report = load_batch_report(report_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Print summary
    if not args.no_summary:
        print_report_summary(report)
    
    # Determine which charts to generate
    charts = args.charts
    if "all" in charts:
        charts = ["accuracy", "performance", "success", "failures", "overview"]
    
    print("\n" + "="*80)
    print("GENERATING CHARTS")
    print("="*80)
    
    try:
        if "accuracy" in charts:
            generate_geometric_accuracy_heatmap(report, output_dir, args.reduction_levels)
        
        if "performance" in charts:
            generate_performance_comparison(report, output_dir)
        
        if "success" in charts:
            generate_success_rate_chart(report, output_dir)
        
        if "failures" in charts:
            generate_failure_breakdown(report, output_dir)
        
        if "overview" in charts:
            generate_combined_overview(report, output_dir)
        
        print("\n" + "="*80)
        print("[SUCCESS] Chart generation complete!")
        print(f"Charts saved to: {output_dir.absolute()}")
        print("="*80)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()