#!/usr/bin/env python3
"""
Mesh Simplification Benchmark - Data Analysis

Generates descriptive statistics, boxplots, and anomaly reports
from batch_report.json.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# CONFIG
INPUT_JSON = "batch_report.json"
OUTPUT_DIR = "analysis_output"
FIG_DPI = 300
FIG_FORMAT = "png"  # png for thesis/word, svg also available

# Colorblind-friendly palette (Wong, 2011 - Nature Methods)
# Order: fast-simplification, Open3D, meshoptimizer, CGAL
METHOD_COLORS = {
    "fast-simplification": "#E69F00",  # orange
    "open3d":              "#56B4E9",  # sky blue
    "meshoptimizer":       "#009E73",  # bluish green
    "cgal":                "#CC79A7",  # reddish purple
}
METHOD_ORDER = ["fast-simplification", "open3d", "meshoptimizer", "cgal"]
METHOD_LABELS = {
    "fast-simplification": "Fast-Simplification",
    "open3d": "Open3D",
    "meshoptimizer": "Meshoptimizer",
    "cgal": "CGAL",
}
REDUCTION_ORDER = ["50%", "80%", "90%"]

# STYLE SETUP
def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': FIG_DPI,
        'savefig.dpi': FIG_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

# DATA LOADING & FLATTENING
def load_and_flatten(json_path):
    """Load batch_report.json and flatten into a pandas DataFrame."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for asset_name, asset_data in data['assets'].items():
        for method_name, reductions in asset_data['methods'].items():
            for reduction_level, red_data in reductions.items():
                for rep in red_data['repetitions']:
                    row = {
                        'asset': asset_name,
                        'method': method_name,
                        'reduction': reduction_level,
                        'run': rep['run_number'],
                        'time_ms': rep['performance']['execution_time_ms'],
                        'memory_mb': rep['performance']['memory_delta_mb'],
                        'peak_memory_mb': rep['performance']['peak_memory_mb'],
                        'input_vertices': rep['input_metrics']['vertex_count'],
                        'input_faces': rep['input_metrics']['face_count'],
                        'output_vertices': rep['output_metrics']['vertex_count'],
                        'output_faces': rep['output_metrics']['face_count'],
                        'target_reduction': rep['target_reduction_ratio'],
                        'actual_reduction': rep['actual_reduction_ratio'],
                        'success': rep['success'],
                        'instability_flag': rep.get('instability_flag', False),
                    }
                    # Geometric accuracy only on run 1
                    if rep.get('geometric_accuracy'):
                        ga = rep['geometric_accuracy']
                        row['hausdorff_pct'] = ga['hausdorff_distance_normalized']
                        row['rmse_pct'] = ga['rmse_normalized']
                        row['hausdorff_raw'] = ga['hausdorff_distance_raw']
                        row['rmse_raw'] = ga['rmse_raw']
                    else:
                        row['hausdorff_pct'] = np.nan
                        row['rmse_pct'] = np.nan
                        row['hausdorff_raw'] = np.nan
                        row['rmse_raw'] = np.nan
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    # Ensure categorical ordering
    df['method'] = pd.Categorical(df['method'], categories=METHOD_ORDER, ordered=True)
    df['reduction'] = pd.Categorical(df['reduction'], categories=REDUCTION_ORDER, ordered=True)
    
    return df, data.get('system_info', {})


# DESCRIPTIVE STATISTICS
def compute_method_summary(df):
    """Per-method aggregated descriptive statistics."""
    # Use all 3 runs for time/memory, only run 1 for geometric accuracy
    time_mem = df.groupby('method', observed=True).agg(
        time_mean=('time_ms', 'mean'),
        time_median=('time_ms', 'median'),
        time_std=('time_ms', 'std'),
        time_iqr=('time_ms', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        time_min=('time_ms', 'min'),
        time_max=('time_ms', 'max'),
        mem_mean=('memory_mb', 'mean'),
        mem_median=('memory_mb', 'median'),
        mem_std=('memory_mb', 'std'),
        mem_min=('memory_mb', 'min'),
        mem_max=('memory_mb', 'max'),
        n_total=('time_ms', 'count'),
        n_unstable=('instability_flag', 'sum'),
    ).round(2)
    
    # Geometric accuracy (run 1 only)
    geo = df.dropna(subset=['hausdorff_pct']).groupby('method', observed=True).agg(
        hausdorff_mean=('hausdorff_pct', 'mean'),
        hausdorff_median=('hausdorff_pct', 'median'),
        hausdorff_std=('hausdorff_pct', 'std'),
        hausdorff_min=('hausdorff_pct', 'min'),
        hausdorff_max=('hausdorff_pct', 'max'),
        rmse_mean=('rmse_pct', 'mean'),
        rmse_median=('rmse_pct', 'median'),
        rmse_std=('rmse_pct', 'std'),
        rmse_min=('rmse_pct', 'min'),
        rmse_max=('rmse_pct', 'max'),
    ).round(4)
    
    return pd.concat([time_mem, geo], axis=1)


def compute_reduction_summary(df):
    """Per-method x per-reduction-level statistics."""
    # All runs for time
    time_stats = df.groupby(['method', 'reduction'], observed=True).agg(
        time_mean=('time_ms', 'mean'),
        time_median=('time_ms', 'median'),
        time_std=('time_ms', 'std'),
        mem_mean=('memory_mb', 'mean'),
        mem_std=('memory_mb', 'std'),
    ).round(2)
    
    # Run 1 for geometry
    geo_stats = df.dropna(subset=['hausdorff_pct']).groupby(
        ['method', 'reduction'], observed=True
    ).agg(
        hausdorff_mean=('hausdorff_pct', 'mean'),
        hausdorff_std=('hausdorff_pct', 'std'),
        rmse_mean=('rmse_pct', 'mean'),
        rmse_std=('rmse_pct', 'std'),
    ).round(4)
    
    return pd.concat([time_stats, geo_stats], axis=1)


def compute_per_asset_breakdown(df):
    """Per-asset x method x reduction breakdown (run 1 metrics)."""
    run1 = df[df['run'] == 1].copy()
    breakdown = run1.groupby(['asset', 'method', 'reduction'], observed=True).agg(
        time_ms=('time_ms', 'first'),
        memory_mb=('memory_mb', 'first'),
        hausdorff_pct=('hausdorff_pct', 'first'),
        rmse_pct=('rmse_pct', 'first'),
        actual_reduction=('actual_reduction', 'first'),
        target_reduction=('target_reduction', 'first'),
        input_faces=('input_faces', 'first'),
        output_faces=('output_faces', 'first'),
    ).round(4)
    return breakdown


# ANOMALY DETECTION & REPORTING
def generate_anomaly_report(df):
    """Identify and document data anomalies for the thesis."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ANOMALY & MEASUREMENT CONSIDERATIONS REPORT")
    report_lines.append("=" * 70)
    
    # 1. Simplification ceiling detection
    report_lines.append("\n1. SIMPLIFICATION CEILING DETECTION")
    report_lines.append("-" * 40)
    run1 = df[df['run'] == 1]
    threshold = 0.02  # >2% deviation from target
    ceiling_cases = run1[
        abs(run1['actual_reduction'] - run1['target_reduction']) > threshold
    ]
    if len(ceiling_cases) > 0:
        report_lines.append(f"Found {len(ceiling_cases)} cases where actual reduction deviates >2% from target:\n")
        for _, row in ceiling_cases.iterrows():
            report_lines.append(
                f"  {row['asset']:12s} | {row['method']:22s} | "
                f"Target: {row['target_reduction']*100:.0f}% | "
                f"Actual: {row['actual_reduction']*100:.1f}% | "
                f"Deviation: {(row['actual_reduction'] - row['target_reduction'])*100:+.1f}pp"
            )
    else:
        report_lines.append("No significant simplification ceiling detected.")
    
    # 2. Memory measurement uniformity (CGAL suspicion)
    report_lines.append("\n\n2. MEMORY MEASUREMENT UNIFORMITY")
    report_lines.append("-" * 40)
    for method in METHOD_ORDER:
        method_data = df[df['method'] == method]
        mem_by_config = method_data.groupby(['asset', 'reduction'], observed=True)['memory_mb'].agg(['mean', 'std', 'min', 'max'])
        zero_variance = (mem_by_config['std'] == 0).sum()
        total_configs = len(mem_by_config)
        report_lines.append(
            f"  {METHOD_LABELS[method]:22s}: {zero_variance}/{total_configs} configurations "
            f"with zero memory variance across runs"
        )
    
    report_lines.append("\n  NOTE: CGAL executes as an external subprocess. Memory measurements")
    report_lines.append("  reflect Python-side overhead (subprocess communication, result reading)")
    report_lines.append("  rather than CGAL's internal memory consumption. This makes CGAL's")
    report_lines.append("  memory figures not directly comparable to the other three methods.")
    
    # 3. Instability flag summary
    report_lines.append("\n\n3. INSTABILITY FLAG SUMMARY")
    report_lines.append("-" * 40)
    for method in METHOD_ORDER:
        method_data = df[df['method'] == method]
        n_unstable = method_data['instability_flag'].sum()
        n_total = len(method_data)
        stability = (1 - n_unstable / n_total) * 100
        report_lines.append(
            f"  {METHOD_LABELS[method]:22s}: {n_unstable:3d}/{n_total} unstable "
            f"(stability: {stability:.1f}%)"
        )
    report_lines.append("\n  NOTE: Instability flags are based on coefficient of variation (CV)")
    report_lines.append("  across 3 repetitions. High memory CV for fast-executing methods")
    report_lines.append("  (fast-simplification, meshoptimizer) likely reflects RSS polling")
    report_lines.append("  granularity rather than genuine behavioral variance. Methods with")
    report_lines.append("  longer execution times (CGAL) provide more polling samples,")
    report_lines.append("  yielding more consistent measurements.")
    
    # 4. Timing variance analysis
    report_lines.append("\n\n4. TIMING CONSISTENCY (CV per configuration)")
    report_lines.append("-" * 40)
    time_cv = df.groupby(['method', 'asset', 'reduction'], observed=True)['time_ms'].agg(
        lambda x: (x.std() / x.mean() * 100) if x.mean() > 0 else 0
    )
    for method in METHOD_ORDER:
        method_cvs = time_cv.loc[method]
        report_lines.append(
            f"  {METHOD_LABELS[method]:22s}: "
            f"mean CV = {method_cvs.mean():.1f}%, "
            f"max CV = {method_cvs.max():.1f}%, "
            f"configs with CV>10%: {(method_cvs > 10).sum()}/{len(method_cvs)}"
        )
    
    return "\n".join(report_lines)


# VISUALIZATIONS
def _get_palette():
    """Return method color palette for seaborn."""
    return [METHOD_COLORS[m] for m in METHOD_ORDER]


def _get_hue_palette():
    """Return palette dict keyed by method name for hue-based coloring."""
    return {m: METHOD_COLORS[m] for m in METHOD_ORDER}


def _label_methods(ax, axis='x'):
    """Replace method names with display labels on axis."""
    labels = [METHOD_LABELS.get(m, m) for m in METHOD_ORDER]
    if axis == 'x':
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels(labels)


def _annotate_medians(ax, data, group_col, value_col, methods_to_label, fmt=".1f"):
    """Add median value labels centered on the median line.
    
    methods_to_label: list of method names to annotate (uses METHOD_ORDER for positioning).
    """
    for i, method in enumerate(METHOD_ORDER):
        if method not in methods_to_label:
            continue
        values = data[data[group_col] == method][value_col].dropna()
        if len(values) == 0:
            continue
        median = values.median()
        ax.annotate(
            f'{median:{fmt}}',
            xy=(i, median), xytext=(0, 0),
            textcoords='offset points', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='#cccccc', alpha=0.9)
        )


def _log_format(val, _):
    """Smart formatter for log axis: adapts decimal places to magnitude."""
    if val == 0:
        return '0'
    if val >= 1:
        return f'{val:,.0f}'
    # Show enough decimals so the value isn't "0.00"
    # e.g. 0.5 -> "0.50", 0.05 -> "0.05", 0.005 -> "0.005"
    if val >= 0.01:
        return f'{val:.2f}'
    if val >= 0.001:
        return f'{val:.3f}'
    return f'{val:.4f}'


def _compute_zoom_limit(data, group_col, value_col, order, min_span_fraction=0.25):
    """Compute y-limit so the smallest VISIBLE boxplot (whisker-to-whisker) occupies min_span_fraction of chart.
    
    Ignores outliers. Iterates to find stable visible/clipped sets.
    """
    method_stats = []
    for method in order:
        values = data[data[group_col] == method][value_col].dropna()
        if len(values) == 0:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        lower_fence = q1 - 1.5 * iqr
        within_upper = values[values <= upper_fence]
        within_lower = values[values >= lower_fence]
        whisker_top = within_upper.max() if len(within_upper) > 0 else q3
        whisker_bottom = within_lower.min() if len(within_lower) > 0 else q1
        whisker_span = whisker_top - whisker_bottom
        method_stats.append({
            'method': method, 'q1': q1, 'q3': q3, 'iqr': iqr,
            'whisker_top': whisker_top, 'whisker_bottom': whisker_bottom,
            'whisker_span': whisker_span, 'median': values.median()
        })
    
    if len(method_stats) < 2:
        return None, []
    
    # Iterate to find stable limit based on visible methods
    visible = list(method_stats)
    limit = None
    for _ in range(5):  # converges in 1-2 iterations
        # Smallest whisker span among currently visible methods
        positive_spans = [s['whisker_span'] for s in visible if s['whisker_span'] > 0]
        if not positive_spans:
            return None, []
        smallest_span = min(positive_spans)
        
        # Set limit so this span = min_span_fraction of chart height
        limit = smallest_span / min_span_fraction
        
        # Check who would be clipped at this limit
        new_visible = [s for s in method_stats if s['whisker_top'] <= limit]
        
        # Need at least 1 visible method
        if len(new_visible) < 1:
            positive_spans.sort()
            if len(positive_spans) > 1:
                limit = positive_spans[1] / min_span_fraction
                new_visible = [s for s in method_stats if s['whisker_top'] <= limit]
            break
        
        # Stable?
        if set(s['method'] for s in new_visible) == set(s['method'] for s in visible):
            break
        visible = new_visible
    
    if limit is None:
        return None, []
    
    # Don't zoom if everything fits
    max_whisker = max(s['whisker_top'] for s in method_stats)
    if max_whisker <= limit:
        return None, []
    
    clipped = [(s['method'], s['median']) for s in method_stats if s['whisker_top'] > limit]
    
    if not clipped:
        return None, []
    
    return limit, clipped


def _boxplot(ax, data, x, y, order, zoom=False, zoom_pct=25, show_labels=True, median_fmt=".1f"):
    """Standardized boxplot with hue-based coloring.
    
    Args:
        zoom: If True, use zoomed linear scale (clips high-value methods).
        zoom_pct: Minimum % of chart height the smallest visible boxplot should occupy.
        show_labels: If True, annotate medians with value labels.
    """
    # Always show outliers; zoom mode clips the view which hides out-of-range ones naturally
    sns.boxplot(
        data=data, x=x, y=y, hue=x, order=order,
        palette=_get_hue_palette(), width=0.6, fliersize=4,
        boxprops=dict(alpha=0.85), ax=ax, legend=False
    )
    _label_methods(ax)
    
    if zoom:
        # Floor at zero if all data is non-negative
        min_val = data[y].min()
        if min_val >= 0:
            ax.set_ylim(bottom=0)
        
        ylim, clipped = _compute_zoom_limit(data, x, y, order, 
                                             min_span_fraction=zoom_pct / 100.0)
        if ylim is not None:
            ax.set_ylim(top=ylim)
            if show_labels:
                # Yellow labels for clipped methods
                for method, median in clipped:
                    idx = order.index(method)
                    ax.annotate(
                        f'{median:{median_fmt}} \u2191',
                        xy=(idx, ylim * 0.96),
                        ha='center', va='top',
                        fontsize=8, fontweight='bold', color='#333333',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3cd',
                                  edgecolor='#ffc107', alpha=0.95)
                    )
            # Normal labels for non-clipped methods
            if show_labels:
                non_clipped = [m for m in order if m not in [c[0] for c in clipped]]
                _annotate_medians(ax, data, x, y, non_clipped, fmt=median_fmt)
        else:
            # No zoom needed
            if show_labels:
                _annotate_medians(ax, data, x, y, order, fmt=median_fmt)
    else:
        # Standard linear scale
        if show_labels:
            _annotate_medians(ax, data, x, y, order, fmt=median_fmt)


def plot_time_by_method(df, output_dir, zoom=False, zoom_pct=25, show_labels=True):
    """Boxplot: Processing time per method (all runs, all assets, all reductions)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _boxplot(ax, df, 'method', 'time_ms', METHOD_ORDER, zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".1f")
    ax.set_ylabel("Processing Time (ms)")
    ax.set_xlabel("")
    ax.set_title("Processing Time Distribution by Method")
    fig.savefig(os.path.join(output_dir, "boxplot_time_by_method.png"))
    plt.close(fig)


def plot_time_by_method_reduction(df, output_dir, zoom=False, zoom_pct=25, show_labels=True):
    """Boxplot: Processing time per method x reduction level."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, red in enumerate(REDUCTION_ORDER):
        subset = df[df['reduction'] == red]
        _boxplot(axes[i], subset, 'method', 'time_ms', METHOD_ORDER, 
                 zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".1f")
        axes[i].set_title(f"{red} Reduction")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Processing Time (ms)" if i == 0 else "")
        axes[i].tick_params(axis='x', rotation=15)
    fig.suptitle("Processing Time by Method and Reduction Level", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_time_by_method_reduction.png"))
    plt.close(fig)


def plot_memory_by_method(df, output_dir, zoom=False, zoom_pct=25, show_labels=True):
    """Boxplot: Memory usage per method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _boxplot(ax, df, 'method', 'memory_mb', METHOD_ORDER, 
             zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".1f")
    ax.set_ylabel("Memory Delta (MB)")
    ax.set_xlabel("")
    ax.set_title("Memory Usage Distribution by Method")
    fig.savefig(os.path.join(output_dir, "boxplot_memory_by_method.png"))
    plt.close(fig)


def plot_geometric_accuracy(df, output_dir, zoom=False, zoom_pct=25, show_labels=True):
    """Boxplots: Hausdorff distance and RMSE per method (run 1 only)."""
    geo = df.dropna(subset=['hausdorff_pct'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Hausdorff
    _boxplot(ax1, geo, 'method', 'hausdorff_pct', METHOD_ORDER, 
             zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".3f")
    ax1.set_ylabel("Hausdorff Distance (% of bbox diagonal)")
    ax1.set_xlabel("")
    ax1.set_title("Geometric Accuracy: Hausdorff Distance")
    
    # RMSE
    _boxplot(ax2, geo, 'method', 'rmse_pct', METHOD_ORDER, 
             zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".4f")
    ax2.set_ylabel("RMSE (% of bbox diagonal)")
    ax2.set_xlabel("")
    ax2.set_title("Geometric Accuracy: RMSE")
    
    fig.suptitle("Geometric Accuracy by Method (Normalized to Bounding Box Diagonal)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_geometric_accuracy.png"))
    plt.close(fig)


def plot_accuracy_by_reduction(df, output_dir, zoom=False, zoom_pct=25, show_labels=True):
    """Boxplots: Hausdorff and RMSE per method x reduction level."""
    geo = df.dropna(subset=['hausdorff_pct'])
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    for i, red in enumerate(REDUCTION_ORDER):
        subset = geo[geo['reduction'] == red]
        
        # Hausdorff row
        _boxplot(axes[0, i], subset, 'method', 'hausdorff_pct', METHOD_ORDER,
                 zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".3f")
        axes[0, i].set_title(f"{red} Reduction")
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("Hausdorff (%)" if i == 0 else "")
        axes[0, i].tick_params(axis='x', rotation=15)
        
        # RMSE row
        _boxplot(axes[1, i], subset, 'method', 'rmse_pct', METHOD_ORDER,
                 zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels, median_fmt=".4f")
        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel("RMSE (%)" if i == 0 else "")
        axes[1, i].tick_params(axis='x', rotation=15)
    
    fig.suptitle("Geometric Accuracy by Method and Reduction Level", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_accuracy_by_reduction.png"))
    plt.close(fig)


def plot_reduction_accuracy_target(df, output_dir):
    """Scatter: Actual vs target reduction ratio per method (identifies ceiling effects)."""
    run1 = df[df['run'] == 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Small x-jitter per method so overlapping dots become visible
    jitter_offsets = {
        "fast-simplification": -0.9,
        "open3d":              -0.3,
        "meshoptimizer":        0.3,
        "cgal":                 0.9,
    }
    
    # Seeded RNG for reproducible y-jitter
    rng = np.random.RandomState(42)
    
    for method in METHOD_ORDER:
        subset = run1[run1['method'] == method]
        y_jitter = rng.uniform(-0.3, 0.3, size=len(subset))
        ax.scatter(
            subset['target_reduction'] * 100 + jitter_offsets[method],
            subset['actual_reduction'] * 100 + y_jitter,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            alpha=0.7, s=40, edgecolors='white', linewidth=0.5
        )
    
    # Perfect accuracy line
    ax.plot([40, 95], [40, 95], 'k--', alpha=0.4, linewidth=1, label='Perfect accuracy')
    
    ax.set_xlabel("Target Reduction (%)")
    ax.set_ylabel("Actual Reduction (%)")
    ax.set_title("Reduction Accuracy: Target vs Actual")
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(40, 95)
    ax.set_ylim(40, 95)
    ax.set_aspect('equal')
    
    fig.savefig(os.path.join(output_dir, "scatter_reduction_accuracy.png"))
    plt.close(fig)


def plot_time_vs_complexity(df, output_dir):
    """Line plot: Processing time vs asset complexity (face count) per method."""
    run1 = df[df['run'] == 1].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for i, red in enumerate(REDUCTION_ORDER):
        subset = run1[run1['reduction'] == red].sort_values('input_faces')
        for method in METHOD_ORDER:
            m_data = subset[subset['method'] == method]
            axes[i].plot(
                m_data['input_faces'], m_data['time_ms'],
                color=METHOD_COLORS[method], marker='o', markersize=5,
                label=METHOD_LABELS[method] if i == 0 else "", linewidth=1.5, alpha=0.8
            )
        axes[i].set_xlabel("Input Face Count")
        axes[i].set_title(f"{red} Reduction")
        axes[i].set_ylabel("Processing Time (ms)" if i == 0 else "")
        axes[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    
    axes[0].legend(loc='upper left', framealpha=0.9)
    fig.suptitle("Processing Time vs Asset Complexity", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "line_time_vs_complexity.png"))
    plt.close(fig)


# TABLE RENDERING (as images for Word)
def render_table_image(df_table, title, output_path, col_widths=None):
    """Render a pandas DataFrame as a clean table image."""
    n_rows, n_cols = df_table.shape
    has_row_labels = bool(df_table.index.name or not isinstance(df_table.index, pd.RangeIndex))
    
    # Estimate column widths from content length
    col_char_widths = []
    for col in df_table.columns:
        max_len = len(str(col))
        for val in df_table[col]:
            max_len = max(max_len, len(str(val)))
        col_char_widths.append(max_len)
    
    if has_row_labels:
        row_label_width = max(len(str(idx)) for idx in df_table.index)
        col_char_widths.insert(0, row_label_width)
    
    # Convert character widths to proportional figure widths
    char_scale = 0.12  # inches per character, approximate
    computed_widths = [max(w * char_scale, 0.8) for w in col_char_widths]
    fig_width = max(8, sum(computed_widths) + 1.5)
    fig_height = max(2.0, (n_rows + 2) * 0.36)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12, loc='left')
    
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        rowLabels=df_table.index if has_row_labels else None,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(list(range(-1 if has_row_labels else 0, n_cols)))
    table.scale(1, 1.4)
    
    # Style header and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#dee2e6')
        
        # Row labels
        if col == -1:
            cell.set_facecolor('#34495e' if row == 0 else '#ecf0f1')
            cell.set_text_props(fontweight='bold', fontsize=8)
    
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def create_summary_table(method_summary, output_dir):
    """Create the main method comparison summary table."""
    # Performance table
    perf = pd.DataFrame({
        'Mean (ms)': method_summary['time_mean'],
        'Median (ms)': method_summary['time_median'],
        'Std (ms)': method_summary['time_std'],
        'IQR (ms)': method_summary['time_iqr'],
        'Min (ms)': method_summary['time_min'],
        'Max (ms)': method_summary['time_max'],
    })
    perf.index = [METHOD_LABELS[m] for m in perf.index]
    perf.index.name = 'Method'
    render_table_image(perf, "Table X: Processing Time Descriptive Statistics",
                       os.path.join(output_dir, "table_time_summary.png"))
    
    # Memory table
    mem = pd.DataFrame({
        'Mean (MB)': method_summary['mem_mean'],
        'Median (MB)': method_summary['mem_median'],
        'Std (MB)': method_summary['mem_std'],
        'Min (MB)': method_summary['mem_min'],
        'Max (MB)': method_summary['mem_max'],
        'Stability (%)': ((method_summary['n_total'] - method_summary['n_unstable']) / method_summary['n_total'] * 100).round(1),
    })
    mem.index = [METHOD_LABELS[m] for m in mem.index]
    mem.index.name = 'Method'
    render_table_image(mem, "Table X: Memory Usage Descriptive Statistics",
                       os.path.join(output_dir, "table_memory_summary.png"))
    
    # Geometric accuracy table
    geo = pd.DataFrame({
        'Hausdorff Mean (%)': method_summary['hausdorff_mean'],
        'Hausdorff Median (%)': method_summary['hausdorff_median'],
        'Hausdorff Std (%)': method_summary['hausdorff_std'],
        'RMSE Mean (%)': method_summary['rmse_mean'],
        'RMSE Median (%)': method_summary['rmse_median'],
        'RMSE Std (%)': method_summary['rmse_std'],
    })
    geo.index = [METHOD_LABELS[m] for m in geo.index]
    geo.index.name = 'Method'
    render_table_image(geo, "Table X: Geometric Accuracy Descriptive Statistics (% of Bounding Box Diagonal)",
                       os.path.join(output_dir, "table_accuracy_summary.png"))


def create_reduction_table(reduction_summary, output_dir):
    """Create per-reduction-level breakdown table."""
    # Reshape for readability
    for red in REDUCTION_ORDER:
        try:
            subset = reduction_summary.xs(red, level='reduction')
        except KeyError:
            continue
        
        table_data = pd.DataFrame({
            'Time Mean (ms)': subset['time_mean'],
            'Time Std (ms)': subset['time_std'],
            'Mem Mean (MB)': subset['mem_mean'],
            'Hausdorff Mean (%)': subset['hausdorff_mean'],
            'Hausdorff Std (%)': subset['hausdorff_std'],
            'RMSE Mean (%)': subset['rmse_mean'],
            'RMSE Std (%)': subset['rmse_std'],
        })
        table_data.index = [METHOD_LABELS[m] for m in table_data.index]
        table_data.index.name = 'Method'
        
        render_table_image(
            table_data.round(3),
            f"Table X: Method Comparison at {red} Reduction Level",
            os.path.join(output_dir, f"table_reduction_{red.replace('%','pct')}.png")
        )


def create_asset_table(per_asset, output_dir):
    """Create per-asset summary tables."""
    # One compact table per asset with all methods and reduction levels
    for asset in per_asset.index.get_level_values('asset').unique():
        asset_data = per_asset.loc[asset]
        
        rows = []
        for method in METHOD_ORDER:
            try:
                method_data = asset_data.loc[method]
            except KeyError:
                continue
            for red in REDUCTION_ORDER:
                try:
                    r = method_data.loc[red]
                    rows.append({
                        'Method': METHOD_LABELS[method],
                        'Reduction': red,
                        'Time (ms)': f"{r['time_ms']:.1f}",
                        'Memory (MB)': f"{r['memory_mb']:.1f}",
                        'Hausdorff (%)': f"{r['hausdorff_pct']:.3f}" if pd.notna(r['hausdorff_pct']) else "N/A",
                        'RMSE (%)': f"{r['rmse_pct']:.4f}" if pd.notna(r['rmse_pct']) else "N/A",
                        'Actual Red.': f"{r['actual_reduction']*100:.1f}%",
                    })
                except KeyError:
                    continue
        
        if rows:
            table_df = pd.DataFrame(rows)
            faces = per_asset.loc[asset].iloc[0]['input_faces']
            render_table_image(
                table_df,
                f"Table X: {asset} ({int(faces):,} faces) Detailed Results",
                os.path.join(output_dir, f"table_asset_{asset.lower()}.png")
            )


# MAIN
def main(input_json, output_dir, dpi=300, zoom=False, zoom_pct=25, show_labels=True):
    # Apply DPI override
    global FIG_DPI
    FIG_DPI = dpi
    plt.rcParams.update({'figure.dpi': dpi, 'savefig.dpi': dpi})
    
    setup_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print("Loading data...")
    df, sys_info = load_and_flatten(input_json)
    print(f"  Loaded {len(df)} data points ({df['asset'].nunique()} assets, "
          f"{df['method'].nunique()} methods, {df['reduction'].nunique()} reduction levels)")
    print(f"  Mode: {'zoomed (' + str(zoom_pct) + '%)' if zoom else 'normal'} | Labels: {'on' if show_labels else 'off'}")
    
    # Descriptive statistics
    print("\nComputing descriptive statistics...")
    method_summary = compute_method_summary(df)
    reduction_summary = compute_reduction_summary(df)
    per_asset = compute_per_asset_breakdown(df)
    
    # Save CSVs
    method_summary.to_csv(os.path.join(csv_dir, "method_summary.csv"))
    reduction_summary.to_csv(os.path.join(csv_dir, "reduction_summary.csv"))
    per_asset.to_csv(os.path.join(csv_dir, "per_asset_breakdown.csv"))
    df.to_csv(os.path.join(csv_dir, "flat_data.csv"), index=False)
    print("  CSVs saved.")
    
    # Tables (as images)
    print("\nRendering tables...")
    create_summary_table(method_summary, tables_dir)
    create_reduction_table(reduction_summary, tables_dir)
    create_asset_table(per_asset, tables_dir)
    print("  Tables saved.")
    
    # Figures
    print("\nGenerating figures...")
    plot_time_by_method(df, figures_dir, zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels)
    print("  [1/6] Processing time boxplot")
    
    plot_time_by_method_reduction(df, figures_dir, zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels)
    print("  [2/6] Processing time x reduction boxplot")
    
    plot_memory_by_method(df, figures_dir, zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels)
    print("  [3/6] Memory usage boxplot")
    
    plot_geometric_accuracy(df, figures_dir, zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels)
    print("  [4/6] Geometric accuracy boxplots")
    
    plot_accuracy_by_reduction(df, figures_dir, zoom=zoom, zoom_pct=zoom_pct, show_labels=show_labels)
    print("  [5/6] Accuracy x reduction boxplots")
    
    plot_reduction_accuracy_target(df, figures_dir)
    print("  [6/6] Reduction accuracy scatter")
    
    plot_time_vs_complexity(df, figures_dir)
    print("  [+1]  Time vs complexity line plot")
    
    # Anomaly report
    print("\nGenerating anomaly report...")
    report = generate_anomaly_report(df)
    report_path = os.path.join(output_dir, "anomaly_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)
    
    # Summary printout
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in: {output_dir}/")
    print(f"  figures/  -- {len(os.listdir(figures_dir))} visualization files")
    print(f"  tables/   -- {len(os.listdir(tables_dir))} table images")
    print(f"  csv/      -- {len(os.listdir(csv_dir))} CSV exports")
    print(f"  anomaly_report.txt")
    
    return df, method_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mesh Simplification Benchmark - Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
               "  python analyze_benchmarks.py\n"
               "  python analyze_benchmarks.py --zoom\n"
               "  python analyze_benchmarks.py --zoom --zoom-pct 15\n"
               "  python analyze_benchmarks.py --zoom --no-labels\n"
               "  python analyze_benchmarks.py -i results/batch_report.json -o results/analysis\n"
               "  python analyze_benchmarks.py --dpi 150  # faster, lower-res for previewing"
    )
    parser.add_argument(
        "-i", "--input",
        default=INPUT_JSON,
        help=f"Path to batch_report.json (default: {INPUT_JSON})"
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIG_DPI,
        help=f"Figure resolution in DPI (default: {FIG_DPI})"
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        default=False,
        help="Generate zoomed boxplots (linear scale, clips high-value methods)"
    )
    parser.add_argument(
        "--zoom-pct",
        type=int,
        default=25,
        metavar="N",
        help="Min %% of chart height for smallest visible boxplot (default: 25)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        default=False,
        help="Suppress median value labels on boxplots"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.dpi, 
         zoom=args.zoom, zoom_pct=args.zoom_pct, show_labels=not args.no_labels)