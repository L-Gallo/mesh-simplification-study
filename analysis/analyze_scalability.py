#!/usr/bin/env python3
"""
Mesh Simplification Benchmark - Scalability Analysis

Generates scaling visualizations and feasibility tables from Stanford
3D Scanning Repository benchmark data (single repetition, no accuracy).

Companion to analyze_benchmarks.py (production asset analysis).
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

# CONFIG
INPUT_JSON = "batch_report.json"
OUTPUT_DIR = "scalability_output"
FIG_DPI = 300
FIG_FORMAT = "png"

# Colorblind-friendly palette (Wong, 2011 -- Nature Methods)
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
METHOD_MARKERS = {
    "fast-simplification": "o",
    "open3d":              "s",
    "meshoptimizer":       "D",
    "cgal":                "^",
}
REDUCTION_ORDER = ["50%", "80%", "90%"]

# Assets are sorted by face count for x-axis ordering
# (populated dynamically from data)
ASSET_ORDER = []

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
    """Load batch_report.json and flatten into a pandas DataFrame.
    
    Designed for scalability data: single repetition, no geometric accuracy.
    """
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
                    }
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    df['method'] = pd.Categorical(df['method'], categories=METHOD_ORDER, ordered=True)
    df['reduction'] = pd.Categorical(df['reduction'], categories=REDUCTION_ORDER, ordered=True)
    
    # Time in seconds for readability at large scales
    df['time_s'] = df['time_ms'] / 1000.0
    
    # Flag unreliable memory measurements:
    # negative delta AND near-zero peak indicates RSS polling missed the actual allocation
    df['memory_unreliable'] = (df['memory_mb'] < 0) & (df['peak_memory_mb'] < 1.0)
    
    # Sort assets by face count (ascending complexity)
    asset_faces = df.groupby('asset')['input_faces'].first().sort_values()
    global ASSET_ORDER
    ASSET_ORDER = asset_faces.index.tolist()
    
    return df, data.get('system_info', {})


def get_asset_complexity_map(df):
    """Return dict of asset_name -> face_count for labeling."""
    return df.groupby('asset')['input_faces'].first().to_dict()


# AXIS FORMATTING HELPERS
def format_face_count(x, _=None):
    """Format face counts for axis labels: 69K, 346K, 871K, 28.1M."""
    if x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.0f}K'
    return str(int(x))


def format_time(x, _=None):
    """Format time values: ms for <1s, s for >=1s, min for >=60s."""
    if x < 1_000:
        return f'{x:.0f}ms'
    elif x < 60_000:
        return f'{x / 1_000:.1f}s'
    else:
        return f'{x / 60_000:.1f}min'


def format_memory(x, _=None):
    """Format memory values: MB or GB."""
    if abs(x) >= 1_000:
        return f'{x / 1_000:.1f}GB'
    return f'{x:.0f}MB'


# VISUALIZATIONS

def _draw_time_lines(ax, subset, label_col_idx=None):
    """Draw time lines for all methods on a given axes. Returns nothing.
    
    label_col_idx: If 0, add legend labels. Otherwise skip labels.
    """
    for method in METHOD_ORDER:
        m_data = subset[subset['method'] == method]
        ax.plot(
            m_data['input_faces'], m_data['time_ms'],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=7, linewidth=1.8, alpha=0.85,
            label=METHOD_LABELS[method] if label_col_idx == 0 else "",
            zorder=3
        )


def _draw_memory_lines(ax, subset, label_col_idx=None):
    """Draw peak memory lines with unreliable-measurement handling."""
    for method in METHOD_ORDER:
        m_data = subset[subset['method'] == method]
        
        reliable = m_data[~m_data['memory_unreliable']]
        unreliable = m_data[m_data['memory_unreliable']]
        
        if len(reliable) > 0:
            ax.plot(
                reliable['input_faces'], reliable['peak_memory_mb'],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=7, linewidth=1.8, alpha=0.85,
                label=METHOD_LABELS[method] if label_col_idx == 0 else "",
                zorder=3
            )
        
        if len(unreliable) > 0:
            ax.plot(
                unreliable['input_faces'], unreliable['peak_memory_mb'],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=8, linewidth=1.2, alpha=0.5,
                linestyle='none', markerfacecolor='white',
                markeredgewidth=1.5, zorder=4
            )
            if len(reliable) > 0:
                last_reliable = reliable.iloc[-1]
                for _, unrel_row in unreliable.iterrows():
                    ax.plot(
                        [last_reliable['input_faces'], unrel_row['input_faces']],
                        [last_reliable['peak_memory_mb'], unrel_row['peak_memory_mb']],
                        color=METHOD_COLORS[method],
                        linewidth=1.0, alpha=0.4, linestyle='--', zorder=2
                    )


def _annotate_points(ax, subset, value_col, fmt_func):
    """Add value labels next to each data point for readability on linear plots."""
    for method in METHOD_ORDER:
        m_data = subset[subset['method'] == method]
        for _, row in m_data.iterrows():
            val = row[value_col]
            if pd.isna(val) or val <= 0:
                continue
            ax.annotate(
                fmt_func(val),
                xy=(row['input_faces'], val),
                xytext=(4, 4), textcoords='offset points',
                fontsize=6.5, color=METHOD_COLORS[method],
                fontweight='bold', alpha=0.85,
                zorder=5
            )


def _format_linear_x(ax, face_counts):
    """Format x-axis with log scale for even spacing of complexity tiers."""
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_face_count))
    ax.set_xticks(face_counts)
    # Suppress minor ticks that log scale adds automatically
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis='x', rotation=30)


# PRIMARY FIGURES (linear scale, main chapter)

def plot_time_linear(df, output_dir):
    """Primary figure: Processing time vs complexity on LINEAR scale.
    
    Top row: full range (all models, Lucy dominates -- shows practical impact).
    Bottom row: zoomed (excluding largest model -- shows sub-million detail).
    """
    complexity = get_asset_complexity_map(df)
    face_counts_all = sorted(complexity.values())
    
    # Identify the largest asset to exclude in zoom row
    largest_asset = ASSET_ORDER[-1]
    df_zoom = df[df['asset'] != largest_asset]
    face_counts_zoom = sorted(
        v for a, v in complexity.items() if a != largest_asset
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    for i, red in enumerate(REDUCTION_ORDER):
        # Full range row
        subset_full = df[df['reduction'] == red].sort_values('input_faces')
        _draw_time_lines(axes[0, i], subset_full, label_col_idx=i)
        
        axes[0, i].set_title(f"{red} Reduction")
        axes[0, i].set_ylabel("Processing Time" if i == 0 else "")
        axes[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
        _format_linear_x(axes[0, i], face_counts_all)
        axes[0, i].set_ylim(bottom=0)
        
        # Zoomed row (excluding largest)
        subset_zoom = df_zoom[df_zoom['reduction'] == red].sort_values('input_faces')
        _draw_time_lines(axes[1, i], subset_zoom, label_col_idx=None)
        
        axes[1, i].set_xlabel("Input Face Count")
        axes[1, i].set_ylabel("Processing Time" if i == 0 else "")
        axes[1, i].yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
        _format_linear_x(axes[1, i], face_counts_zoom)
        axes[1, i].set_ylim(bottom=0)
    
    axes[0, 0].legend(loc='upper left', framealpha=0.9)
    
    # Row labels
    axes[0, 0].annotate('All models', xy=(0, 0.5), xytext=(-65, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=10, fontweight='bold', ha='center', va='center',
                         rotation=90)
    axes[1, 0].annotate(f'Excluding {largest_asset.capitalize()}', xy=(0, 0.5), xytext=(-65, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=10, fontweight='bold', ha='center', va='center',
                         rotation=90)
    
    fig.suptitle("Processing Time Scaling by Input Complexity", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1)
    fig.savefig(os.path.join(output_dir, "time_scaling_linear.png"))
    plt.close(fig)


def plot_memory_linear(df, output_dir):
    """Primary figure: Peak memory vs complexity on LINEAR scale.
    
    Top row: full range. Bottom row: zoomed (excluding largest model).
    """
    complexity = get_asset_complexity_map(df)
    face_counts_all = sorted(complexity.values())
    has_unreliable = df['memory_unreliable'].any()
    
    largest_asset = ASSET_ORDER[-1]
    df_zoom = df[df['asset'] != largest_asset]
    face_counts_zoom = sorted(
        v for a, v in complexity.items() if a != largest_asset
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    for i, red in enumerate(REDUCTION_ORDER):
        # Full range
        subset_full = df[df['reduction'] == red].sort_values('input_faces')
        _draw_memory_lines(axes[0, i], subset_full, label_col_idx=i)
        
        axes[0, i].set_title(f"{red} Reduction")
        axes[0, i].set_ylabel("Peak Memory" if i == 0 else "")
        axes[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(format_memory))
        _format_linear_x(axes[0, i], face_counts_all)
        axes[0, i].set_ylim(bottom=0)
        
        # Zoomed (excluding largest)
        subset_zoom = df_zoom[df_zoom['reduction'] == red].sort_values('input_faces')
        _draw_memory_lines(axes[1, i], subset_zoom, label_col_idx=None)
        
        axes[1, i].set_xlabel("Input Face Count")
        axes[1, i].set_ylabel("Peak Memory" if i == 0 else "")
        axes[1, i].yaxis.set_major_formatter(ticker.FuncFormatter(format_memory))
        _format_linear_x(axes[1, i], face_counts_zoom)
        axes[1, i].set_ylim(bottom=0)
    
    axes[0, 0].legend(loc='upper left', framealpha=0.9)
    
    axes[0, 0].annotate('All models', xy=(0, 0.5), xytext=(-65, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=10, fontweight='bold', ha='center', va='center',
                         rotation=90)
    axes[1, 0].annotate(f'Excluding {largest_asset.capitalize()}', xy=(0, 0.5), xytext=(-65, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=10, fontweight='bold', ha='center', va='center',
                         rotation=90)
    
    title = "Peak Memory Scaling by Input Complexity"
    if has_unreliable:
        title += "\n(open markers = unreliable RSS measurement)"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1)
    fig.savefig(os.path.join(output_dir, "memory_scaling_linear.png"))
    plt.close(fig)


def plot_combined_linear(df, output_dir):
    """Primary combined figure: time + memory, full range only, LINEAR scale.
    
    For the main chapter -- single compact figure with both metrics.
    """
    complexity = get_asset_complexity_map(df)
    face_counts = sorted(complexity.values())
    has_unreliable = df['memory_unreliable'].any()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex='col')
    
    for i, red in enumerate(REDUCTION_ORDER):
        subset = df[df['reduction'] == red].sort_values('input_faces')
        
        # Time row
        _draw_time_lines(axes[0, i], subset, label_col_idx=i)
        axes[0, i].set_title(f"{red} Reduction")
        axes[0, i].set_ylabel("Processing Time" if i == 0 else "")
        axes[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
        axes[0, i].set_ylim(bottom=0)
        
        # Memory row
        _draw_memory_lines(axes[1, i], subset, label_col_idx=None)
        axes[1, i].set_xlabel("Input Face Count")
        axes[1, i].set_ylabel("Peak Memory" if i == 0 else "")
        axes[1, i].yaxis.set_major_formatter(ticker.FuncFormatter(format_memory))
        axes[1, i].xaxis.set_major_formatter(ticker.FuncFormatter(format_face_count))
        axes[1, i].set_ylim(bottom=0)
        _format_linear_x(axes[1, i], face_counts)
    
    axes[0, 0].legend(loc='upper left', framealpha=0.9)
    title = "Scalability: Processing Time and Peak Memory by Input Complexity"
    if has_unreliable:
        title += "\n(open markers = unreliable RSS measurement)"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_scaling_linear.png"))
    plt.close(fig)


# SUPPLEMENTARY FIGURES (log-log scale, appendix)

def plot_time_loglog(df, output_dir):
    """Supplementary: Processing time on log-log scale (reveals scaling exponents)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    complexity = get_asset_complexity_map(df)
    face_counts = sorted(complexity.values())
    
    for i, red in enumerate(REDUCTION_ORDER):
        ax = axes[i]
        subset = df[df['reduction'] == red].sort_values('input_faces')
        _draw_time_lines(ax, subset, label_col_idx=i)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Input Face Count")
        ax.set_title(f"{red} Reduction")
        ax.set_ylabel("Processing Time" if i == 0 else "")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_face_count))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
        ax.set_xticks(face_counts)
        ax.tick_params(axis='x', rotation=30)
    
    axes[0].legend(loc='upper left', framealpha=0.9)
    fig.suptitle("Processing Time Scaling (Log-Log)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "time_scaling_loglog.png"))
    plt.close(fig)


def plot_memory_loglog(df, output_dir):
    """Supplementary: Peak memory on log-log scale (reveals scaling exponents)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    complexity = get_asset_complexity_map(df)
    face_counts = sorted(complexity.values())
    has_unreliable = df['memory_unreliable'].any()
    
    for i, red in enumerate(REDUCTION_ORDER):
        ax = axes[i]
        subset = df[df['reduction'] == red].sort_values('input_faces')
        _draw_memory_lines(ax, subset, label_col_idx=i)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Input Face Count")
        ax.set_title(f"{red} Reduction")
        ax.set_ylabel("Peak Memory" if i == 0 else "")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_face_count))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_memory))
        ax.set_xticks(face_counts)
        ax.tick_params(axis='x', rotation=30)
    
    axes[0].legend(loc='upper left', framealpha=0.9)
    title = "Peak Memory Scaling (Log-Log)"
    if has_unreliable:
        title += "\n(open markers = unreliable RSS measurement)"
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "memory_scaling_loglog.png"))
    plt.close(fig)


def plot_reduction_accuracy(df, output_dir):
    """Scatter: Actual vs target reduction ratio (scalability models)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    jitter_offsets = {
        "fast-simplification": -0.9,
        "open3d":              -0.3,
        "meshoptimizer":        0.3,
        "cgal":                 0.9,
    }
    rng = np.random.RandomState(42)
    
    for method in METHOD_ORDER:
        subset = df[df['method'] == method]
        y_jitter = rng.uniform(-0.3, 0.3, size=len(subset))
        ax.scatter(
            subset['target_reduction'] * 100 + jitter_offsets[method],
            subset['actual_reduction'] * 100 + y_jitter,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            alpha=0.7, s=50, edgecolors='white', linewidth=0.5,
            marker=METHOD_MARKERS[method]
        )
    
    ax.plot([40, 95], [40, 95], 'k--', alpha=0.4, linewidth=1, label='Perfect accuracy')
    ax.set_xlabel("Target Reduction (%)")
    ax.set_ylabel("Actual Reduction (%)")
    ax.set_title("Reduction Accuracy: Stanford Models")
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(40, 95)
    ax.set_ylim(40, 95)
    ax.set_aspect('equal')
    
    fig.savefig(os.path.join(output_dir, "scatter_reduction_accuracy_scalability.png"))
    plt.close(fig)


# TABLE RENDERING (as images for Word)
def render_table_image(df_table, title, output_path):
    """Render a pandas DataFrame as a clean table image for Word.
    
    Handles multi-line cell content (newlines in values) by computing
    widths from the longest line and scaling row height accordingly.
    """
    n_rows, n_cols = df_table.shape
    has_row_labels = bool(df_table.index.name or not isinstance(df_table.index, pd.RangeIndex))
    
    def _max_line_len(text):
        """Return the length of the longest line in a potentially multi-line string."""
        return max(len(line) for line in str(text).split('\n')) if str(text) else 0
    
    def _line_count(text):
        """Return number of lines in a cell value."""
        return str(text).count('\n') + 1
    
    # Compute column widths from longest single line (not total string length)
    col_char_widths = []
    for col in df_table.columns:
        max_len = _max_line_len(col)
        for val in df_table[col]:
            max_len = max(max_len, _max_line_len(val))
        col_char_widths.append(max_len)
    
    if has_row_labels:
        row_label_width = max(_max_line_len(idx) for idx in df_table.index)
        col_char_widths.insert(0, row_label_width)
    
    # Detect whether any cell has multi-line content
    max_lines_per_row = []
    for _, row_data in df_table.iterrows():
        row_max = max(_line_count(val) for val in row_data)
        max_lines_per_row.append(row_max)
    # Header row
    header_lines = max(_line_count(col) for col in df_table.columns)
    
    char_scale = 0.14  # slightly more generous than 0.12
    computed_widths = [max(w * char_scale, 1.0) for w in col_char_widths]
    fig_width = max(8, sum(computed_widths) + 2.0)
    
    # Row height scales with max line count per row
    base_row_height = 0.36
    total_row_height = header_lines * base_row_height
    for ml in max_lines_per_row:
        total_row_height += ml * base_row_height
    fig_height = max(2.5, total_row_height + 1.5)
    
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
    
    # Manually set column widths -- auto_set_column_width fails on multi-line text.
    # Normalize our computed char widths to proportional table widths.
    total_computed = sum(computed_widths)
    for (row, col), cell in table.get_celld().items():
        if col == -1:
            # Row label column
            cell.set_width(computed_widths[0] / total_computed)
        else:
            idx = col + (1 if has_row_labels else 0)
            cell.set_width(computed_widths[idx] / total_computed)
    
    # Scale row height: base 1.4, bump for multi-line rows
    has_multiline = any(ml > 1 for ml in max_lines_per_row) or header_lines > 1
    row_scale = 1.8 if has_multiline else 1.4
    table.scale(1, row_scale)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#dee2e6')
        
        if col == -1:
            cell.set_facecolor('#34495e' if row == 0 else '#ecf0f1')
            cell.set_text_props(fontweight='bold', fontsize=8)
    
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# TABLES
def create_complete_results_table(df, output_dir):
    """Full results table: every asset x method x reduction."""
    complexity = get_asset_complexity_map(df)
    
    rows = []
    for asset in ASSET_ORDER:
        faces = complexity[asset]
        for method in METHOD_ORDER:
            for red in REDUCTION_ORDER:
                mask = (df['asset'] == asset) & (df['method'] == method) & (df['reduction'] == red)
                r = df[mask].iloc[0]
                mem_str = format_memory(r['peak_memory_mb'])
                if r['memory_unreliable']:
                    mem_str += ' *'
                rows.append({
                    'Asset': asset.capitalize(),
                    'Faces': f'{int(faces):,}',
                    'Method': METHOD_LABELS[method],
                    'Reduction': red,
                    'Time': format_time(r['time_ms']),
                    'Peak Mem.': mem_str,
                    'Actual Red.': f"{r['actual_reduction'] * 100:.1f}%",
                })
    
    table_df = pd.DataFrame(rows)
    
    title = "Table X: Scalability Test Complete Results (Stanford Models)"
    if df['memory_unreliable'].any():
        title += "  (* = unreliable RSS measurement)"
    
    render_table_image(
        table_df,
        title,
        os.path.join(output_dir, "table_scalability_full.png")
    )


def create_feasibility_table(df, output_dir):
    """Feasibility summary: per method x asset, showing time and memory at 50% reduction.
    
    Uses 50% as the representative reduction level (lightest workload = best-case).
    """
    complexity = get_asset_complexity_map(df)
    subset = df[df['reduction'] == '50%']
    
    rows = []
    for method in METHOD_ORDER:
        row = {'Method': METHOD_LABELS[method]}
        for asset in ASSET_ORDER:
            mask = (subset['asset'] == asset) & (subset['method'] == method)
            r = subset[mask].iloc[0]
            time_str = format_time(r['time_ms'])
            mem_str = format_memory(r['peak_memory_mb'])
            if r['memory_unreliable']:
                mem_str += ' *'
            row[f'{asset.capitalize()}\n({format_face_count(complexity[asset])})'] = f'{time_str}\n{mem_str}'
        rows.append(row)
    
    table_df = pd.DataFrame(rows).set_index('Method')
    table_df.index.name = 'Method'
    
    title = "Table X: Feasibility at 50% Reduction (Time / Peak Memory)"
    if df['memory_unreliable'].any():
        title += "  (* = unreliable RSS measurement)"
    
    render_table_image(
        table_df,
        title,
        os.path.join(output_dir, "table_feasibility.png")
    )


def create_scaling_factor_table(df, output_dir):
    """Compute and display scaling factors between adjacent complexity tiers.
    
    For each pair of consecutive assets (by face count), compute:
      factor = time(larger_asset) / time(smaller_asset)
      face_ratio = faces(larger) / faces(smaller)
    
    This reveals whether methods scale linearly, super-linearly, etc.
    """
    if len(ASSET_ORDER) < 2:
        return
    
    complexity = get_asset_complexity_map(df)
    subset = df[df['reduction'] == '50%']  # representative level
    
    rows = []
    for method in METHOD_ORDER:
        row = {'Method': METHOD_LABELS[method]}
        for j in range(1, len(ASSET_ORDER)):
            prev_asset = ASSET_ORDER[j - 1]
            curr_asset = ASSET_ORDER[j]
            
            face_ratio = complexity[curr_asset] / complexity[prev_asset]
            
            prev_time = subset[(subset['asset'] == prev_asset) & 
                               (subset['method'] == method)]['time_ms'].iloc[0]
            curr_time = subset[(subset['asset'] == curr_asset) & 
                               (subset['method'] == method)]['time_ms'].iloc[0]
            
            time_factor = curr_time / prev_time if prev_time > 0 else float('inf')
            
            label = (f'{prev_asset.capitalize()} -> {curr_asset.capitalize()}\n'
                     f'({face_ratio:.1f}x faces)')
            row[label] = f'{time_factor:.1f}x'
        rows.append(row)
    
    table_df = pd.DataFrame(rows).set_index('Method')
    table_df.index.name = 'Method'
    render_table_image(
        table_df,
        "Table X: Time Scaling Factors Between Complexity Tiers (at 50% Reduction)",
        os.path.join(output_dir, "table_scaling_factors.png")
    )


def create_asset_overview_table(df, output_dir):
    """Asset overview: face/vertex counts and vertex-to-face ratio."""
    complexity = get_asset_complexity_map(df)
    vertex_map = df.groupby('asset')['input_vertices'].first().to_dict()
    
    rows = []
    for asset in ASSET_ORDER:
        faces = complexity[asset]
        verts = vertex_map[asset]
        rows.append({
            'Asset': asset.capitalize(),
            'Vertices': f'{int(verts):,}',
            'Faces': f'{int(faces):,}',
            'V:F Ratio': f'{verts / faces:.2f}',
        })
    
    table_df = pd.DataFrame(rows)
    render_table_image(
        table_df,
        "Table X: Stanford 3D Scanning Repository Models -- Scalability Test Suite",
        os.path.join(output_dir, "table_asset_overview.png")
    )


# SCALABILITY REPORT
def generate_scalability_report(df):
    """Generate a text report summarizing scalability findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("SCALABILITY ANALYSIS REPORT")
    lines.append("=" * 70)
    
    complexity = get_asset_complexity_map(df)
    
    # 1. Dataset overview
    lines.append("\n1. DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Models: {len(ASSET_ORDER)}")
    lines.append(f"  Complexity range: {format_face_count(min(complexity.values()))} "
                 f"to {format_face_count(max(complexity.values()))} faces "
                 f"({max(complexity.values()) / min(complexity.values()):.0f}x range)")
    lines.append(f"  Methods: {len(METHOD_ORDER)}")
    lines.append(f"  Reduction levels: {', '.join(REDUCTION_ORDER)}")
    lines.append(f"  Repetitions: 1 (feasibility and scaling trends, not measurement stability)")
    lines.append(f"  Geometric accuracy: Disabled (scalability tier)")
    lines.append(f"  Total operations: {len(df)}")
    lines.append(f"  Success rate: {df['success'].mean() * 100:.0f}%")
    
    # 2. Completion times at extreme (Lucy)
    lines.append("\n\n2. LARGEST MODEL PERFORMANCE (Lucy -- "
                 f"{format_face_count(complexity.get('lucy', 0))} faces)")
    lines.append("-" * 40)
    lucy = df[df['asset'] == 'lucy']
    if len(lucy) > 0:
        for method in METHOD_ORDER:
            m_data = lucy[lucy['method'] == method]
            if len(m_data) == 0:
                continue
            times = m_data['time_ms'].values
            mems = m_data['peak_memory_mb'].values
            lines.append(
                f"  {METHOD_LABELS[method]:22s}: "
                f"{format_time(min(times)):>8s} - {format_time(max(times)):>8s} | "
                f"Peak memory: {format_memory(max(mems))}"
            )
    else:
        lines.append("  (No data for lucy)")
    
    # 3. Scaling factors at 50% reduction
    lines.append("\n\n3. TIME SCALING FACTORS (50% reduction)")
    lines.append("-" * 40)
    subset = df[df['reduction'] == '50%']
    for method in METHOD_ORDER:
        factors = []
        for j in range(1, len(ASSET_ORDER)):
            prev = ASSET_ORDER[j - 1]
            curr = ASSET_ORDER[j]
            face_ratio = complexity[curr] / complexity[prev]
            prev_t = subset[(subset['asset'] == prev) & (subset['method'] == method)]['time_ms'].iloc[0]
            curr_t = subset[(subset['asset'] == curr) & (subset['method'] == method)]['time_ms'].iloc[0]
            time_factor = curr_t / prev_t if prev_t > 0 else float('inf')
            factors.append(f'{face_ratio:.1f}xfaces -> {time_factor:.1f}xtime')
        lines.append(f"  {METHOD_LABELS[method]:22s}: {' | '.join(factors)}")
    
    # 4. Reduction accuracy check
    lines.append("\n\n4. REDUCTION ACCURACY")
    lines.append("-" * 40)
    threshold = 0.02
    deviations = df[abs(df['actual_reduction'] - df['target_reduction']) > threshold]
    if len(deviations) > 0:
        lines.append(f"  {len(deviations)} cases with >2pp deviation from target:")
        for _, row in deviations.iterrows():
            lines.append(
                f"    {row['asset']:12s} | {METHOD_LABELS[row['method']]:22s} | "
                f"Target: {row['target_reduction']*100:.0f}% | "
                f"Actual: {row['actual_reduction']*100:.1f}% | "
                f"Deviation: {(row['actual_reduction'] - row['target_reduction'])*100:+.1f}pp"
            )
    else:
        lines.append("  All reductions within 2pp of target.")
    
    # 5. Memory notes
    lines.append("\n\n5. MEMORY MEASUREMENT NOTES")
    lines.append("-" * 40)
    lines.append("  CGAL executes as an external subprocess. Peak memory measurements")
    lines.append("  reflect Python-side RSS of the subprocess wrapper, not CGAL's internal")
    lines.append("  allocation. Values are directionally correct but not directly comparable")
    lines.append("  to the in-process methods (fast-simplification, Open3D, meshoptimizer).")
    
    # Check for negative memory deltas (Open3D RSS artifact)
    neg_mem = df[df['memory_mb'] < 0]
    if len(neg_mem) > 0:
        lines.append(f"\n  {len(neg_mem)} negative memory delta(s) detected (RSS polling artifact):")
        for _, row in neg_mem.iterrows():
            lines.append(
                f"    {row['asset']:12s} | {METHOD_LABELS[row['method']]:22s} | "
                f"{row['reduction']} | delta: {row['memory_mb']:.1f} MB"
            )
        lines.append("  Peak memory values used instead of deltas for scalability figures.")
    
    return "\n".join(lines)


# CSV EXPORTS
def export_csvs(df, csv_dir):
    """Export flat data and computed scaling factors."""
    # Flat data
    df.to_csv(os.path.join(csv_dir, "scalability_flat_data.csv"), index=False)
    
    # Scaling factors
    complexity = get_asset_complexity_map(df)
    subset = df[df['reduction'] == '50%']
    
    sf_rows = []
    for method in METHOD_ORDER:
        for j in range(1, len(ASSET_ORDER)):
            prev = ASSET_ORDER[j - 1]
            curr = ASSET_ORDER[j]
            face_ratio = complexity[curr] / complexity[prev]
            
            for red in REDUCTION_ORDER:
                red_sub = df[df['reduction'] == red]
                prev_t = red_sub[(red_sub['asset'] == prev) & (red_sub['method'] == method)]['time_ms'].iloc[0]
                curr_t = red_sub[(red_sub['asset'] == curr) & (red_sub['method'] == method)]['time_ms'].iloc[0]
                prev_m = red_sub[(red_sub['asset'] == prev) & (red_sub['method'] == method)]['peak_memory_mb'].iloc[0]
                curr_m = red_sub[(red_sub['asset'] == curr) & (red_sub['method'] == method)]['peak_memory_mb'].iloc[0]
                
                sf_rows.append({
                    'method': method,
                    'reduction': red,
                    'from_asset': prev,
                    'to_asset': curr,
                    'face_ratio': round(face_ratio, 2),
                    'time_factor': round(curr_t / prev_t, 2) if prev_t > 0 else None,
                    'memory_factor': round(curr_m / prev_m, 2) if prev_m > 0 else None,
                })
    
    pd.DataFrame(sf_rows).to_csv(
        os.path.join(csv_dir, "scaling_factors.csv"), index=False
    )


# MAIN
def main(input_json, output_dir, dpi=300):
    global FIG_DPI
    FIG_DPI = dpi
    plt.rcParams.update({'figure.dpi': dpi, 'savefig.dpi': dpi})
    
    setup_style()
    
    # Create output directories
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    csv_dir = os.path.join(output_dir, "csv")
    for d in [figures_dir, tables_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)
    
    print("Loading data...")
    df, sys_info = load_and_flatten(input_json)
    print(f"  Loaded {len(df)} data points ({df['asset'].nunique()} assets, "
          f"{df['method'].nunique()} methods, {df['reduction'].nunique()} levels)")
    print(f"  Asset order (by complexity): {', '.join(ASSET_ORDER)}")
    
    complexity = get_asset_complexity_map(df)
    for asset in ASSET_ORDER:
        print(f"    {asset:12s}: {int(complexity[asset]):>12,} faces")
    
    # Figures
    print("\nGenerating primary figures (linear scale)...")
    
    plot_time_linear(df, figures_dir)
    print("  [1/5] Time scaling (linear, full + zoomed)")
    
    plot_memory_linear(df, figures_dir)
    print("  [2/5] Memory scaling (linear, full + zoomed)")
    
    plot_combined_linear(df, figures_dir)
    print("  [3/5] Combined scaling (linear)")
    
    plot_reduction_accuracy(df, figures_dir)
    print("  [4/5] Reduction accuracy scatter")
    
    print("\nGenerating supplementary figures (log-log scale)...")
    
    plot_time_loglog(df, figures_dir)
    print("  [5a/5] Time scaling (log-log)")
    
    plot_memory_loglog(df, figures_dir)
    print("  [5b/5] Memory scaling (log-log)")
    
    # Tables
    print("\nGenerating tables...")
    
    create_asset_overview_table(df, tables_dir)
    print("  [1/4] Asset overview")
    
    create_complete_results_table(df, tables_dir)
    print("  [2/4] Complete results")
    
    create_feasibility_table(df, tables_dir)
    print("  [3/4] Feasibility summary")
    
    create_scaling_factor_table(df, tables_dir)
    print("  [4/4] Scaling factors")
    
    # CSV exports
    print("\nExporting CSVs...")
    export_csvs(df, csv_dir)
    print("  CSVs saved.")
    
    # Report
    print("\nGenerating scalability report...")
    report = generate_scalability_report(df)
    report_path = os.path.join(output_dir, "scalability_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    
    # Summary
    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in: {output_dir}/")
    print(f"  figures/  -- {len(os.listdir(figures_dir))} visualization files")
    print(f"  tables/   -- {len(os.listdir(tables_dir))} table images")
    print(f"  csv/      -- {len(os.listdir(csv_dir))} CSV exports")
    print(f"  scalability_report.txt")
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mesh Simplification Benchmark -- Scalability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
               "  python analyze_scalability.py\n"
               "  python analyze_scalability.py -i scalability_report.json\n"
               "  python analyze_scalability.py --dpi 150  # fast preview"
    )
    parser.add_argument(
        "-i", "--input",
        default=INPUT_JSON,
        help=f"Path to batch_report.json (default: {INPUT_JSON})"
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIG_DPI,
        help=f"Figure resolution (default: {FIG_DPI})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.dpi)