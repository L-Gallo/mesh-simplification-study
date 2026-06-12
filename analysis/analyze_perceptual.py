#!/usr/bin/env python3
"""
Perceptual Quality Assessment - Data Analysis & Figure Generation

Generates descriptive statistics, figures, and correlation analysis
from participant_responses.json + batch_report.json.

Matches visual conventions from analyze_benchmarks.py (Wong palette,
publication-quality figures, same method ordering).
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

# CONFIG
PERCEPTUAL_JSON = "participant_responses.json"
BENCHMARK_JSON = "batch_report.json"
OUTPUT_DIR = "perceptual_output"
FIG_DPI = 300
FIG_FORMAT = "png"

# Colorblind-friendly palette (Wong, 2011 - Nature Methods)
# Matches analyze_benchmarks.py exactly
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
METHOD_LABELS_FULL = {
    "fast-simplification": "Fast-Simplification",
    "open3d": "Open3D",
    "meshoptimizer": "Meshoptimizer",
    "cgal": "CGAL",
}
REDUCTION_ORDER = ["50", "80", "90"]
REDUCTION_LABELS = {"50": "50%", "80": "80%", "90": "90%"}
VIEW_ORDER = ["distant", "close"]

ASSET_LABELS = {
    "AK74": "AK-74", "LAV": "LAV", "M9_pistol": "M9 Pistol",
    "Mi8": "Mi-8", "bunker": "Bunker", "church": "Church",
    "jeep": "Jeep", "watermill": "Watermill",
}
ASSET_ORDER = ["AK74", "LAV", "M9_pistol", "Mi8", "bunker", "church", "jeep", "watermill"]


# STYLE SETUP (matches benchmark script)
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


# DATA LOADING
def load_perceptual(json_path):
    """Load perceptual responses and parse pair_id components."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data['responses'])

    # Parse pair_id: {asset}_{reduction}_{methodA}_vs_{methodB}
    def parse_pair(pair_id):
        vs_idx = pair_id.index('_vs_')
        before = pair_id[:vs_idx]
        method_b = pair_id[vs_idx + 4:]
        parts = before.rsplit('_', 2)
        return parts[0], parts[1], parts[2], method_b

    parsed = df['pair_id'].apply(lambda x: pd.Series(parse_pair(x)))
    df[['asset', 'reduction', 'method_a', 'method_b']] = parsed
    return df


def load_geometric_accuracy(json_path):
    """Load geometric accuracy metrics from batch_report.json.
    Returns DataFrame with one row per (asset, method, reduction) using run 1."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for asset_name, asset_data in data['assets'].items():
        for method_name, reductions in asset_data['methods'].items():
            for red_level, red_data in reductions.items():
                # Use first repetition (run 1) for geometric accuracy
                rep = red_data['repetitions'][0]
                ga = rep.get('geometric_accuracy')
                if ga:
                    rows.append({
                        'asset': asset_name,
                        'method': method_name,
                        'reduction': red_level.replace('%', ''),
                        'hausdorff_norm': ga['hausdorff_distance_normalized'],
                        'rmse_norm': ga['rmse_normalized'],
                        'hausdorff_raw': ga['hausdorff_distance_raw'],
                        'rmse_raw': ga['rmse_raw'],
                    })
    return pd.DataFrame(rows)


# CORE ANALYSIS FUNCTIONS
def compute_win_rates(df, groupby_cols=None):
    """Compute win rates per method as wins/appearances (correct denominator).

    Each response is a pairwise comparison between two methods.
    A method's win rate = times chosen / times it appeared.

    Parameters
    ----------
    df : DataFrame of responses
    groupby_cols : optional list of columns to group by before computing

    Returns
    -------
    DataFrame with columns: method, wins, appearances, win_rate
        (plus groupby columns if specified)
    """
    if groupby_cols is None:
        groupby_cols = []

    results = []

    if groupby_cols:
        groups = df.groupby(groupby_cols, observed=True)
    else:
        groups = [("_all_", df)]

    for group_key, group_df in groups:
        if isinstance(group_key, str) and group_key == "_all_":
            group_vals = {}
        elif isinstance(group_key, tuple):
            group_vals = dict(zip(groupby_cols, group_key))
        else:
            group_vals = {groupby_cols[0]: group_key}

        for method in METHOD_ORDER:
            # Appearances: method was either chosen or not chosen
            appearances = len(group_df[
                (group_df['chosen_method'] == method) |
                (group_df['not_chosen_method'] == method)
            ])
            wins = len(group_df[group_df['chosen_method'] == method])

            if appearances > 0:
                row = {**group_vals, 'method': method,
                       'wins': wins, 'appearances': appearances,
                       'win_rate': wins / appearances}
                results.append(row)

    return pd.DataFrame(results)


def compute_agreement(df):
    """Compute distant/close agreement rate.

    For each (participant, pair) with both views, check if
    the same method was chosen at both distances.

    Returns
    -------
    dict with agreement_rate, n_agree, n_disagree, total_paired
    """
    agree = 0
    disagree = 0

    for (pid, pair_id), pair_data in df.groupby(['participant_id', 'pair_id']):
        if len(pair_data) == 2:
            distant = pair_data[pair_data['view_type'] == 'distant']
            close = pair_data[pair_data['view_type'] == 'close']
            if len(distant) == 1 and len(close) == 1:
                if distant['chosen_method'].values[0] == close['chosen_method'].values[0]:
                    agree += 1
                else:
                    disagree += 1

    total = agree + disagree
    return {
        'agreement_rate': agree / total if total > 0 else 0,
        'n_agree': agree,
        'n_disagree': disagree,
        'total_paired': total,
    }


def compute_reaction_time_stats(df):
    """Compute reaction time descriptive statistics (using median, IQR)."""
    rt = df['reaction_time_ms'] / 1000  # convert to seconds

    result = {}
    for view in ['distant', 'close']:
        sub = df[df['view_type'] == view]['reaction_time_ms'] / 1000
        result[view] = {
            'median': sub.median(),
            'mean': sub.mean(),
            'q1': sub.quantile(0.25),
            'q3': sub.quantile(0.75),
            'iqr': sub.quantile(0.75) - sub.quantile(0.25),
            'min': sub.min(),
            'max': sub.max(),
            'n_under_1s': (sub < 1).sum(),
            'n_over_60s': (sub > 60).sum(),
            'n': len(sub),
        }

    result['overall'] = {
        'median': rt.median(),
        'mean': rt.mean(),
        'q1': rt.quantile(0.25),
        'q3': rt.quantile(0.75),
        'n': len(rt),
        'n_under_1s': (rt < 1).sum(),
        'n_over_60s': (rt > 60).sum(),
    }
    return result


def compute_spearman_correlation(perceptual_df, geo_df):
    """Compute Spearman correlation between perceptual win rates and geometric accuracy.

    For each (asset, method, reduction) combination, we have:
    - Perceptual win rate (from pairwise comparisons)
    - Hausdorff distance (normalized)
    - RMSE (normalized)

    We expect: higher geometric error -> lower win rate (negative correlation).

    Returns dict with correlation results per view type and combined.
    """
    results = {}

    for view_label, view_df in [('combined', perceptual_df),
                                 ('distant', perceptual_df[perceptual_df['view_type'] == 'distant']),
                                 ('close', perceptual_df[perceptual_df['view_type'] == 'close'])]:

        wr = compute_win_rates(view_df, groupby_cols=['asset', 'reduction'])

        # Merge with geometric accuracy
        merged = wr.merge(geo_df, on=['asset', 'method', 'reduction'], how='inner')

        if len(merged) < 5:
            results[view_label] = {'n': len(merged), 'error': 'insufficient data'}
            continue

        # Spearman: win_rate vs hausdorff (expect negative: more error -> fewer wins)
        rho_h, p_h = stats.spearmanr(merged['win_rate'], merged['hausdorff_norm'])
        rho_r, p_r = stats.spearmanr(merged['win_rate'], merged['rmse_norm'])

        results[view_label] = {
            'n': len(merged),
            'hausdorff_rho': rho_h,
            'hausdorff_p': p_h,
            'rmse_rho': rho_r,
            'rmse_p': p_r,
        }

    return results


# FIGURE GENERATION
def _get_palette():
    return [METHOD_COLORS[m] for m in METHOD_ORDER]


def plot_overall_win_rates(df, output_dir):
    """Bar chart: overall win rate per method, split by viewing distance."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    wr_distant = compute_win_rates(df[df['view_type'] == 'distant'])
    wr_close = compute_win_rates(df[df['view_type'] == 'close'])

    x = np.arange(len(METHOD_ORDER))
    width = 0.35

    distant_rates = []
    close_rates = []
    for m in METHOD_ORDER:
        d_row = wr_distant[wr_distant['method'] == m]
        c_row = wr_close[wr_close['method'] == m]
        distant_rates.append(d_row['win_rate'].values[0] * 100 if len(d_row) > 0 else 0)
        close_rates.append(c_row['win_rate'].values[0] * 100 if len(c_row) > 0 else 0)

    bars_d = ax.bar(x - width / 2, distant_rates, width, label='Distant',
                    color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.7,
                    edgecolor='white', linewidth=0.5)
    bars_c = ax.bar(x + width / 2, close_rates, width, label='Close',
                    color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=1.0,
                    edgecolor='white', linewidth=0.5)

    # Add hatching to distant bars to distinguish from close
    for bar in bars_d:
        bar.set_hatch('///')

    # Value labels
    for bars in [bars_d, bars_c]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Perceptual Win Rates by Method and Viewing Distance')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS_FULL[m] for m in METHOD_ORDER])
    ax.legend(title='Viewing Distance')
    ax.set_ylim(0, max(max(distant_rates), max(close_rates)) * 1.15)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')

    plt.tight_layout()
    path = os.path.join(output_dir, f'perceptual_win_rates_by_distance.{FIG_FORMAT}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_win_rates_by_reduction(df, output_dir):
    """Grouped bar chart: win rates per method across reduction levels."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for idx, red in enumerate(REDUCTION_ORDER):
        ax = axes[idx]
        sub = df[df['reduction'] == red]
        wr = compute_win_rates(sub)

        rates = []
        for m in METHOD_ORDER:
            row = wr[wr['method'] == m]
            rates.append(row['win_rate'].values[0] * 100 if len(row) > 0 else 0)

        bars = ax.bar(range(len(METHOD_ORDER)), rates,
                      color=[METHOD_COLORS[m] for m in METHOD_ORDER],
                      edgecolor='white', linewidth=0.5)

        for bar, rate in zip(bars, rates):
            ax.annotate(f'{rate:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, rate),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{red}% Reduction')
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=30, ha='right')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

        if idx == 0:
            ax.set_ylabel('Win Rate (%)')

    axes[0].set_ylim(0, 75)
    fig.suptitle('Perceptual Win Rates by Reduction Level', fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'perceptual_win_rates_by_reduction.{FIG_FORMAT}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_win_rate_heatmap(df, output_dir):
    """Heatmap: win rate per method per asset (the key variation figure)."""
    wr = compute_win_rates(df, groupby_cols=['asset'])

    # Pivot to matrix
    pivot = wr.pivot(index='asset', columns='method', values='win_rate')
    pivot = pivot.reindex(index=ASSET_ORDER, columns=METHOD_ORDER) * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                vmin=0, vmax=75,
                linewidths=0.5, linecolor='white',
                yticklabels=[ASSET_LABELS.get(a, a) for a in ASSET_ORDER],
                xticklabels=[METHOD_LABELS_FULL[m] for m in METHOD_ORDER],
                cbar_kws={'label': 'Win Rate (%)', 'shrink': 0.8},
                ax=ax)

    ax.set_title('Perceptual Win Rates by Asset and Method')
    ax.set_ylabel('')
    ax.set_xlabel('')

    plt.tight_layout()
    path = os.path.join(output_dir, f'perceptual_heatmap_asset_method.{FIG_FORMAT}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_spearman_scatter(perceptual_df, geo_df, output_dir):
    """Scatter plots: win rate vs geometric error with Spearman rho annotation."""
    wr = compute_win_rates(perceptual_df, groupby_cols=['asset', 'reduction'])
    merged = wr.merge(geo_df, on=['asset', 'method', 'reduction'], how='inner')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (metric, metric_label) in enumerate([
        ('hausdorff_norm', 'Hausdorff Distance (normalized)'),
        ('rmse_norm', 'RMSE (normalized)')
    ]):
        ax = axes[idx]

        # Add small jitter to reveal overlapping points
        np.random.seed(42)
        x_range = merged[metric].max() - merged[metric].min()
        y_range = 100  # win rate range
        jitter_x = np.random.normal(0, x_range * 0.012, size=len(merged))
        jitter_y = np.random.normal(0, y_range * 0.008, size=len(merged))

        for method in METHOD_ORDER:
            mask = merged['method'] == method
            sub = merged[mask]
            jx = jitter_x[mask.values]
            jy = jitter_y[mask.values]
            ax.scatter(sub[metric].values + jx, sub['win_rate'].values * 100 + jy,
                       c=METHOD_COLORS[method], label=METHOD_LABELS_FULL[method],
                       s=35, alpha=0.7, edgecolors='white', linewidth=0.3)

        # Ideal trend line: what rho = -1.0 would look like
        # Pair sorted errors (ascending) with sorted win rates (descending)
        sorted_errors = np.sort(merged[metric].values)
        sorted_wr_desc = np.sort(merged['win_rate'].values * 100)[::-1]
        # Smooth with scipy UnivariateSpline (already a dependency via scipy.stats)
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(sorted_errors, sorted_wr_desc, s=len(sorted_errors) * 20)
        x_smooth = np.linspace(sorted_errors.min(), sorted_errors.max(), 200)
        ax.plot(x_smooth, spline(x_smooth),
                color='#D55E00', linestyle='--', linewidth=2.0, alpha=0.7,
                label='_nolegend_', zorder=1)

        # Spearman annotation
        rho, p = stats.spearmanr(merged[metric], merged['win_rate'])
        p_str = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        ax.text(0.97, 0.97, f'\u03c1 = {rho:.3f}\n{p_str}\nn = {len(merged)}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', edgecolor='gray', alpha=0.8))

        ax.set_xlabel(metric_label)
        ax.set_ylabel('Win Rate (%)' if idx == 0 else '')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.4)

    # Legend below the figure, outside plot area
    handles, labels = axes[0].get_legend_handles_labels()
    # Add ideal line entry
    from matplotlib.lines import Line2D
    ideal_handle = Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.0, alpha=0.7)
    handles.append(ideal_handle)
    labels.append('Ideal (\u03c1 = \u22121.0)')
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Perceptual Quality vs. Geometric Accuracy', fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(output_dir, f'perceptual_vs_geometric_scatter.{FIG_FORMAT}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_spearman_by_distance(perceptual_df, geo_df, output_dir):
    """Scatter: win rate vs geometric error, separate panels for distant/close."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row_idx, view in enumerate(['distant', 'close']):
        view_df = perceptual_df[perceptual_df['view_type'] == view]
        wr = compute_win_rates(view_df, groupby_cols=['asset', 'reduction'])
        merged = wr.merge(geo_df, on=['asset', 'method', 'reduction'], how='inner')

        for col_idx, (metric, metric_label) in enumerate([
            ('hausdorff_norm', 'Hausdorff Distance (normalized)'),
            ('rmse_norm', 'RMSE (normalized)')
        ]):
            ax = axes[row_idx][col_idx]

            # Add small jitter to reveal overlapping points
            np.random.seed(42 + row_idx * 10 + col_idx)
            x_range = merged[metric].max() - merged[metric].min()
            y_range = 100
            jitter_x = np.random.normal(0, x_range * 0.012, size=len(merged))
            jitter_y = np.random.normal(0, y_range * 0.008, size=len(merged))

            for method in METHOD_ORDER:
                mask = merged['method'] == method
                sub = merged[mask]
                jx = jitter_x[mask.values]
                jy = jitter_y[mask.values]
                ax.scatter(sub[metric].values + jx, sub['win_rate'].values * 100 + jy,
                           c=METHOD_COLORS[method], label=METHOD_LABELS_FULL[method],
                           s=30, alpha=0.7, edgecolors='white', linewidth=0.3)

            # Ideal trend line: what rho = -1.0 would look like
            sorted_errors = np.sort(merged[metric].values)
            sorted_wr_desc = np.sort(merged['win_rate'].values * 100)[::-1]
            from scipy.interpolate import UnivariateSpline
            spline = UnivariateSpline(sorted_errors, sorted_wr_desc, s=len(sorted_errors) * 20)
            x_smooth = np.linspace(sorted_errors.min(), sorted_errors.max(), 200)
            ax.plot(x_smooth, spline(x_smooth),
                    color='#D55E00', linestyle='--', linewidth=2.0, alpha=0.7,
                    label='_nolegend_', zorder=1)

            rho, p = stats.spearmanr(merged[metric], merged['win_rate'])
            p_str = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            ax.text(0.97, 0.97, f'\u03c1 = {rho:.3f}\n{p_str}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white', edgecolor='gray', alpha=0.8))

            ax.set_title(f'{view.capitalize()}: {metric_label}', fontsize=10)
            ax.axhline(y=50, color='gray', linestyle=':', alpha=0.4)
            if col_idx == 0:
                ax.set_ylabel('Win Rate (%)')
            if row_idx == 1:
                ax.set_xlabel(metric_label)

    # Legend below the figure, outside plot area
    handles, labels = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    ideal_handle = Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.0, alpha=0.7)
    handles.append(ideal_handle)
    labels.append('Ideal (\u03c1 = \u22121.0)')
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Perceptual Quality vs. Geometric Accuracy by Viewing Distance',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(output_dir, f'perceptual_vs_geometric_by_distance.{FIG_FORMAT}')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# TABLE RENDERING (matches benchmark script)
def render_table_image(df_table, title, output_path, col_widths=None):
    """Render a DataFrame as a styled PNG table image."""
    n_rows, n_cols = df_table.shape

    # Measure column widths from content (header + data)
    char_width = 0.09  # approximate width per character in inches at fontsize 9
    min_col_width = 1.0
    padding = 0.6  # extra space per column for padding

    col_pixel_widths = []
    has_row_labels = df_table.index.name or not df_table.index.equals(
        pd.RangeIndex(len(df_table)))

    for col_idx, col_name in enumerate(df_table.columns):
        max_len = len(str(col_name))
        for val in df_table.iloc[:, col_idx]:
            max_len = max(max_len, len(str(val)))
        col_pixel_widths.append(max(min_col_width, max_len * char_width + padding))

    # Add row label column width if present
    row_label_width = 0
    if has_row_labels:
        max_row_len = max(len(str(idx)) for idx in df_table.index)
        row_label_width = max(min_col_width, max_row_len * char_width + padding)

    fig_width = sum(col_pixel_widths) + row_label_width + 0.5
    fig_height = max(2, (n_rows + 1) * 0.4 + 1.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=12, fontweight='bold')

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        rowLabels=df_table.index if has_row_labels else None,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Auto-fit column widths
    table.auto_set_column_width(list(range(n_cols)))

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F8F9FA')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#DEE2E6')

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_summary_table(df, output_dir):
    """Create overall perceptual results summary table (PNG)."""
    wr_all = compute_win_rates(df)
    wr_distant = compute_win_rates(df[df['view_type'] == 'distant'])
    wr_close = compute_win_rates(df[df['view_type'] == 'close'])

    rows = []
    for m in METHOD_ORDER:
        all_row = wr_all[wr_all['method'] == m].iloc[0]
        d_row = wr_distant[wr_distant['method'] == m].iloc[0]
        c_row = wr_close[wr_close['method'] == m].iloc[0]
        rows.append({
            'Method': METHOD_LABELS_FULL[m],
            'Overall Win Rate': f"{all_row['win_rate']*100:.1f}%",
            'Overall Wins': f"{int(all_row['wins'])}/{int(all_row['appearances'])}",
            'Distant Win Rate': f"{d_row['win_rate']*100:.1f}%",
            'Close Win Rate': f"{c_row['win_rate']*100:.1f}%",
            'Delta (Close - Distant)': f"{(c_row['win_rate'] - d_row['win_rate'])*100:+.1f}pp",
        })

    table_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'table_perceptual_summary.{FIG_FORMAT}')
    render_table_image(table_df, 'Perceptual Quality: Overall Win Rates', path)
    return path


def create_reduction_table(df, output_dir):
    """Create win rates by reduction level table (PNG)."""
    rows = []
    for red in REDUCTION_ORDER:
        sub = df[df['reduction'] == red]
        wr = compute_win_rates(sub)
        row = {'Reduction': f'{red}%'}
        for m in METHOD_ORDER:
            m_row = wr[wr['method'] == m]
            if len(m_row) > 0:
                row[METHOD_LABELS_FULL[m]] = f"{m_row.iloc[0]['win_rate']*100:.1f}%"
            else:
                row[METHOD_LABELS_FULL[m]] = "--"
        rows.append(row)

    table_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'table_perceptual_by_reduction.{FIG_FORMAT}')
    render_table_image(table_df, 'Win Rates by Reduction Level', path)
    return path


def create_spearman_table(corr_results, output_dir):
    """Create Spearman correlation summary table (PNG)."""
    rows = []
    for view in ['combined', 'distant', 'close']:
        r = corr_results[view]
        if 'error' in r:
            rows.append({
                'View': view.capitalize(),
                'n': r['n'],
                'Hausdorff rho': '--', 'Hausdorff p': '--',
                'RMSE rho': '--', 'RMSE p': '--',
            })
        else:
            rows.append({
                'View': view.capitalize(),
                'n': r['n'],
                'Hausdorff rho': f"{r['hausdorff_rho']:.3f}",
                'Hausdorff p': f"<0.001" if r['hausdorff_p'] < 0.001 else f"{r['hausdorff_p']:.3f}",
                'RMSE rho': f"{r['rmse_rho']:.3f}",
                'RMSE p': f"<0.001" if r['rmse_p'] < 0.001 else f"{r['rmse_p']:.3f}",
            })

    table_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'table_spearman_correlation.{FIG_FORMAT}')
    render_table_image(table_df,
                       'Spearman Correlation: Perceptual Win Rate vs. Geometric Error',
                       path)
    return path


def create_study_overview_table(df, output_dir):
    """Create study parameters overview table (PNG)."""
    counts = Counter(df.groupby('participant_id').size())
    n_complete = counts.get(20, 0)
    n_partial = sum(v for k, v in counts.items() if k < 20)

    rows = [
        {'Parameter': 'Total participants', 'Value': str(df['participant_id'].nunique())},
        {'Parameter': 'Complete sessions (20 comparisons)', 'Value': str(n_complete)},
        {'Parameter': 'Partial sessions', 'Value': str(n_partial)},
        {'Parameter': 'Total responses', 'Value': str(len(df))},
        {'Parameter': 'Unique pairwise comparisons', 'Value': str(df['pair_id'].nunique())},
        {'Parameter': 'Assets evaluated', 'Value': str(df['asset'].nunique())},
        {'Parameter': 'Reduction levels', 'Value': '50%, 80%, 90%'},
        {'Parameter': 'Methods compared', 'Value': str(len(METHOD_ORDER))},
        {'Parameter': 'Viewing conditions', 'Value': 'Distant, Close'},
        {'Parameter': 'Evaluations per pair (median)', 'Value': str(int(
            df.groupby(['pair_id', 'view_type']).size().median()))},
    ]

    table_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'table_study_overview.{FIG_FORMAT}')
    render_table_image(table_df, 'Perceptual Study Parameters', path)
    return path


# TEXT REPORT
def generate_text_report(df, geo_df, corr_results, agreement, rt_stats, output_dir):
    """Generate a structured text report of all analysis results."""
    path = os.path.join(output_dir, 'perceptual_analysis_report.txt')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("PERCEPTUAL QUALITY ASSESSMENT -- ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Study overview
        counts = Counter(df.groupby('participant_id').size())
        n_complete = counts.get(20, 0)
        n_partial = sum(v for k, v in counts.items() if k < 20)

        f.write("STUDY OVERVIEW\n")
        f.write(f"  Participants: {df['participant_id'].nunique()} "
                f"({n_complete} complete, {n_partial} partial)\n")
        f.write(f"  Total responses: {len(df)}\n")
        f.write(f"  Unique pairs: {df['pair_id'].nunique()}/144 expected\n")
        f.write(f"  Evaluations per pair: median={int(df.groupby(['pair_id','view_type']).size().median())}, "
                f"range={df.groupby(['pair_id','view_type']).size().min()}-"
                f"{df.groupby(['pair_id','view_type']).size().max()}\n\n")

        # Overall win rates
        f.write("OVERALL WIN RATES (wins/appearances)\n")
        f.write("-" * 50 + "\n")
        wr = compute_win_rates(df)
        for m in METHOD_ORDER:
            row = wr[wr['method'] == m].iloc[0]
            f.write(f"  {METHOD_LABELS_FULL[m]:<22s} "
                    f"{row['win_rate']*100:5.1f}%  "
                    f"({int(row['wins'])}/{int(row['appearances'])})\n")

        # By distance
        f.write("\nWIN RATES BY VIEWING DISTANCE\n")
        f.write("-" * 50 + "\n")
        for view in VIEW_ORDER:
            wr_v = compute_win_rates(df[df['view_type'] == view])
            f.write(f"\n  {view.upper()}:\n")
            for m in METHOD_ORDER:
                row = wr_v[wr_v['method'] == m].iloc[0]
                f.write(f"    {METHOD_LABELS_FULL[m]:<22s} {row['win_rate']*100:5.1f}%\n")

        # Agreement
        f.write(f"\nDISTANT/CLOSE AGREEMENT\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Agreement rate: {agreement['agreement_rate']:.1%}\n")
        f.write(f"  Consistent: {agreement['n_agree']}/{agreement['total_paired']}\n")
        f.write(f"  Inconsistent: {agreement['n_disagree']}/{agreement['total_paired']}\n\n")

        # By reduction
        f.write("WIN RATES BY REDUCTION LEVEL\n")
        f.write("-" * 50 + "\n")
        for red in REDUCTION_ORDER:
            sub = df[df['reduction'] == red]
            wr_r = compute_win_rates(sub)
            f.write(f"\n  {red}% Reduction:\n")
            for m in METHOD_ORDER:
                row = wr_r[wr_r['method'] == m].iloc[0]
                f.write(f"    {METHOD_LABELS_FULL[m]:<22s} {row['win_rate']*100:5.1f}%\n")

        # By asset
        f.write("\nWIN RATES BY ASSET\n")
        f.write("-" * 50 + "\n")
        for asset in ASSET_ORDER:
            sub = df[df['asset'] == asset]
            wr_a = compute_win_rates(sub)
            f.write(f"\n  {ASSET_LABELS.get(asset, asset)}:\n")
            for m in METHOD_ORDER:
                row = wr_a[wr_a['method'] == m]
                if len(row) > 0:
                    f.write(f"    {METHOD_LABELS_FULL[m]:<22s} {row.iloc[0]['win_rate']*100:5.1f}%\n")

        # Reaction times
        f.write(f"\nREACTION TIMES\n")
        f.write("-" * 50 + "\n")
        for view in ['distant', 'close']:
            r = rt_stats[view]
            f.write(f"  {view.capitalize():8s}: median={r['median']:.1f}s, "
                    f"IQR=[{r['q1']:.1f}, {r['q3']:.1f}]s, "
                    f"range=[{r['min']:.1f}, {r['max']:.1f}]s\n")
        r = rt_stats['overall']
        f.write(f"  {'Overall':8s}: median={r['median']:.1f}s\n")
        f.write(f"  Responses <1s: {r['n_under_1s']}\n")
        f.write(f"  Responses >60s: {r['n_over_60s']}\n\n")

        # Spearman
        f.write("SPEARMAN CORRELATION (Win Rate vs. Geometric Error)\n")
        f.write("-" * 50 + "\n")
        if not corr_results:
            f.write("  No matching geometric data available.\n\n")
        else:
            for view in ['combined', 'distant', 'close']:
                r = corr_results[view]
                if 'error' in r:
                    f.write(f"  {view.capitalize()}: insufficient data (n={r['n']})\n")
                else:
                    h_p = '<0.001' if r['hausdorff_p'] < 0.001 else f"{r['hausdorff_p']:.3f}"
                    r_p = '<0.001' if r['rmse_p'] < 0.001 else f"{r['rmse_p']:.3f}"
                    f.write(f"  {view.capitalize()} (n={r['n']}):\n")
                    f.write(f"    Hausdorff: rho={r['hausdorff_rho']:.3f}, p={h_p}\n")
                    f.write(f"    RMSE:      rho={r['rmse_rho']:.3f}, p={r_p}\n")

        f.write(f"\nInterpretation guide:\n")
        f.write(f"  |rho| > 0.7: strong correlation\n")
        f.write(f"  |rho| 0.4-0.7: moderate correlation\n")
        f.write(f"  |rho| < 0.4: weak correlation\n")
        f.write(f"  Negative rho expected: higher error -> lower win rate\n")

    print(f"  Saved: {path}")
    return path


# CSV EXPORT
def export_csvs(df, geo_df, output_dir):
    """Export analysis data as CSVs for reproducibility."""
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    # Overall win rates
    wr = compute_win_rates(df)
    wr.to_csv(os.path.join(csv_dir, 'win_rates_overall.csv'), index=False)

    # By distance
    for view in VIEW_ORDER:
        wr_v = compute_win_rates(df[df['view_type'] == view])
        wr_v.to_csv(os.path.join(csv_dir, f'win_rates_{view}.csv'), index=False)

    # By asset
    wr_asset = compute_win_rates(df, groupby_cols=['asset'])
    wr_asset.to_csv(os.path.join(csv_dir, 'win_rates_by_asset.csv'), index=False)

    # By reduction
    wr_red = compute_win_rates(df, groupby_cols=['reduction'])
    wr_red.to_csv(os.path.join(csv_dir, 'win_rates_by_reduction.csv'), index=False)

    # Full granular: asset x reduction x method
    wr_full = compute_win_rates(df, groupby_cols=['asset', 'reduction'])
    wr_full.to_csv(os.path.join(csv_dir, 'win_rates_full_granular.csv'), index=False)

    # Merged with geometric accuracy (if available)
    if geo_df is not None and len(geo_df) > 0:
        merged = wr_full.merge(geo_df, on=['asset', 'method', 'reduction'], how='inner')
        merged.to_csv(os.path.join(csv_dir, 'perceptual_vs_geometric.csv'), index=False)

    print(f"  Saved CSVs to: {csv_dir}/")


# MAIN
def main(perceptual_json=PERCEPTUAL_JSON, benchmark_json=BENCHMARK_JSON,
         output_dir=OUTPUT_DIR, dpi=300):
    """Run complete perceptual analysis pipeline."""
    global FIG_DPI
    FIG_DPI = dpi

    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PERCEPTUAL QUALITY ANALYSIS -- FIGURE & TABLE GENERATION")
    print("=" * 60)
    print()

    # Load data
    print("[1/7] Loading data...")
    df = load_perceptual(perceptual_json)
    geo_df = load_geometric_accuracy(benchmark_json)

    # Filter orderings to methods/assets actually present in the data
    global METHOD_ORDER, ASSET_ORDER
    present_methods = set(df['chosen_method'].unique()) | set(df['not_chosen_method'].unique())
    METHOD_ORDER = [m for m in METHOD_ORDER if m in present_methods]
    present_assets = set(df['asset'].unique())
    ASSET_ORDER = [a for a in ASSET_ORDER if a in present_assets]
    # Add any assets not in the predefined order
    for a in sorted(present_assets):
        if a not in ASSET_ORDER:
            ASSET_ORDER.append(a)

    has_geo = len(geo_df) > 0
    print(f"  Perceptual: {len(df)} responses, "
          f"{df['participant_id'].nunique()} participants")
    print(f"  Geometric:  {len(geo_df)} measurements"
          f"{'' if has_geo else ' (skipping correlation analysis)'}")
    print()

    # Compute core analysis
    print("[2/7] Computing analysis...")
    agreement = compute_agreement(df)
    rt_stats = compute_reaction_time_stats(df)
    corr_results = compute_spearman_correlation(df, geo_df) if has_geo else None
    print(f"  Agreement rate: {agreement['agreement_rate']:.1%}")
    if corr_results:
        print(f"  Spearman (combined, Hausdorff): "
              f"rho={corr_results['combined']['hausdorff_rho']:.3f}")
        print(f"  Spearman (combined, RMSE):      "
              f"rho={corr_results['combined']['rmse_rho']:.3f}")
    print()

    # Generate figures
    print("[3/7] Generating figures...")
    plot_overall_win_rates(df, output_dir)
    plot_win_rates_by_reduction(df, output_dir)
    plot_win_rate_heatmap(df, output_dir)
    if has_geo:
        plot_spearman_scatter(df, geo_df, output_dir)
        plot_spearman_by_distance(df, geo_df, output_dir)
    print()

    # Generate tables
    print("[4/7] Generating tables...")
    create_study_overview_table(df, output_dir)
    create_summary_table(df, output_dir)
    create_reduction_table(df, output_dir)
    if corr_results:
        create_spearman_table(corr_results, output_dir)
    print()

    # Text report
    print("[5/7] Generating text report...")
    generate_text_report(df, geo_df if has_geo else None,
                         corr_results, agreement, rt_stats, output_dir)
    print()

    # CSV export
    print("[6/7] Exporting CSVs...")
    export_csvs(df, geo_df if has_geo else None, output_dir)
    print()

    # Summary
    print("[7/7] Done!")
    print()
    print("=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Directory: {output_dir}/")
    print()
    print("Figures (for thesis):")
    print("  perceptual_win_rates_by_distance.png -- Main win rate comparison")
    print("  perceptual_win_rates_by_reduction.png -- Reduction level trends")
    print("  perceptual_heatmap_asset_method.png   -- Asset-level variation")
    print("  perceptual_vs_geometric_scatter.png   -- Spearman correlation (combined)")
    print("  perceptual_vs_geometric_by_distance.png -- Spearman by view distance")
    print()
    print("Tables (for thesis):")
    print("  table_study_overview.png              -- Study parameters")
    print("  table_perceptual_summary.png          -- Overall results")
    print("  table_perceptual_by_reduction.png     -- Reduction-level breakdown")
    print("  table_spearman_correlation.png        -- Correlation summary")
    print()
    print("Data (for reproducibility):")
    print("  perceptual_analysis_report.txt        -- Full text report")
    print("  csv/                                  -- All raw analysis data")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perceptual quality analysis')
    parser.add_argument('--perceptual', default=PERCEPTUAL_JSON)
    parser.add_argument('--benchmark', default=BENCHMARK_JSON)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    main(args.perceptual, args.benchmark, args.output, args.dpi)