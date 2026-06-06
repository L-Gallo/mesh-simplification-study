# Data

Raw research data in JSON format. These files are inputs to the analysis
scripts (`analyze_benchmarks.py`, `analyze_perceptual.py`,
`analyze_scalability.py`).

## batch_report.json

Benchmark results for the Stanford 3D Scanning Repository models
(scalability analysis, single repetition, with geometric accuracy).

Structure:

```
system_info          Hardware and library versions
summary              Totals: 4 assets, 48 tests, success/stability rates
method_statistics    Per-method aggregates (time, memory, accuracy)
assets               Per-asset results:
  {asset_name}
    original_metrics   Vertex/face count, bounding box, file size
    methods
      {method}
        {reduction%}
          repetitions    Array of run results, each containing:
            performance          time (s/ms), memory (MB)
            geometric_accuracy   Hausdorff, RMSE (raw + normalized), sample count
            actual_reduction_ratio
          statistics     Mean, std, CV, stability flag
```

Assets: armadillo (345K faces), bunny (69K), dragon (871K), lucy (28M).
Methods: fast-simplification, open3d, meshoptimizer, cgal.
Reduction levels: 50%, 80%, 90%.

## participant_responses.json

Anonymized responses from the pairwise comparison perceptual study
(2769 judgments, 150 participants, collected March 4-13 2026).

Each entry:

```json
{
  "participant_id": "p_1772610881528_kyvn77uh1",
  "pair_id": "LAV_90_cgal_vs_fast-simplification",
  "view_type": "close",
  "chosen_side": "right",
  "chosen_method": "fast-simplification",
  "not_chosen_method": "cgal",
  "reaction_time_ms": 6453,
  "order_in_pair": 1,
  "timestamp": "2026-03-04T07:54:47.327914"
}
```

Fields:
- `participant_id` -- random ID, no personal information collected
- `pair_id` -- format: `{model}_{reduction}_{methodA}_vs_{methodB}`
- `view_type` -- `distant` (scaled down) or `close` (full size)
- `chosen_side` -- `left` or `right` (position was randomized)
- `chosen_method` / `not_chosen_method` -- which method won/lost
- `reaction_time_ms` -- time from image display to choice
- `order_in_pair` -- 1 or 2 (whether this was the first or second view)

Models: AK74, bunker, church, jeep, LAV, M9_pistol, Mi8, watermill
(Arma Reforger production assets, provided by Bohemia Interactive).

## What's not included

The industry benchmark data (Arma Reforger assets tested with 3 repetitions
and geometric accuracy) is not published due to NDA restrictions. The
benchmark pipeline can be run on any OBJ meshes to produce equivalent data.
The Stanford models in `test_meshes/` and the full set from the Stanford
3D Scanning Repository can be used for replication.