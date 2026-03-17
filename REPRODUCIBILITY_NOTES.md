# Reproducibility Notes

## What this public repository reproduces

This repository supports **analysis-level reproducibility** of the revised paper:

- the numeric dataset used by the public analysis package (`data_public.csv`)
- the Python analysis code under `src/`
- regeneration of the released CSV, figure, table, and JSON outputs
- verification against the paper baselines via `python verify_results.py`
- the reviewer-facing revision-support artifacts added during peer review

## What this repository does not reproduce

The public package does **not** include:

- manuscript sources or PDFs
- rebuttal files
- the full dataset with LLM-generated free-text justifications
- the verbatim provider-specific prompt template used during the original collection window

Because that prompt template was not versioned publicly, this package does **not** claim literal prompt replay. Its purpose is to support verification and regeneration of the reported numeric analyses and visualizations.

## Minimal command sequence

```bash
pip install -r requirements.txt
python verify_results.py
python main.py
```

## Public dataset boundary

- `data_public.csv` contains metadata plus `Q1value`-`Q4value`
- justification columns (`Q1reason`-`Q4reason`) are intentionally absent from the public package
- the code path used here does not require those unpublished free-text fields

## Revision-support artifacts

The revised paper added the following public-facing support outputs:

- `results/csv/quadrant_threshold_sensitivity.csv`
- `results/csv/quadrant_threshold_reassignments.csv`
- `results/csv/temperature_variance_profiles_representative.csv`
- `results/csv/entropy_interpretation_anchors.csv`
- `results/csv/entropy_reliability_robustness.csv`
- `results/csv/pca_loadings_pc1_pc3.csv`
- `results/figures/figure2b_temperature_variance_profiles_representative.(png|pdf)`

These are now generated directly from the public repository code.
