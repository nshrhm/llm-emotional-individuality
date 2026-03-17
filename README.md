# Mathematical Framework for Characterizing Emotional Individuality in Large Language Models

This repository is the **public reproducibility package** for the paper:

**"Mathematical Framework for Characterizing Emotional Individuality in Large Language Models: Temperature Control, Fuzzy Entropy, and Persona-Based Diversity Analysis"**

It is intentionally scoped to **analysis-level reproducibility**. The repository contains the analysis code, the numeric-only public dataset, and the generated reviewer-facing outputs cited by the revised paper.

## Included

- `data_public.csv`: numeric-only dataset (metadata + 0-100 emotion scores)
- `src/`, `main.py`, `verify_results.py`: analysis and verification code
- `results/`: curated generated CSV, JSON, figure, and table outputs
- `QUICKSTART.md`, `OUTPUTS.md`, `REPRODUCIBILITY_NOTES.md`: public reproducibility docs

## Not Included

- Manuscript sources, PDFs, rebuttal files, or other review materials
- The full dataset with LLM-generated free-text justifications (`Q1reason`--`Q4reason`)
- The verbatim provider-specific prompt template used during the original collection window

The public package therefore supports regeneration and verification of the reported **numeric results and figures**, but does **not** claim literal prompt replay.

## Repository Layout

```text
llm-emotional-individuality/
├── data_public.csv
├── src/
├── main.py
├── verify_results.py
├── QUICKSTART.md
├── OUTPUTS.md
├── REPRODUCIBILITY_NOTES.md
├── results/
│   ├── csv/
│   ├── json/
│   ├── figures/
│   └── tables/
├── requirements.txt
├── LICENSE
└── LICENSE-DATA.md
```

## Quick Reproduction

```bash
git clone https://github.com/nshrhm/llm-emotional-individuality.git
cd llm-emotional-individuality
pip install -r requirements.txt
python verify_results.py
python main.py
```

By default, both commands use `data_public.csv`.

## Public Dataset Boundary

`data_public.csv` contains:

- Metadata: `timestamp`, `text`, `developer`, `model`, `persona`, `temperature`, `trial`
- Scores: `Q1value`, `Q2value`, `Q3value`, `Q4value`

The analysis code tolerates missing justification columns and does not require unpublished free-text fields.

## Revision-Support Outputs

The revised paper added a small set of reviewer-facing support artifacts that are now generated directly by the public repo:

- `results/csv/quadrant_threshold_sensitivity.csv`
- `results/csv/quadrant_threshold_reassignments.csv`
- `results/csv/temperature_variance_profiles_representative.csv`
- `results/csv/entropy_interpretation_anchors.csv`
- `results/csv/entropy_reliability_robustness.csv`
- `results/csv/pca_loadings_pc1_pc3.csv`
- `results/figures/figure2b_temperature_variance_profiles_representative.png`
- `results/figures/figure2b_temperature_variance_profiles_representative.pdf`

These support the revised appendix analyses on threshold sensitivity, entropy interpretation, entropy-reliability robustness, PCA loadings, and representative temperature-variance behavior.

## Documentation

- `QUICKSTART.md`: exact commands for reviewers
- `OUTPUTS.md`: output inventory and purpose
- `REPRODUCIBILITY_NOTES.md`: scope, limits, and data boundary

## Licenses

- Code: MIT (`LICENSE`)
- Data: CC BY 4.0 (`LICENSE-DATA.md`)
