# Quick Start

## 1. Setup

```bash
git clone https://github.com/nshrhm/llm-emotional-individuality.git
cd llm-emotional-individuality
pip install -r requirements.txt
```

## 2. Verify the bundled public dataset

```bash
python verify_results.py
```

This checks the current `data_public.csv` and generated CSV outputs against the baseline values in `src/config.py:PAPER_VALUES`.

## 3. Regenerate the analysis outputs

```bash
python main.py
```

This regenerates:

- `results/csv/`
- `results/json/`
- `results/figures/`
- `results/tables/`

## 4. Run a smaller check if needed

```bash
python main.py --no-viz
python main.py --section 3.1
python main.py --section 3.2
python main.py --section 3.3
python main.py --section 3.4
python main.py --section 3.5
```

## 5. Reviewer-relevant files

- `results/json/verification_report.json`
- `results/csv/model_reliability.csv`
- `results/csv/temperature_correlation.csv`
- `results/csv/model_entropy_profiles.csv`
- `results/csv/quadrant_threshold_sensitivity.csv`
- `results/csv/entropy_interpretation_anchors.csv`
- `results/csv/entropy_reliability_robustness.csv`
- `results/csv/pca_loadings_pc1_pc3.csv`
- `results/figures/figure1_reliability_controllability_matrix.pdf`
- `results/figures/figure2_temperature_entropy_relationship.pdf`
- `results/figures/figure2b_temperature_variance_profiles_representative.pdf`

## Notes

- The public workflow defaults to `data_public.csv`.
- The repository does not include manuscript or rebuttal files.
- The repository does not include the full dataset with free-text justifications.
