# Outputs

All generated artifacts are written under `results/`.

## CSV Outputs

Core analysis exports:

- `results/csv/model_reliability.csv`
- `results/csv/temperature_correlation.csv`
- `results/csv/vendor_temperature_response.csv`
- `results/csv/temperature_variance_profiles.csv`
- `results/csv/model_entropy_profiles.csv`
- `results/csv/model_temperature_entropy_profiles.csv`
- `results/csv/model_sizes_inferred.csv`
- `results/csv/model_scale_entropy_correlation.csv`
- `results/csv/vendor_fuzzy_chi2_tests.csv`
- `results/csv/persona_effect_sizes.csv`
- `results/csv/persona_chameleon_metrics.csv`
- `results/csv/poet_robot_mannwhitney.csv`
- `results/csv/persona_temperature_rank_correlation.csv`
- `results/csv/persona_temperature_synergy_pearson.csv`
- `results/csv/vendor_persona_anova_typ2.csv`
- `results/csv/consistency_scores.csv`
- `results/csv/clustering_stability_ari.csv`
- `results/csv/clustering_stability_ari_summary.csv`
- `results/csv/cross_text_consistency.csv`
- `results/csv/within_model_variance.csv`
- `results/csv/text_effects_anova.csv`
- `results/csv/text_effects_kruskal_wallis.csv`
- `results/csv/text_correlation_pairs.csv`
- `results/csv/temperature_entropy_correlation_high_kappa.csv`

Revision-support exports added for the revised paper:

- `results/csv/quadrant_threshold_sensitivity.csv`
  Threshold grid summary for the reliability-controllability quadrants.
- `results/csv/quadrant_threshold_reassignments.csv`
  Model-level quadrant changes under nearby threshold choices.
- `results/csv/temperature_variance_profiles_representative.csv`
  Representative positive, negative, and near-flat temperature-variance profiles.
- `results/csv/entropy_interpretation_anchors.csv`
  Theoretical and observed anchors for interpreting the entropy scale.
- `results/csv/entropy_reliability_robustness.csv`
  Pearson and Spearman robustness summary for entropy vs. reliability.
- `results/csv/pca_loadings_pc1_pc3.csv`
  Long-format PC1-PC3 loadings with explained variance.

## Figures

- `results/figures/figure1_reliability_controllability_matrix.(png|pdf)`
- `results/figures/figure2_temperature_entropy_relationship.(png|pdf)`
- `results/figures/figure2b_temperature_variance_profiles_representative.(png|pdf)`
- `results/figures/figure3_vendor_fuzzy_distributions.(png|pdf)`
- `results/figures/figure4_entropy_distribution_violin.(png|pdf)`
- `results/figures/figure5_tsne_clustering.(png|pdf)`
- `results/figures/figure6_correlation_matrices.(png|pdf)`

## Tables

- `results/tables/table2_model_reliability.(csv|tex)`
- `results/tables/table3_temperature_correlation.(csv|tex)`
- `results/tables/table6_fuzzy_entropy.(csv|tex)`

## JSON

- `results/json/analysis_metadata.json`
- `results/json/analysis_summary.json`
- `results/json/verification_report.json`

## Scope

These outputs are derived from the public numeric dataset and are intended to reproduce the reported analysis artifacts, not manuscript files.
