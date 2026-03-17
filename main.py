"""
Main analysis pipeline for LLM Emotional Individuality research.

This script orchestrates the complete analysis pipeline from data loading
through all analytical sections to visualization and result export.

Usage:
    python main.py                    # Run complete analysis
    python main.py --section 3.1      # Run specific section only
    python main.py --no-viz           # Skip visualizations
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CSV_OUTPUT, JSON_OUTPUT
from src.data_loader import load_and_validate_data, load_raw_data
from src.temperature_analysis import TemperatureAnalyzer
from src.fuzzy_analysis import FuzzyAnalyzer
from src.persona_analysis import PersonaAnalyzer
from src.clustering_analysis import ClusteringAnalyzer
from src.correlation_analysis import CorrelationAnalyzer
from src.revision_support import build_revision_support_exports
from src.visualization import Visualizer
from src.utils import export_to_json, export_to_csv


def run_complete_analysis(skip_visualization=False, data_path: str | None = None):
    """Run complete analysis pipeline.

    Args:
        skip_visualization: If True, skip visualization generation
        data_path: Optional path to input CSV (default: data_public.csv)

    Returns:
        Dictionary with all results
    """
    print("=" * 70)
    print("Mathematical Framework for LLM Emotional Individuality Analysis")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading and validating data...")
    data_path_kw = {} if data_path is None else {"data_path": Path(data_path)}
    df, validation_report = load_and_validate_data(**data_path_kw)
    raw_df = load_raw_data(**data_path_kw)  # Load raw data for reliability analysis

    print(f"  ✓ Loaded {len(df)} valid data points")
    missing_rate = validation_report["overall_missing_rate"]
    valid_rate = validation_report.get("valid_score_rate", 1 - missing_rate)
    print(f"  ✓ Valid-score rate (valid/collected): {valid_rate:.1%}")
    print(f"  ✓ Missing-data rate (missing/collected): {missing_rate:.1%}")

    # Section 3.1: Temperature Analysis
    # Note: Use raw_df for reliability (MDR) calculation, df for temperature correlation
    print("\n[2/7] Section 3.1: Temperature-Controlled Response Characteristics")
    temp_analyzer = TemperatureAnalyzer(raw_df, df)  # Pass both raw and valid data
    temp_results = temp_analyzer.run_all_analyses()
    temp_summary = temp_analyzer.get_summary_statistics()

    print(f"  ✓ Temperature correlation range: "
          f"{temp_summary['temperature_correlation']['correlation_range']}")

    # Section 3.2: Fuzzy Analysis
    print("\n[3/7] Section 3.2: Fuzzy-Theoretic Emotional Characterization")
    fuzzy_analyzer = FuzzyAnalyzer(df)
    fuzzy_results = fuzzy_analyzer.run_all_analyses()
    fuzzy_summary = fuzzy_analyzer.get_summary_statistics()

    print(f"  ✓ Entropy range: "
          f"{fuzzy_summary['entropy_range']['min']:.2f} - "
          f"{fuzzy_summary['entropy_range']['max']:.2f}")

    # Section 3.3: Persona Analysis
    print("\n[4/7] Section 3.3: Persona-Based Cognitive Diversity")
    persona_analyzer = PersonaAnalyzer(df)
    persona_results = persona_analyzer.run_all_analyses()
    persona_summary = persona_analyzer.get_summary_statistics()

    if 'persona_temperature_synergy' in persona_summary:
        pts = persona_summary['persona_temperature_synergy']
        print(f"  ✓ Persona-temperature synergy: r={pts['r']:.3f}, p={pts['p_value']:.4f}")

    # Section 3.4: Clustering Analysis
    print("\n[5/7] Section 3.4: Model Clustering and Consistency Analysis")
    clustering_analyzer = ClusteringAnalyzer(df)
    clustering_results = clustering_analyzer.run_all_analyses()
    clustering_summary = clustering_analyzer.get_summary_statistics()

    if 'consistency' in clustering_summary:
        cons = clustering_summary['consistency']
        print(f"  ✓ Consistency range: {cons['range'][0]:.3f} - {cons['range'][1]:.3f}")

    # Section 3.5: Correlation Analysis
    print("\n[6/7] Section 3.5: Text-Dependent Emotional Correlations")
    corr_analyzer = CorrelationAnalyzer(df)
    corr_results = corr_analyzer.run_all_analyses()
    corr_summary = corr_analyzer.get_summary_statistics()

    if 'interest_sadness_by_genre' in corr_summary:
        print("  ✓ Interest-Sadness correlations by genre:")
        for genre, stats in corr_summary['interest_sadness_by_genre'].items():
            print(f"      {genre}: r={stats['r']:.3f}")

    revision_support = build_revision_support_exports(
        temp_results,
        fuzzy_results,
        persona_results,
    )

    # Compile all results
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'completion_rate': 1 - validation_report['overall_missing_rate'],
            'valid_data_points': validation_report['valid_trials']
        },
        'validation': validation_report,
        'temperature': temp_results,
        'fuzzy': fuzzy_results,
        'persona': persona_results,
        'clustering': clustering_results,
        'correlation': corr_results,
        'revision_support': revision_support,
    }

    # Export summary results
    print("\n[7/7] Exporting results...")

    # Export summaries to JSON (excluding DataFrames that cause circular refs)
    summary_data = {
        'metadata': all_results['metadata'],
        'temperature_summary': temp_summary,
        'fuzzy_summary': fuzzy_summary,
        'persona_summary': persona_summary,
        'clustering_summary': clustering_summary,
        'correlation_summary': corr_summary
    }

    try:
        export_to_json(summary_data, 'analysis_summary.json', JSON_OUTPUT)
        export_to_json({'metadata': all_results['metadata']},
                      'analysis_metadata.json', JSON_OUTPUT)
    except Exception as e:
        print(f"  Warning: Could not export full JSON summary: {e}")
        # Export metadata-only fallbacks (keep analysis_summary.json valid for tooling)
        export_to_json({'metadata': all_results['metadata']},
                      'analysis_summary.json', JSON_OUTPUT)
        export_to_json({'metadata': all_results['metadata']},
                      'analysis_metadata.json', JSON_OUTPUT)

    # Export key tables to CSV
    if 'model_reliability' in temp_results:
        export_to_csv(temp_results['model_reliability'],
                     'model_reliability.csv', CSV_OUTPUT)

    if 'temperature_correlation' in temp_results:
        export_to_csv(temp_results['temperature_correlation'],
                     'temperature_correlation.csv', CSV_OUTPUT)

    if 'vendor_temperature_response' in temp_results:
        export_to_csv(temp_results['vendor_temperature_response'],
                      'vendor_temperature_response.csv', CSV_OUTPUT)

    if 'temperature_variance_profiles' in temp_results:
        export_to_csv(temp_results['temperature_variance_profiles'],
                      'temperature_variance_profiles.csv', CSV_OUTPUT)

    if 'model_entropy_profiles' in fuzzy_results:
        export_to_csv(fuzzy_results['model_entropy_profiles'],
                     'model_entropy_profiles.csv', CSV_OUTPUT)

    if 'model_temperature_entropy_profiles' in fuzzy_results:
        export_to_csv(fuzzy_results['model_temperature_entropy_profiles'],
                     'model_temperature_entropy_profiles.csv', CSV_OUTPUT)

    if 'model_sizes_inferred' in fuzzy_results:
        export_to_csv(fuzzy_results['model_sizes_inferred'],
                      'model_sizes_inferred.csv', CSV_OUTPUT)

    if 'model_scale_entropy_correlation' in fuzzy_results:
        export_to_csv(fuzzy_results['model_scale_entropy_correlation'],
                      'model_scale_entropy_correlation.csv', CSV_OUTPUT)

    if 'vendor_fuzzy_chi2_tests' in fuzzy_results:
        export_to_csv(fuzzy_results['vendor_fuzzy_chi2_tests'],
                      'vendor_fuzzy_chi2_tests.csv', CSV_OUTPUT)

    if 'consistency_scores' in clustering_results:
        export_to_csv(clustering_results['consistency_scores'],
                     'consistency_scores.csv', CSV_OUTPUT)

    if 'clustering_stability_ari_samples' in clustering_results:
        export_to_csv(clustering_results['clustering_stability_ari_samples'],
                      'clustering_stability_ari.csv', CSV_OUTPUT)

    if 'clustering_stability_ari_summary' in clustering_results:
        export_to_csv(clustering_results['clustering_stability_ari_summary'],
                      'clustering_stability_ari_summary.csv', CSV_OUTPUT)

    if 'persona_effect_sizes' in persona_results:
        export_to_csv(persona_results['persona_effect_sizes'],
                      'persona_effect_sizes.csv', CSV_OUTPUT)

    if 'persona_chameleon_metrics' in persona_results:
        export_to_csv(persona_results['persona_chameleon_metrics'],
                      'persona_chameleon_metrics.csv', CSV_OUTPUT)

    if 'poet_robot_mannwhitney' in persona_results:
        export_to_csv(persona_results['poet_robot_mannwhitney'],
                      'poet_robot_mannwhitney.csv', CSV_OUTPUT)

    if 'persona_temperature_rank_correlation' in persona_results:
        export_to_csv(persona_results['persona_temperature_rank_correlation'],
                      'persona_temperature_rank_correlation.csv', CSV_OUTPUT)

    if 'persona_temperature_synergy_pearson' in persona_results:
        export_to_csv(persona_results['persona_temperature_synergy_pearson'],
                      'persona_temperature_synergy_pearson.csv', CSV_OUTPUT)

    if 'vendor_persona_anova_typ2' in persona_results:
        export_to_csv(persona_results['vendor_persona_anova_typ2'],
                      'vendor_persona_anova_typ2.csv', CSV_OUTPUT)

    if 'cross_text_consistency' in corr_results:
        export_to_csv(corr_results['cross_text_consistency'],
                     'cross_text_consistency.csv', CSV_OUTPUT)

    if 'within_model_variance' in corr_results:
        export_to_csv(corr_results['within_model_variance'],
                     'within_model_variance.csv', CSV_OUTPUT)

    if 'text_effects_anova_table' in corr_results:
        export_to_csv(corr_results['text_effects_anova_table'],
                      'text_effects_anova.csv', CSV_OUTPUT)

    if 'text_effects_kruskal_wallis_table' in corr_results:
        export_to_csv(corr_results['text_effects_kruskal_wallis_table'],
                      'text_effects_kruskal_wallis.csv', CSV_OUTPUT)

    if 'text_correlation_pairs' in corr_results:
        export_to_csv(corr_results['text_correlation_pairs'],
                      'text_correlation_pairs.csv', CSV_OUTPUT)

    revision_support_exports = {
        'quadrant_threshold_sensitivity': 'quadrant_threshold_sensitivity.csv',
        'quadrant_threshold_reassignments': 'quadrant_threshold_reassignments.csv',
        'temperature_variance_profiles_representative': 'temperature_variance_profiles_representative.csv',
        'entropy_interpretation_anchors': 'entropy_interpretation_anchors.csv',
        'entropy_reliability_robustness': 'entropy_reliability_robustness.csv',
        'pca_loadings_pc1_pc3': 'pca_loadings_pc1_pc3.csv',
    }
    for key, filename in revision_support_exports.items():
        if key in revision_support:
            export_to_csv(revision_support[key], filename, CSV_OUTPUT)

    # Export pooled temperature-entropy correlation for high-controllability models
    if ('temperature_correlation' in temp_results and
            'model_temperature_entropy_profiles' in fuzzy_results):
        kappa_threshold = 0.7
        model_kappa = temp_results['temperature_correlation'][['developer', 'model', 'controllability']]
        high_kappa_models = model_kappa[model_kappa['controllability'] > kappa_threshold][['developer', 'model']]

        profiles = fuzzy_results['model_temperature_entropy_profiles']
        pooled = profiles.merge(high_kappa_models, on=['developer', 'model'], how='inner')

        if len(pooled) > 3:
            x = pooled['temperature'].to_numpy(dtype=float)
            y = pooled['mean_entropy'].to_numpy(dtype=float)

            r = float(np.corrcoef(x, y)[0, 1])
            z = float(np.arctanh(r))
            se = float(1.0 / np.sqrt(len(pooled) - 3))
            ci_low, ci_high = (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))

            try:
                from scipy.stats import pearsonr
                _, p_value = pearsonr(x, y)
                p_value = float(p_value)
            except Exception:
                p_value = float('nan')

            df_export = pd.DataFrame([{
                'kappa_threshold': kappa_threshold,
                'n_models': int(high_kappa_models.drop_duplicates().shape[0]),
                'n_points': int(len(pooled)),
                'r': r,
                'ci_low_95': ci_low,
                'ci_high_95': ci_high,
                'p_value': p_value,
                'ci_method': 'Fisher z',
            }])
            export_to_csv(df_export, 'temperature_entropy_correlation_high_kappa.csv', CSV_OUTPUT)

    print("  ✓ Results exported to results/csv/ and results/json/")

    # Visualization
    if not skip_visualization:
        print("\n[Visualization] Generating figures and tables...")
        visualizer = Visualizer(all_results)
        visualizer.generate_all_visualizations()
        print("  ✓ Figures saved to results/figures/")
        print("  ✓ Tables saved to results/tables/")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return all_results


def run_section(section_number: str, data_path: str | None = None):
    """Run specific analysis section only.

    Args:
        section_number: Section number (e.g., '3.1', '3.2')
        data_path: Optional path to input CSV (default: data_public.csv)
    """
    print(f"\nRunning Section {section_number} only...")

    data_path_kw = {} if data_path is None else {"data_path": Path(data_path)}
    df, _ = load_and_validate_data(**data_path_kw)
    raw_df = load_raw_data(**data_path_kw)

    if section_number == '3.1':
        analyzer = TemperatureAnalyzer(raw_df, df)
        results = analyzer.run_all_analyses()
        summary = analyzer.get_summary_statistics()

    elif section_number == '3.2':
        analyzer = FuzzyAnalyzer(df)
        results = analyzer.run_all_analyses()
        summary = analyzer.get_summary_statistics()

    elif section_number == '3.3':
        analyzer = PersonaAnalyzer(df)
        results = analyzer.run_all_analyses()
        summary = analyzer.get_summary_statistics()

    elif section_number == '3.4':
        analyzer = ClusteringAnalyzer(df)
        results = analyzer.run_all_analyses()
        summary = analyzer.get_summary_statistics()

    elif section_number == '3.5':
        analyzer = CorrelationAnalyzer(df)
        results = analyzer.run_all_analyses()
        summary = analyzer.get_summary_statistics()

    else:
        print(f"Error: Unknown section {section_number}")
        print("Available sections: 3.1, 3.2, 3.3, 3.4, 3.5")
        sys.exit(1)

    print(f"\n=== Section {section_number} Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    return results, summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='LLM Emotional Individuality Analysis Pipeline'
    )
    parser.add_argument(
        '--section',
        type=str,
        help='Run specific section only (3.1, 3.2, 3.3, 3.4, or 3.5)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to input CSV (default: data_public.csv)'
    )

    args = parser.parse_args()

    try:
        if args.section:
            run_section(args.section, data_path=args.data)
        else:
            run_complete_analysis(skip_visualization=args.no_viz, data_path=args.data)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
