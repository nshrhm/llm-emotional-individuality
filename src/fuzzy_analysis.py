"""
Fuzzy-Theoretic Emotional Characterization (Section 3.2).

Implements:
- Fuzzy membership distribution analysis
- Fuzzy entropy profiling by model and vendor
- Vendor-specific fuzzy patterns
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional

from .config import EMOTION_COLUMNS, EMOTIONS, VENDORS
from .fuzzy_framework import (
    process_fuzzy_data,
    calculate_vendor_fuzzy_distribution,
    get_entropy_statistics
)
from .utils import calculate_correlation


class FuzzyAnalyzer:
    """Analyze fuzzy-theoretic characteristics of emotion assessments."""

    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer.

        Args:
            df: DataFrame with emotion data
        """
        self.df = df
        self.results = {}
        self._process_fuzzy_memberships()

    def _process_fuzzy_memberships(self):
        """Process all emotion columns to add fuzzy membership and entropy."""
        print("Processing fuzzy memberships for all emotions...")
        for col in EMOTION_COLUMNS:
            self.df = process_fuzzy_data(self.df, col)
        print("✓ Fuzzy processing complete")

    def calculate_vendor_fuzzy_profiles(self) -> Dict[str, pd.DataFrame]:
        """Calculate fuzzy membership distributions for each vendor and emotion.

        Returns:
            Dictionary mapping emotion -> vendor distribution DataFrame
        """
        profiles = {}

        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']
            distribution = calculate_vendor_fuzzy_distribution(
                self.df, col, group_by='developer'
            )
            distribution = distribution.sort_values('mean_score', ascending=False)
            profiles[emotion_name] = distribution

        self.results['vendor_fuzzy_profiles'] = profiles
        return profiles

    def calculate_model_entropy_profiles(self) -> pd.DataFrame:
        """Calculate fuzzy entropy profiles for each model.

        Returns:
            DataFrame with entropy statistics per model
        """
        results = []

        # Define per-assessment fuzzy entropy H_{f,i} as the mean across emotion entropies for that row.
        entropy_cols = [f'{col}_fuzzy_entropy' for col in EMOTION_COLUMNS]
        df_with_row_entropy = self.df.copy()
        df_with_row_entropy['H_f'] = df_with_row_entropy[entropy_cols].mean(axis=1)

        for (developer, model), group in df_with_row_entropy.groupby(['developer', 'model']):
            model_stats = {
                'developer': developer,
                'model': model,
                'n_assessments': len(group)
            }

            # Calculate entropy statistics for each emotion
            for i, col in enumerate(EMOTION_COLUMNS, 1):
                entropy_col = f'{col}_fuzzy_entropy'
                emotion_name = EMOTIONS[f'Q{i}']

                model_stats[f'{emotion_name}_mean_entropy'] = group[entropy_col].mean()
                model_stats[f'{emotion_name}_std_entropy'] = group[entropy_col].std()

            # Overall entropy (average across all emotions)
            model_stats['overall_mean_entropy'] = group[entropy_cols].mean().mean()
            model_stats['overall_std_entropy'] = group[entropy_cols].mean().std()
            model_stats['overall_var_entropy'] = group['H_f'].var(ddof=1)

            results.append(model_stats)

        df_entropy = pd.DataFrame(results)
        df_entropy = df_entropy.sort_values('overall_mean_entropy')

        self.results['model_entropy_profiles'] = df_entropy
        return df_entropy

    def get_entropy_extremes(
        self,
        n_top: int = 5,
        n_bottom: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """Get models with highest and lowest fuzzy entropy.

        Args:
            n_top: Number of high-entropy models
            n_bottom: Number of low-entropy models

        Returns:
            Dictionary with 'high_entropy' and 'low_entropy' DataFrames
        """
        entropy_df = self.results.get('model_entropy_profiles')

        return {
            'low_entropy': entropy_df.head(n_bottom),
            'high_entropy': entropy_df.tail(n_top)
        }

    def analyze_entropy_temperature_relationship(self) -> pd.DataFrame:
        """Analyze relationship between temperature and fuzzy entropy.

        Returns:
            DataFrame with mean entropy by temperature level
        """
        results = []

        for temp in sorted(self.df['temperature'].unique()):
            temp_data = self.df[self.df['temperature'] == temp]

            temp_stats = {'temperature': temp}

            for i, col in enumerate(EMOTION_COLUMNS, 1):
                entropy_col = f'{col}_fuzzy_entropy'
                emotion_name = EMOTIONS[f'Q{i}']
                temp_stats[f'{emotion_name}_mean_entropy'] = temp_data[entropy_col].mean()

            # Overall mean entropy
            entropy_cols = [f'{col}_fuzzy_entropy' for col in EMOTION_COLUMNS]
            temp_stats['overall_mean_entropy'] = temp_data[entropy_cols].mean().mean()

            results.append(temp_stats)

        df_temp_entropy = pd.DataFrame(results)
        self.results['temperature_entropy_relationship'] = df_temp_entropy

        return df_temp_entropy

    def calculate_model_temperature_entropy_profiles(self) -> pd.DataFrame:
        """Calculate temperature-entropy relationship for each model.

        For Figure 2: Shows how each model's fuzzy entropy changes with temperature.

        Returns:
            DataFrame with columns: developer, model, temperature, mean_entropy
        """
        results = []

        for (developer, model), model_group in self.df.groupby(['developer', 'model']):
            for temp in sorted(model_group['temperature'].unique()):
                temp_data = model_group[model_group['temperature'] == temp]

                if len(temp_data) > 0:
                    # Calculate mean entropy across all emotions for this model at this temperature
                    entropy_cols = [f'{col}_fuzzy_entropy' for col in EMOTION_COLUMNS]
                    mean_entropy = temp_data[entropy_cols].mean().mean()

                    results.append({
                        'developer': developer,
                        'model': model,
                        'temperature': temp,
                        'mean_entropy': mean_entropy,
                        'n_assessments': len(temp_data)
                    })

        df_model_temp_entropy = pd.DataFrame(results)
        self.results['model_temperature_entropy_profiles'] = df_model_temp_entropy

        return df_model_temp_entropy

    def calculate_chi_square_fuzzy_distribution(self) -> Dict:
        """Chi-square test for vendor-fuzzy distribution independence.

        Tests H0: Fuzzy membership distribution is independent of vendor

        Returns:
            Dictionary with chi-square test results
        """
        from scipy.stats import chi2_contingency

        results = {}

        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']
            dominant_col = f'{col}_fuzzy_dominant'

            # Create contingency table
            contingency = pd.crosstab(
                self.df['developer'],
                self.df[dominant_col]
            )

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            results[emotion_name] = {
                'chi2': float(chi2),
                'p_value': float(p_value),
                'dof': int(dof),
                'significant': p_value < 0.05,
                'n_observations': int(contingency.to_numpy().sum()),
                'min_expected': float(np.min(expected)) if expected.size else float('nan'),
                'contingency_table': contingency
            }

        self.results['chi_square_tests'] = results
        return results

    def calculate_vendor_fuzzy_chi2_tests_table(self) -> pd.DataFrame:
        """Create a flat table of chi-square test results for export/reproducibility.

        Returns:
            DataFrame with columns:
            - emotion
            - chi2
            - dof
            - p_value
            - n_observations
            - min_expected
            - significant_0_05
        """
        chi_square = self.results.get('chi_square_tests')
        if chi_square is None:
            chi_square = self.calculate_chi_square_fuzzy_distribution()

        rows = []
        for emotion, result in chi_square.items():
            rows.append({
                'emotion': emotion,
                'chi2': result.get('chi2'),
                'dof': result.get('dof'),
                'p_value': result.get('p_value'),
                'n_observations': result.get('n_observations'),
                'min_expected': result.get('min_expected'),
                'significant_0_05': bool(result.get('significant')),
            })

        df_export = pd.DataFrame(rows)
        self.results['vendor_fuzzy_chi2_tests'] = df_export
        return df_export

    @staticmethod
    def _infer_parameter_count_from_model_name(model: str) -> Optional[float]:
        """Infer parameter count from model name when explicitly encoded.

        This is intentionally conservative: it only uses patterns like "70B" or "12b"
        that appear directly in the model identifier (e.g., open-weight model IDs).

        Returns:
            Parameter count as a float (absolute count), or None if not inferable.
        """
        # Prefer explicit "B"/"b" tokens and take the largest one (e.g., "235B-A22B").
        candidates: List[float] = []

        for match in re.finditer(r'(?i)(\d+(?:\.\d+)?)\s*B\b', model):
            candidates.append(float(match.group(1)))

        for match in re.finditer(r'(?i)(?:^|[^A-Za-z0-9])(\d+(?:\.\d+)?)\s*b\b', model):
            candidates.append(float(match.group(1)))

        if not candidates:
            return None

        return max(candidates) * 1e9

    def calculate_model_scale_entropy_correlation(self) -> Dict[str, pd.DataFrame]:
        """Analyze association between model scale and mean fuzzy entropy.

        Since parameter counts are not consistently disclosed for proprietary models,
        this analysis is restricted to models whose parameter counts are explicitly
        encoded in the model identifier (e.g., "7B", "70B", "235B").
        """
        entropy_df = self.results.get('model_entropy_profiles')
        if entropy_df is None:
            entropy_df = self.calculate_model_entropy_profiles()

        sizes_df = entropy_df[['developer', 'model', 'overall_mean_entropy']].copy()
        sizes_df['parameter_count'] = sizes_df['model'].apply(self._infer_parameter_count_from_model_name)
        sizes_df = sizes_df.dropna(subset=['parameter_count']).reset_index(drop=True)
        sizes_df['log10_parameter_count'] = np.log10(sizes_df['parameter_count'].astype(float))

        corr_df = pd.DataFrame()
        if len(sizes_df) >= 3:
            corr = calculate_correlation(
                sizes_df['log10_parameter_count'].to_numpy(dtype=float),
                sizes_df['overall_mean_entropy'].to_numpy(dtype=float),
            )
            corr_df = pd.DataFrame([{
                'r': corr.r,
                'p_value': corr.p_value,
                'n_models': corr.n,
                'x_variable': 'log10(parameter_count)',
                'y_variable': 'overall_mean_entropy',
                'parameter_count_source': 'model_name_only',
            }])

        self.results['model_sizes_inferred'] = sizes_df
        self.results['model_scale_entropy_correlation'] = corr_df
        return {'model_sizes_inferred': sizes_df, 'model_scale_entropy_correlation': corr_df}

    def run_all_analyses(self) -> Dict:
        """Run all fuzzy-theoretic analyses.

        Returns:
            Dictionary with all analysis results
        """
        print("Running Fuzzy Analysis (Section 3.2)...")

        print("  - Calculating vendor fuzzy profiles...")
        self.calculate_vendor_fuzzy_profiles()

        print("  - Calculating model entropy profiles...")
        self.calculate_model_entropy_profiles()

        print("  - Analyzing temperature-entropy relationship...")
        self.analyze_entropy_temperature_relationship()

        print("  - Calculating model-specific temperature-entropy profiles...")
        self.calculate_model_temperature_entropy_profiles()

        print("  - Analyzing model scale vs. entropy (name-inferred sizes)...")
        self.calculate_model_scale_entropy_correlation()

        print("  - Running chi-square tests...")
        self.calculate_chi_square_fuzzy_distribution()
        self.calculate_vendor_fuzzy_chi2_tests_table()

        # Include processed DataFrame with fuzzy entropy columns for visualization
        self.results['processed_df'] = self.df

        print("✓ Fuzzy analysis complete")

        return self.results

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for fuzzy analysis.

        Returns:
            Dictionary with key findings
        """
        summary = {}

        # Entropy range
        entropy_df = self.results.get('model_entropy_profiles')
        if entropy_df is not None:
            summary['entropy_range'] = {
                'min': entropy_df['overall_mean_entropy'].min(),
                'max': entropy_df['overall_mean_entropy'].max(),
                'mean': entropy_df['overall_mean_entropy'].mean(),
                'std': entropy_df['overall_mean_entropy'].std()
            }

        # Vendor profiles
        vendor_profiles = self.results.get('vendor_fuzzy_profiles')
        if vendor_profiles is not None and 'Interest' in vendor_profiles:
            interest_profile = vendor_profiles['Interest']
            summary['vendor_interest_range'] = {
                'max_vendor': interest_profile.iloc[0]['developer'],
                'max_score': interest_profile.iloc[0]['mean_score'],
                'min_vendor': interest_profile.iloc[-1]['developer'],
                'min_score': interest_profile.iloc[-1]['mean_score'],
                'range': interest_profile.iloc[0]['mean_score'] - interest_profile.iloc[-1]['mean_score']
            }

        # Chi-square tests
        chi_square = self.results.get('chi_square_tests')
        if chi_square is not None:
            summary['chi_square_significance'] = {
                emotion: result['significant']
                for emotion, result in chi_square.items()
            }

        return summary


if __name__ == "__main__":
    from .data_loader import load_and_validate_data

    print("Loading data...")
    df, _ = load_and_validate_data()

    print("\nRunning fuzzy analysis...")
    analyzer = FuzzyAnalyzer(df)
    results = analyzer.run_all_analyses()

    print("\n=== Summary Statistics ===")
    summary = analyzer.get_summary_statistics()

    print(f"\nEntropy Range:")
    entropy_range = summary['entropy_range']
    print(f"  Min: {entropy_range['min']:.3f}")
    print(f"  Max: {entropy_range['max']:.3f}")
    print(f"  Mean: {entropy_range['mean']:.3f}")

    if 'vendor_interest_range' in summary:
        print(f"\nVendor Interest Scores:")
        vir = summary['vendor_interest_range']
        print(f"  Highest: {vir['max_vendor']} ({vir['max_score']:.1f})")
        print(f"  Lowest: {vir['min_vendor']} ({vir['min_score']:.1f})")
        print(f"  Range: {vir['range']:.1f} points")
