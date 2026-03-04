"""
Temperature-Controlled Response Characteristics Analysis (Section 3.1).

Implements:
- Missing Data Rate (MDR) calculation
- Temperature-variance correlation analysis
- Reliability-Controllability matrix classification
- Vendor-level temperature responsiveness
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from scipy import stats

from .config import (
    EMOTION_COLUMNS,
    TEMPERATURE_LEVELS,
    RELIABILITY_THRESHOLD,
    CONTROLLABILITY_THRESHOLD,
    VENDORS
)
from .utils import calculate_correlation, calculate_missing_data_rate


class TemperatureAnalyzer:
    """Analyze temperature effects on LLM responses."""

    def __init__(self, raw_df: pd.DataFrame, valid_df: pd.DataFrame = None):
        """Initialize analyzer.

        Args:
            raw_df: DataFrame with all data including missing values (for MDR calculation)
            valid_df: DataFrame with only valid data (for temperature correlation)
                     If None, will use raw_df (after filtering NaN)
        """
        self.raw_df = raw_df
        self.valid_df = valid_df if valid_df is not None else raw_df.dropna(subset=EMOTION_COLUMNS)
        self.results = {}

    def calculate_model_reliability(self) -> pd.DataFrame:
        """Calculate Missing Data Rate (MDR) for each model.

        MDR = proportion of trials with invalid/missing emotion values
        Reliability ρ = 1 - MDR

        Returns:
            DataFrame with columns:
            - developer
            - model
            - overall_mdr: Overall missing rate
            - Q1_mdr, Q2_mdr, Q3_mdr, Q4_mdr: Per-emotion missing rates
            - reliability: ρ = 1 - overall_mdr
        """
        results = []

        for (developer, model), group in self.raw_df.groupby(['developer', 'model']):
            total_trials = len(group)

            # Overall MDR (any emotion missing)
            any_missing = group[EMOTION_COLUMNS].isna().any(axis=1).sum()
            overall_mdr = any_missing / total_trials

            # Per-emotion MDR
            mdr_by_emotion = {}
            for i, col in enumerate(EMOTION_COLUMNS, 1):
                col_missing = group[col].isna().sum()
                mdr_by_emotion[f'Q{i}_mdr'] = col_missing / total_trials

            results.append({
                'developer': developer,
                'model': model,
                'total_trials': total_trials,
                'missing_trials': any_missing,
                'overall_mdr': overall_mdr,
                'reliability': 1 - overall_mdr,
                **mdr_by_emotion
            })

        df_reliability = pd.DataFrame(results)
        df_reliability = df_reliability.sort_values('overall_mdr', ascending=False)

        self.results['model_reliability'] = df_reliability
        return df_reliability

    def calculate_temperature_correlation(self) -> pd.DataFrame:
        """Calculate correlation between temperature and response variance.

        For each model, compute:
        r_T,σ² = corr(T, σ²(T))

        where T are the actual temperature values in data and σ²(T) is response variance at temperature T.

        Returns:
            DataFrame with columns:
            - developer
            - model
            - r_T_sigma2: Correlation coefficient
            - abs_r: Absolute correlation (controllability κ)
            - p_value: Statistical significance
            - n_temperatures: Number of temperature levels tested
        """
        results = []

        for (developer, model), group in self.valid_df.groupby(['developer', 'model']):
            # Get valid data only (already filtered in valid_df)
            valid_group = group

            if len(valid_group) == 0:
                continue

            # Get unique temperature values for this model
            available_temps = sorted(valid_group['temperature'].unique())

            # Calculate variance at each temperature level
            temp_variances = []
            temp_values = []

            for temp in available_temps:
                temp_data = valid_group[valid_group['temperature'] == temp]
                if len(temp_data) > 1:  # Need at least 2 points for variance
                    # Calculate variance across all emotions
                    all_values = temp_data[EMOTION_COLUMNS].values.flatten()
                    variance = np.var(all_values, ddof=1)
                    temp_variances.append(variance)
                    temp_values.append(temp)

            # Calculate correlation if we have enough data points
            if len(temp_values) >= 3:  # Need at least 3 points
                r, p = stats.pearsonr(temp_values, temp_variances)

                results.append({
                    'developer': developer,
                    'model': model,
                    'r_T_sigma2': r,
                    'abs_r': abs(r),
                    'p_value': p,
                    'n_temperatures': len(temp_values),
                    'controllability': abs(r)  # κ = |r_T,σ²|
                })

        df_temp_corr = pd.DataFrame(results)

        # Only sort if we have results
        if len(df_temp_corr) > 0:
            df_temp_corr = df_temp_corr.sort_values('abs_r', ascending=False)

        self.results['temperature_correlation'] = df_temp_corr
        return df_temp_corr

    def classify_reliability_controllability(
        self,
        reliability_df: pd.DataFrame = None,
        temp_corr_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Classify models in Reliability-Controllability matrix.

        Quadrants:
        - Ideal: ρ > 0.95, κ > 0.5
        - Stable: ρ > 0.95, κ ≤ 0.5
        - Controllable: ρ ≤ 0.95, κ > 0.5
        - Avoid: ρ ≤ 0.95, κ ≤ 0.5

        Args:
            reliability_df: Model reliability DataFrame
            temp_corr_df: Temperature correlation DataFrame

        Returns:
            DataFrame with classification
        """
        if reliability_df is None:
            reliability_df = self.results.get('model_reliability')
        if temp_corr_df is None:
            temp_corr_df = self.results.get('temperature_correlation')

        # Merge reliability and controllability
        merged = pd.merge(
            reliability_df[['developer', 'model', 'reliability']],
            temp_corr_df[['developer', 'model', 'controllability']],
            on=['developer', 'model'],
            how='outer'
        )

        # Fill missing values (models without temperature data)
        merged['controllability'] = merged['controllability'].fillna(0)

        # Classify into quadrants
        def classify(row):
            rho = row['reliability']
            kappa = row['controllability']

            if rho > RELIABILITY_THRESHOLD and kappa > CONTROLLABILITY_THRESHOLD:
                return 'Ideal'
            elif rho > RELIABILITY_THRESHOLD and kappa <= CONTROLLABILITY_THRESHOLD:
                return 'Stable'
            elif rho <= RELIABILITY_THRESHOLD and kappa > CONTROLLABILITY_THRESHOLD:
                return 'Controllable'
            else:
                return 'Avoid'

        merged['quadrant'] = merged.apply(classify, axis=1)

        self.results['reliability_controllability'] = merged
        return merged

    def calculate_vendor_temperature_response(
        self,
        temp_corr_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Calculate vendor-level temperature responsiveness.

        Args:
            temp_corr_df: Temperature correlation DataFrame

        Returns:
            DataFrame with vendor statistics
        """
        if temp_corr_df is None:
            temp_corr_df = self.results.get('temperature_correlation')

        vendor_stats = temp_corr_df.groupby('developer').agg({
            'abs_r': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()

        vendor_stats.columns = [
            'developer', 'mean_abs_r', 'std_abs_r',
            'min_abs_r', 'max_abs_r', 'n_models'
        ]

        # Classify response pattern
        def classify_pattern(mean_r, std_r):
            if mean_r > 0.8:
                return 'Highly responsive'
            elif mean_r > 0.6:
                if std_r > 0.2:
                    return 'Mixed (bimodal)'
                else:
                    return 'Moderately responsive'
            else:
                return 'Lower responsiveness'

        vendor_stats['pattern'] = vendor_stats.apply(
            lambda row: classify_pattern(row['mean_abs_r'], row['std_abs_r']),
            axis=1
        )

        vendor_stats = vendor_stats.sort_values('mean_abs_r', ascending=False)

        self.results['vendor_temperature_response'] = vendor_stats
        return vendor_stats

    def run_all_analyses(self) -> Dict:
        """Run all temperature-related analyses.

        Returns:
            Dictionary with all analysis results
        """
        print("Running Temperature Analysis (Section 3.1)...")

        print("  - Calculating model reliability (MDR)...")
        self.calculate_model_reliability()

        print("  - Calculating temperature-variance correlations...")
        self.calculate_temperature_correlation()

        print("  - Classifying reliability-controllability matrix...")
        self.classify_reliability_controllability()

        print("  - Calculating vendor-level temperature response...")
        self.calculate_vendor_temperature_response()

        print("✓ Temperature analysis complete")

        return self.results

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for temperature analysis.

        Returns:
            Dictionary with key findings
        """
        reliability_df = self.results.get('model_reliability')
        temp_corr_df = self.results.get('temperature_correlation')
        classification_df = self.results.get('reliability_controllability')

        summary = {}

        if reliability_df is not None:
            summary['reliability'] = {
                'perfect_reliability_count': (reliability_df['overall_mdr'] == 0).sum(),
                'mean_mdr': reliability_df['overall_mdr'].mean(),
                'max_mdr': reliability_df['overall_mdr'].max(),
                'min_mdr': reliability_df['overall_mdr'].min()
            }

        if temp_corr_df is not None:
            summary['temperature_correlation'] = {
                'mean_correlation': temp_corr_df['r_T_sigma2'].mean(),
                'max_abs_correlation': temp_corr_df['abs_r'].max(),
                'min_abs_correlation': temp_corr_df['abs_r'].min(),
                'correlation_range': (
                    temp_corr_df['r_T_sigma2'].min(),
                    temp_corr_df['r_T_sigma2'].max()
                )
            }

        if classification_df is not None:
            quadrant_counts = classification_df['quadrant'].value_counts()
            total_models = len(classification_df)

            summary['classification'] = {
                'total_models': total_models,
                'ideal_count': quadrant_counts.get('Ideal', 0),
                'stable_count': quadrant_counts.get('Stable', 0),
                'controllable_count': quadrant_counts.get('Controllable', 0),
                'avoid_count': quadrant_counts.get('Avoid', 0),
                'ideal_percentage': (quadrant_counts.get('Ideal', 0) / total_models) * 100
            }

        return summary


if __name__ == "__main__":
    # Test with sample data
    from .data_loader import load_and_validate_data

    print("Loading data...")
    df, _ = load_and_validate_data()

    print("\nRunning temperature analysis...")
    analyzer = TemperatureAnalyzer(df)
    results = analyzer.run_all_analyses()

    print("\n=== Summary Statistics ===")
    summary = analyzer.get_summary_statistics()

    print(f"\nReliability:")
    print(f"  Perfect reliability models: {summary['reliability']['perfect_reliability_count']}")
    print(f"  Mean MDR: {summary['reliability']['mean_mdr']:.3f}")

    print(f"\nTemperature Correlation:")
    corr_range = summary['temperature_correlation']['correlation_range']
    print(f"  Range: r={corr_range[0]:.3f} to r={corr_range[1]:.3f}")

    print(f"\nClassification:")
    print(f"  Ideal: {summary['classification']['ideal_count']} "
          f"({summary['classification']['ideal_percentage']:.1f}%)")
