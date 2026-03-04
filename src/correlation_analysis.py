"""
Text-Dependent Emotional Correlations Analysis (Section 3.5).

Implements:
- Genre-specific correlation matrices
- Interest-Sadness anti-correlation analysis
- ANOVA for text effects
- Emotional tension analysis
- Variance component analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

from .config import EMOTION_COLUMNS, EMOTIONS, TEXTS, ALPHA
from .utils import calculate_anova, calculate_correlation, kruskal_wallis_test


class CorrelationAnalyzer:
    """Analyze text-dependent emotional correlations."""

    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer.

        Args:
            df: DataFrame with emotion data
        """
        self.df = df
        self.results = {}

    def calculate_emotion_correlations_by_text(self) -> Dict[str, pd.DataFrame]:
        """Calculate correlation matrices for each text (genre).

        Returns:
            Dictionary mapping text_id -> correlation matrix DataFrame
        """
        correlation_matrices = {}

        for text_id in sorted(self.df['text'].unique()):
            text_data = self.df[self.df['text'] == text_id]

            # Calculate correlation matrix
            corr_matrix = text_data[EMOTION_COLUMNS].corr(method='pearson')

            # Rename columns/index to emotion names
            emotion_names = [EMOTIONS[f'Q{i+1}'] for i in range(4)]
            corr_matrix.columns = emotion_names
            corr_matrix.index = emotion_names

            # Get genre name
            genre = TEXTS[text_id]['genre']

            correlation_matrices[text_id] = {
                'matrix': corr_matrix,
                'genre': genre,
                'n_samples': len(text_data)
            }

        self.results['correlation_matrices_by_text'] = correlation_matrices

        return correlation_matrices

    def analyze_interest_sadness_correlation(self) -> pd.DataFrame:
        """Analyze Interest-Sadness correlation across texts.

        This is the key finding showing text-dependent emotional structure.

        Returns:
            DataFrame with correlation statistics by text
        """
        results = []

        for text_id in sorted(self.df['text'].unique()):
            text_data = self.df[self.df['text'] == text_id]

            # Interest (Q1) vs Sadness (Q3)
            interest = text_data['Q1value'].dropna()
            sadness = text_data['Q3value'].dropna()

            # Ensure matching indices
            common_idx = interest.index.intersection(sadness.index)
            interest = interest.loc[common_idx]
            sadness = sadness.loc[common_idx]

            if len(interest) > 2:
                corr_result = calculate_correlation(
                    interest.values,
                    sadness.values,
                    alpha=ALPHA
                )

                results.append({
                    'text': text_id,
                    'genre': TEXTS[text_id]['genre'],
                    'title': TEXTS[text_id]['title'],
                    'r': corr_result.r,
                    'p_value': corr_result.p_value,
                    'n': corr_result.n,
                    'significant': corr_result.significant
                })

        df_interest_sadness = pd.DataFrame(results)
        self.results['interest_sadness_correlation'] = df_interest_sadness

        return df_interest_sadness

    def calculate_text_effects_anova(self) -> Dict[str, Dict]:
        """Perform ANOVA to test text effects on each emotion.

        Returns:
            Dictionary mapping emotion -> ANOVA results
        """
        anova_results = {}
        anova_rows = []
        kw_rows = {}

        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']

            # One-way ANOVA
            anova = calculate_anova(self.df, 'text', col, alpha=ALPHA)

            # Also perform Kruskal-Wallis (non-parametric)
            kw_result = kruskal_wallis_test(self.df, 'text', col, alpha=ALPHA)
            kw_rows[emotion_name] = kw_result

            # Descriptive statistics by text
            text_stats = []
            for text_id in sorted(self.df['text'].unique()):
                text_data = self.df[self.df['text'] == text_id][col].dropna()
                text_stats.append({
                    'text': text_id,
                    'genre': TEXTS[text_id]['genre'],
                    'mean': text_data.mean(),
                    'std': text_data.std(),
                    'n': len(text_data)
                })

            anova_results[emotion_name] = {
                'anova': anova,
                'kruskal_wallis': kw_result,
                'descriptive_stats': pd.DataFrame(text_stats)
            }

            anova_rows.append({
                'emotion': emotion_name,
                'f_statistic': float(anova.f_statistic),
                'p_value': float(anova.p_value),
                'df_between': int(anova.df_between),
                'df_within': int(anova.df_within),
                'eta_squared': float(anova.eta_squared),
                'significant': bool(anova.significant),
            })

        self.results['text_effects_anova'] = anova_results
        self.results['text_effects_anova_table'] = pd.DataFrame(anova_rows)
        self.results['text_effects_kruskal_wallis_table'] = pd.DataFrame([
            {
                'emotion': emotion,
                'h_statistic': float(result['h_statistic']),
                'p_value': float(result['p_value']),
                'epsilon_squared': float(result.get('epsilon_squared', float('nan'))),
                'n_groups': int(result['n_groups']),
                'total_n': int(result['total_n']),
                'significant': bool(result['significant']),
            }
            for emotion, result in kw_rows.items()
        ])

        return anova_results

    def calculate_emotional_tension(self) -> pd.DataFrame:
        """Calculate emotional tension metrics for each text.

        Emotional tension is defined as the strength of anti-correlation
        between positive (Interest) and negative (Sadness) emotions.

        Returns:
            DataFrame with tension metrics by text
        """
        tension_results = []

        for text_id in sorted(self.df['text'].unique()):
            text_data = self.df[self.df['text'] == text_id]

            # Interest vs Sadness
            interest_sadness_r = text_data[['Q1value', 'Q3value']].corr().iloc[0, 1]

            # Interest vs Anger
            interest_anger_r = text_data[['Q1value', 'Q4value']].corr().iloc[0, 1]

            # Surprise vs Sadness
            surprise_sadness_r = text_data[['Q2value', 'Q3value']].corr().iloc[0, 1]

            # Tension score: magnitude of negative correlations
            tension_score = abs(min(interest_sadness_r, interest_anger_r, 0))

            tension_results.append({
                'text': text_id,
                'genre': TEXTS[text_id]['genre'],
                'title': TEXTS[text_id]['title'],
                'interest_sadness_r': interest_sadness_r,
                'interest_anger_r': interest_anger_r,
                'surprise_sadness_r': surprise_sadness_r,
                'tension_score': tension_score
            })

        df_tension = pd.DataFrame(tension_results)
        df_tension = df_tension.sort_values('tension_score', ascending=False)

        self.results['emotional_tension'] = df_tension

        return df_tension

    def calculate_text_correlation_pairs(self) -> pd.DataFrame:
        """Calculate per-text Pearson correlations for selected emotion pairs.

        This provides a reproducible artifact for Table `tab:emotional_tension`
        by exporting r_{ij}(\\tau) for key pairs.
        """
        pairs = [
            ('Interest', 'Sadness', 'Q1value', 'Q3value'),
            ('Interest', 'Anger', 'Q1value', 'Q4value'),
            ('Surprise', 'Sadness', 'Q2value', 'Q3value'),
            ('Surprise', 'Anger', 'Q2value', 'Q4value'),
        ]

        rows = []
        for text_id in sorted(self.df['text'].unique()):
            text_data = self.df[self.df['text'] == text_id]
            for e1, e2, c1, c2 in pairs:
                corr = calculate_correlation(
                    text_data[c1].to_numpy(),
                    text_data[c2].to_numpy(),
                    alpha=ALPHA
                )
                rows.append({
                    'text': text_id,
                    'genre': TEXTS[text_id]['genre'],
                    'title': TEXTS[text_id]['title'],
                    'emotion_1': e1,
                    'emotion_2': e2,
                    'r': corr.r,
                    'p_value': corr.p_value,
                    'n': corr.n,
                })

        out = pd.DataFrame(rows)
        self.results['text_correlation_pairs'] = out
        return out

    def calculate_variance_components(self) -> pd.DataFrame:
        """Calculate variance components for Interest dimension.

        Decomposes total variance into:
        - Between-texts variance
        - Between-models variance
        - Between-personas variance
        - Residual variance

        Returns:
            DataFrame with variance component analysis
        """
        # Focus on Interest (Q1)
        data = self.df[['text', 'model', 'persona', 'Q1value']].dropna()

        # Total variance
        total_var = data['Q1value'].var()

        # Between-texts variance
        text_means = data.groupby('text')['Q1value'].mean()
        grand_mean = data['Q1value'].mean()
        text_counts = data.groupby('text').size()
        ss_text = sum(text_counts * (text_means - grand_mean) ** 2)
        var_text = ss_text / (len(data) - 1)

        # Between-models variance
        model_means = data.groupby('model')['Q1value'].mean()
        model_counts = data.groupby('model').size()
        ss_model = sum(model_counts * (model_means - grand_mean) ** 2)
        var_model = ss_model / (len(data) - 1)

        # Between-personas variance
        persona_means = data.groupby('persona')['Q1value'].mean()
        persona_counts = data.groupby('persona').size()
        ss_persona = sum(persona_counts * (persona_means - grand_mean) ** 2)
        var_persona = ss_persona / (len(data) - 1)

        # Calculate percentage of total variance
        components = pd.DataFrame([
            {
                'component': 'Text',
                'variance': var_text,
                'percentage': (var_text / total_var) * 100
            },
            {
                'component': 'Model',
                'variance': var_model,
                'percentage': (var_model / total_var) * 100
            },
            {
                'component': 'Persona',
                'variance': var_persona,
                'percentage': (var_persona / total_var) * 100
            },
            {
                'component': 'Total',
                'variance': total_var,
                'percentage': 100.0
            }
        ])

        self.results['variance_components'] = components

        return components

    def calculate_cross_text_consistency(self) -> pd.DataFrame:
        """Calculate cross-text consistency (CTC_M) for each model.

        This implements the manuscript definition:

            σ^2_{M,i} := Var( (1/4) Σ_{d=1}^4 e_d | M, τ_i )
            σ̄^2_M := max_i σ^2_{M,i}
            CTC_M := (1/3) Σ_i σ^2_{M,i} / σ̄^2_M,
            with σ̄^2_M = 0 ⇒ CTC_M := 1.

        Returns:
            DataFrame with CTC values per model.
        """
        text_ids = sorted(TEXTS.keys())
        if len(text_ids) != 3:
            raise ValueError(
                f"Expected exactly 3 texts for CTC_M, got {len(text_ids)}: {text_ids}"
            )

        data = self.df.dropna(subset=EMOTION_COLUMNS + ['text', 'model', 'developer']).copy()
        data['avg_score'] = data[EMOTION_COLUMNS].mean(axis=1)

        per_text_var = (
            data.groupby(['developer', 'model', 'text'])['avg_score']
            .var(ddof=1)
            .reset_index(name='sigma2')
        )
        per_text_var['sigma2'] = per_text_var['sigma2'].fillna(0.0)

        pivot = per_text_var.pivot_table(
            index=['developer', 'model'],
            columns='text',
            values='sigma2',
            aggfunc='first'
        ).reindex(columns=text_ids)

        pivot = pivot.dropna()

        bar_sigma2 = pivot.max(axis=1)
        ratios = pivot.div(bar_sigma2.replace(0.0, np.nan), axis=0)
        ctc = ratios.mean(axis=1)
        ctc = ctc.where(bar_sigma2 > 0.0, 1.0)

        out = pivot.reset_index()
        out = out.rename(columns={t: f'sigma2_{t}' for t in text_ids})
        out['bar_sigma2'] = bar_sigma2.values
        out['CTC_M'] = ctc.values

        out = out[
            ['developer', 'model', 'CTC_M', 'bar_sigma2']
            + [f'sigma2_{t}' for t in text_ids]
        ].sort_values('CTC_M', ascending=False).reset_index(drop=True)

        self.results['cross_text_consistency'] = out
        return out

    def calculate_within_model_variance(self) -> pd.DataFrame:
        """Calculate within-model variance of per-assessment mean intensity.

        This implements the manuscript definition:

            σ^2_{M,avg} := Var( (1/4) Σ_{d=1}^4 e_d | M ),

        where the variance is taken over all valid assessments for model M.

        Returns:
            DataFrame with sigma2_M_avg per (developer, model).
        """
        data = self.df.dropna(subset=EMOTION_COLUMNS + ['model', 'developer']).copy()
        data['avg_score'] = data[EMOTION_COLUMNS].mean(axis=1)

        grouped = data.groupby(['developer', 'model'])['avg_score']
        out = grouped.agg(
            n_assessments='count',
            mean_avg_score='mean',
            sigma2_M_avg=lambda s: s.var(ddof=1),
        ).reset_index()

        self.results['within_model_variance'] = out
        return out

    def run_all_analyses(self) -> Dict:
        """Run all correlation and text-effect analyses.

        Returns:
            Dictionary with all analysis results
        """
        print("Running Correlation Analysis (Section 3.5)...")

        print("  - Calculating emotion correlations by text...")
        self.calculate_emotion_correlations_by_text()

        print("  - Analyzing Interest-Sadness correlation...")
        self.analyze_interest_sadness_correlation()

        print("  - Calculating text effects (ANOVA)...")
        self.calculate_text_effects_anova()

        print("  - Calculating emotional tension...")
        self.calculate_emotional_tension()

        print("  - Calculating per-text correlation pairs...")
        self.calculate_text_correlation_pairs()

        print("  - Calculating variance components...")
        self.calculate_variance_components()

        print("  - Calculating cross-text consistency (CTC_M)...")
        self.calculate_cross_text_consistency()

        print("  - Calculating within-model variance (sigma2_M_avg)...")
        self.calculate_within_model_variance()

        print("✓ Correlation analysis complete")

        return self.results

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for correlation analysis.

        Returns:
            Dictionary with key findings
        """
        summary = {}

        # Interest-Sadness correlations
        interest_sadness = self.results.get('interest_sadness_correlation')
        if interest_sadness is not None:
            summary['interest_sadness_by_genre'] = {
                row['genre']: {
                    'r': row['r'],
                    'p_value': row['p_value'],
                    'significant': row['significant']
                }
                for _, row in interest_sadness.iterrows()
            }

        # Emotional tension
        tension = self.results.get('emotional_tension')
        if tension is not None:
            summary['emotional_tension'] = {
                row['genre']: {
                    'tension_score': row['tension_score'],
                    'interest_sadness_r': row['interest_sadness_r']
                }
                for _, row in tension.iterrows()
            }

        # Variance components
        var_comp = self.results.get('variance_components')
        if var_comp is not None:
            summary['variance_components'] = {
                row['component']: {
                    'variance': row['variance'],
                    'percentage': row['percentage']
                }
                for _, row in var_comp.iterrows()
            }

        # Cross-text consistency
        ctc_df = self.results.get('cross_text_consistency')
        if ctc_df is not None and len(ctc_df) > 0:
            summary['cross_text_consistency'] = {
                'mean': float(ctc_df['CTC_M'].mean()),
                'std': float(ctc_df['CTC_M'].std(ddof=1)),
                'min': float(ctc_df['CTC_M'].min()),
                'max': float(ctc_df['CTC_M'].max()),
                'n_models': int(len(ctc_df))
            }

        # Within-model variance
        within_var = self.results.get('within_model_variance')
        if within_var is not None and len(within_var) > 0:
            summary['within_model_variance'] = {
                'mean': float(within_var['sigma2_M_avg'].mean()),
                'std': float(within_var['sigma2_M_avg'].std(ddof=1)),
                'min': float(within_var['sigma2_M_avg'].min()),
                'max': float(within_var['sigma2_M_avg'].max()),
                'n_models': int(len(within_var))
            }

        return summary


if __name__ == "__main__":
    from .data_loader import load_and_validate_data

    print("Loading data...")
    df, _ = load_and_validate_data()

    print("\nRunning correlation analysis...")
    analyzer = CorrelationAnalyzer(df)
    results = analyzer.run_all_analyses()

    print("\n=== Summary Statistics ===")
    summary = analyzer.get_summary_statistics()

    if 'interest_sadness_by_genre' in summary:
        print("\nInterest-Sadness Correlation by Genre:")
        for genre, stats in summary['interest_sadness_by_genre'].items():
            print(f"  {genre}: r={stats['r']:.3f}, p={stats['p_value']:.4f}")
