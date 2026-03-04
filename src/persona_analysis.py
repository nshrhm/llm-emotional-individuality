"""
Persona-Based Cognitive Diversity Analysis (Section 3.3).

Implements:
- Principal Component Analysis (PCA)
- ANOVA for persona effects
- Persona-temperature correlation validation
- Poet vs. Robot statistical comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import EMOTION_COLUMNS, EMOTIONS, PERSONAS, ALPHA
from .utils import calculate_anova, calculate_correlation, calculate_effect_size_cohens_d


class PersonaAnalyzer:
    """Analyze persona-based cognitive diversity."""

    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer.

        Args:
            df: DataFrame with emotion data
        """
        self.df = df
        self.results = {}

    def perform_pca_analysis(self) -> Dict:
        """Perform PCA on persona × emotion assessment matrix.

        Returns:
            Dictionary with PCA results
        """
        # Prepare data: pivot to get persona × emotion features
        # Aggregate by model and text to get mean scores per persona
        pivot_data = self.df.groupby(['model', 'text', 'persona'])[EMOTION_COLUMNS].mean()

        # Reshape to wide format: each row is a (model, text) combination
        # columns are persona_emotion combinations (16 features)
        feature_data = []
        index_data = []

        for (model, text), group in self.df.groupby(['model', 'text']):
            row_features = []
            for persona in sorted(self.df['persona'].unique()):
                persona_data = group[group['persona'] == persona]
                if len(persona_data) > 0:
                    mean_emotions = persona_data[EMOTION_COLUMNS].mean()
                    row_features.extend(mean_emotions.values)
                else:
                    row_features.extend([np.nan] * len(EMOTION_COLUMNS))

            if not any(np.isnan(row_features)):
                feature_data.append(row_features)
                index_data.append((model, text))

        X = np.array(feature_data)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Store results
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': pca.n_components_,
            'feature_names': [
                f'{persona}_{EMOTIONS[f"Q{i+1}"]}'
                for persona in ['p1', 'p2', 'p3', 'p4']
                for i in range(4)
            ]
        }

        # Find persona-related principal component
        # Look for PC with high variance explained and persona separation
        pc3_idx = 2  # Usually PC3 captures persona effects
        pc3_loadings = pca.components_[pc3_idx]

        # Organize loadings by persona
        loadings_by_persona = {}
        for i, persona in enumerate(['p1', 'p2', 'p3', 'p4']):
            persona_loadings = pc3_loadings[i*4:(i+1)*4]
            loadings_by_persona[persona] = {
                EMOTIONS[f'Q{j+1}']: loading
                for j, loading in enumerate(persona_loadings)
            }

        pca_results['pc3_loadings_by_persona'] = loadings_by_persona
        pca_results['pc3_variance_explained'] = pca.explained_variance_ratio_[pc3_idx]

        self.results['pca'] = pca_results
        return pca_results

    def analyze_persona_effects(self) -> Dict[str, pd.DataFrame]:
        """Perform ANOVA to test persona effects on each emotion.

        Returns:
            Dictionary mapping emotion -> ANOVA results DataFrame
        """
        anova_results = {}

        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']

            # Perform one-way ANOVA
            anova = calculate_anova(self.df, 'persona', col, alpha=ALPHA)

            # Post-hoc: pairwise comparisons
            personas = sorted(self.df['persona'].unique())
            pairwise = []

            for j in range(len(personas)):
                for k in range(j+1, len(personas)):
                    p1, p2 = personas[j], personas[k]
                    data1 = self.df[self.df['persona'] == p1][col].dropna()
                    data2 = self.df[self.df['persona'] == p2][col].dropna()

                    # t-test
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    cohens_d = calculate_effect_size_cohens_d(data1.values, data2.values)

                    pairwise.append({
                        'persona1': p1,
                        'persona2': p2,
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'cohens_d': cohens_d,
                        'significant': p_val < ALPHA
                    })

            anova_results[emotion_name] = {
                'anova': anova,
                'pairwise': pd.DataFrame(pairwise)
            }

        self.results['persona_anova'] = anova_results
        return anova_results

    def calculate_persona_effect_sizes_by_model(self) -> pd.DataFrame:
        """Calculate per-model persona effect size and significance.

        Computes within-model one-way ANOVA of the per-assessment mean intensity
        \\bar{e} = (1/4) \\sum_{d=1}^4 e_d across personas.

        Returns:
            DataFrame with per-model eta-squared and ANOVA p-values.
        """
        data = self.df.dropna(subset=EMOTION_COLUMNS + ['developer', 'model', 'persona']).copy()
        data['avg_score'] = data[EMOTION_COLUMNS].mean(axis=1)

        rows = []
        for (developer, model), group in data.groupby(['developer', 'model']):
            persona_groups = [
                g['avg_score'].to_numpy()
                for _, g in group.groupby('persona')
                if len(g) > 0
            ]

            # Conservative defaults when ANOVA is not well-defined
            f_stat = 0.0
            p_value = 1.0

            if len(persona_groups) >= 2 and all(len(arr) >= 2 for arr in persona_groups):
                f_stat, p_value = stats.f_oneway(*persona_groups)

            grand_mean = group['avg_score'].mean()
            ss_total = ((group['avg_score'] - grand_mean) ** 2).sum()
            ss_between = sum(
                len(g) * (g['avg_score'].mean() - grand_mean) ** 2
                for _, g in group.groupby('persona')
            )
            eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0

            rows.append({
                'developer': developer,
                'model': model,
                'n_assessments': int(len(group)),
                'eta2_M_persona': eta2,
                'p_value_M_persona': float(p_value),
                'f_statistic_M_persona': float(f_stat),
            })

        out = pd.DataFrame(rows).sort_values('eta2_M_persona', ascending=False).reset_index(drop=True)
        self.results['persona_effect_sizes'] = out
        return out

    def validate_persona_temperature_synergy(self) -> Dict:
        """Validate persona-temperature correlation hypothesis.

        Tests correlation between assigned temperatures and observed variances.

        Returns:
            Dictionary with correlation results
        """
        # Theoretical temperature assignments
        persona_temps = {
            'p1': PERSONAS['p1']['temperature'],
            'p2': PERSONAS['p2']['temperature'],
            'p3': PERSONAS['p3']['temperature'],
            'p4': PERSONAS['p4']['temperature']
        }

        # Calculate observed variance for each persona
        persona_variances = {}

        for persona in ['p1', 'p2', 'p3', 'p4']:
            persona_data = self.df[self.df['persona'] == persona]
            # Variance across all emotions
            all_values = persona_data[EMOTION_COLUMNS].values.flatten()
            variance = np.var(all_values[~np.isnan(all_values)], ddof=1)
            persona_variances[persona] = variance

        # Correlation between temperature and variance
        temps = [persona_temps[p] for p in ['p1', 'p2', 'p3', 'p4']]
        variances = [persona_variances[p] for p in ['p1', 'p2', 'p3', 'p4']]

        corr_result = calculate_correlation(
            np.array(temps),
            np.array(variances),
            alpha=ALPHA
        )

        # Spearman rank correlation between assigned temperatures and observed variances
        rho_s, p_s = stats.spearmanr(temps, variances)
        temp_ranks = stats.rankdata(temps, method='average')  # 1 = lowest
        variance_ranks = stats.rankdata(variances, method='average')  # 1 = lowest

        synergy_results = {
            'persona_temperatures': persona_temps,
            'persona_variances': persona_variances,
            'correlation': corr_result.to_dict(),
            'spearman_rank_correlation': {
                'rho_s': float(rho_s),
                'p_value': float(p_s),
                'n': len(temps),
                'temperature_ranks': temp_ranks.tolist(),
                'variance_ranks': variance_ranks.tolist(),
            },
            'temperatures': temps,
            'variances': variances
        }

        self.results['persona_temperature_synergy'] = synergy_results

        self.results['persona_temperature_synergy_pearson'] = pd.DataFrame([{
            'r': float(corr_result.r),
            'p_value': float(corr_result.p_value),
            'n': int(corr_result.n),
            'alpha': float(ALPHA),
            'significant': bool(corr_result.significant),
            'persona_order': ['p1', 'p2', 'p3', 'p4'],
            'temperatures': temps,
            'variances': variances,
        }])

        self.results['persona_temperature_rank_correlation'] = pd.DataFrame([{
            'rho_s': float(rho_s),
            'p_value': float(p_s),
            'n': int(len(temps)),
            'temperatures': temps,
            'variances': variances,
            'temperature_ranks': temp_ranks.tolist(),
            'variance_ranks': variance_ranks.tolist(),
            'rank_convention': '1=lowest',
        }])

        return synergy_results

    def calculate_persona_chameleon_metrics(self) -> pd.DataFrame:
        """Compute per-model metrics for the persona-pattern heuristics.

        This supports the manuscript's "Fixed Individuality" and "Persona Chameleon" criteria by
        computing, for each model M:

            ratio(M) := Var_P(s_{M,P}) / Var_{M' in Mset}(s_{M'})

        along with the within-model persona eta-squared.

        Returns:
            DataFrame with per-model ratio, eta-squared, and heuristic flags.
        """
        data = self.df.dropna(subset=EMOTION_COLUMNS + ['developer', 'model', 'persona']).copy()
        data['avg_score'] = data[EMOTION_COLUMNS].mean(axis=1)

        model_means = (
            data.groupby(['developer', 'model'])['avg_score']
            .mean()
            .reset_index(name='s_M')
        )
        var_models = float(model_means['s_M'].var(ddof=1)) if len(model_means) >= 2 else 0.0

        persona_means = (
            data.groupby(['developer', 'model', 'persona'])['avg_score']
            .mean()
            .reset_index(name='s_M_persona')
        )
        var_persona = (
            persona_means.groupby(['developer', 'model'])['s_M_persona']
            .var(ddof=1)
            .reset_index(name='var_over_personas')
        )
        var_persona['var_over_personas'] = var_persona['var_over_personas'].fillna(0.0)

        out = model_means.merge(var_persona, on=['developer', 'model'], how='left')
        out['var_over_personas'] = out['var_over_personas'].fillna(0.0)
        out['var_models_global'] = var_models
        out['ratio_varP_to_varM'] = (
            out['var_over_personas'] / var_models if var_models > 0 else np.nan
        )

        effect_sizes = self.results.get('persona_effect_sizes')
        if effect_sizes is None:
            effect_sizes = self.calculate_persona_effect_sizes_by_model()

        out = out.merge(
            effect_sizes[
                ['developer', 'model', 'eta2_M_persona', 'p_value_M_persona', 'f_statistic_M_persona']
            ],
            on=['developer', 'model'],
            how='left'
        )

        out['fixed_individuality_flag'] = (
            (out['ratio_varP_to_varM'] < 1.0) & (out['eta2_M_persona'] < 0.01)
        )
        out['persona_chameleon_flag'] = (
            (out['ratio_varP_to_varM'] >= 0.1) & (out['eta2_M_persona'] > 0.05)
        )

        out = out.sort_values('ratio_varP_to_varM', ascending=False).reset_index(drop=True)
        self.results['persona_chameleon_metrics'] = out
        return out

    def compare_poet_vs_robot(self) -> Dict:
        """Comprehensive statistical comparison between Poet (p3) and Robot (p4).

        Returns:
            Dictionary with comparison statistics
        """
        poet_data = self.df[self.df['persona'] == 'p3']
        robot_data = self.df[self.df['persona'] == 'p4']

        comparisons = {}

        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']

            poet_values = poet_data[col].dropna()
            robot_values = robot_data[col].dropna()

            # Descriptive statistics
            poet_mean = poet_values.mean()
            poet_std = poet_values.std()
            robot_mean = robot_values.mean()
            robot_std = robot_values.std()

            # t-test
            t_stat, p_val = stats.ttest_ind(poet_values, robot_values)

            # Effect size
            cohens_d = calculate_effect_size_cohens_d(poet_values.values, robot_values.values)

            comparisons[emotion_name] = {
                'poet_mean': poet_mean,
                'poet_std': poet_std,
                'robot_mean': robot_mean,
                'robot_std': robot_std,
                'difference': poet_mean - robot_mean,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < ALPHA
            }

        self.results['poet_vs_robot'] = comparisons
        return comparisons

    def calculate_poet_robot_mannwhitney(self) -> pd.DataFrame:
        """Compute Poet (p3) vs. Robot (p4) Mann--Whitney U tests for each emotion."""
        poet_data = self.df[self.df['persona'] == 'p3']
        robot_data = self.df[self.df['persona'] == 'p4']

        rows = []
        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']
            poet_values = poet_data[col].dropna().to_numpy()
            robot_values = robot_data[col].dropna().to_numpy()

            u_stat = float('nan')
            p_value = float('nan')
            method_used = 'asymptotic'

            try:
                res = stats.mannwhitneyu(
                    poet_values,
                    robot_values,
                    alternative='two-sided',
                    method='asymptotic',
                )
                u_stat = float(res.statistic)
                p_value = float(res.pvalue)
            except TypeError:
                # Fallback for older SciPy versions without the 'method' keyword.
                res = stats.mannwhitneyu(
                    poet_values,
                    robot_values,
                    alternative='two-sided',
                )
                u_stat = float(res.statistic)
                p_value = float(res.pvalue)
                method_used = 'two-sided (scipy default)'

            rows.append({
                'emotion': emotion_name,
                'persona_1': 'p3',
                'persona_2': 'p4',
                'n_persona_1': int(len(poet_values)),
                'n_persona_2': int(len(robot_values)),
                'u_statistic': u_stat,
                'p_value': p_value,
                'alternative': 'two-sided',
                'method': method_used,
            })

        out = pd.DataFrame(rows)
        self.results['poet_robot_mannwhitney'] = out
        return out

    def analyze_vendor_persona_interaction(self) -> pd.DataFrame:
        """Analyze Vendor × Persona interaction effects.

        Returns:
            DataFrame with interaction analysis results
        """
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        interaction_results = []
        anova_rows: List[Dict] = []

        for i, col in enumerate(EMOTION_COLUMNS, 1):
            emotion_name = EMOTIONS[f'Q{i}']

            # Prepare data
            analysis_df = self.df[['developer', 'persona', col]].dropna()
            analysis_df.columns = ['vendor', 'persona', 'value']

            # Two-way ANOVA with interaction
            try:
                model = ols('value ~ C(vendor) + C(persona) + C(vendor):C(persona)',
                           data=analysis_df).fit()
                anova_table = anova_lm(model, typ=2)
                ss_error = float(anova_table.loc['Residual', 'sum_sq'])

                interaction_results.append({
                    'emotion': emotion_name,
                    'vendor_F': anova_table.loc['C(vendor)', 'F'],
                    'vendor_p': anova_table.loc['C(vendor)', 'PR(>F)'],
                    'persona_F': anova_table.loc['C(persona)', 'F'],
                    'persona_p': anova_table.loc['C(persona)', 'PR(>F)'],
                    'interaction_F': anova_table.loc['C(vendor):C(persona)', 'F'],
                    'interaction_p': anova_table.loc['C(vendor):C(persona)', 'PR(>F)'],
                    'interaction_significant': anova_table.loc['C(vendor):C(persona)', 'PR(>F)'] < ALPHA
                })

                for effect_label, effect_name in [
                    ('C(vendor)', 'vendor'),
                    ('C(persona)', 'persona'),
                    ('C(vendor):C(persona)', 'vendor:persona'),
                    ('Residual', 'error'),
                ]:
                    ss = float(anova_table.loc[effect_label, 'sum_sq'])
                    df = int(anova_table.loc[effect_label, 'df'])
                    ms = float(ss / df) if df > 0 else float('nan')
                    f_stat = (
                        float(anova_table.loc[effect_label, 'F'])
                        if effect_label != 'Residual'
                        else float('nan')
                    )
                    p_value = (
                        float(anova_table.loc[effect_label, 'PR(>F)'])
                        if effect_label != 'Residual'
                        else float('nan')
                    )
                    eta_p2 = (
                        float(ss / (ss + ss_error))
                        if effect_label not in ('Residual',) and (ss + ss_error) > 0
                        else float('nan')
                    )

                    anova_rows.append({
                        'emotion': emotion_name,
                        'effect': effect_name,
                        'ss': ss,
                        'df': df,
                        'ms': ms,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'eta_p2': eta_p2,
                        'anova_type': 'II',
                    })
            except Exception as e:
                print(f"Warning: Could not compute interaction for {emotion_name}: {e}")

        df_interaction = pd.DataFrame(interaction_results)
        self.results['vendor_persona_interaction'] = df_interaction
        self.results['vendor_persona_anova_typ2'] = pd.DataFrame(anova_rows)

        return df_interaction

    def run_all_analyses(self) -> Dict:
        """Run all persona-based analyses.

        Returns:
            Dictionary with all analysis results
        """
        print("Running Persona Analysis (Section 3.3)...")

        print("  - Performing PCA...")
        self.perform_pca_analysis()

        print("  - Analyzing persona effects (ANOVA)...")
        self.analyze_persona_effects()

        print("  - Validating persona-temperature synergy...")
        self.validate_persona_temperature_synergy()

        print("  - Calculating per-model persona effect sizes...")
        self.calculate_persona_effect_sizes_by_model()

        print("  - Calculating persona-pattern ratio metrics...")
        self.calculate_persona_chameleon_metrics()

        print("  - Comparing Poet vs. Robot...")
        self.compare_poet_vs_robot()

        print("  - Running non-parametric Poet vs. Robot tests...")
        self.calculate_poet_robot_mannwhitney()

        print("  - Analyzing vendor-persona interaction...")
        self.analyze_vendor_persona_interaction()

        print("✓ Persona analysis complete")

        return self.results

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for persona analysis.

        Returns:
            Dictionary with key findings
        """
        summary = {}

        # PCA results
        pca = self.results.get('pca')
        if pca is not None:
            summary['pca'] = {
                'pc3_variance_explained': pca['pc3_variance_explained'],
                'total_variance_pc1_3': pca['cumulative_variance'][2]
            }

        # Persona-temperature synergy
        synergy = self.results.get('persona_temperature_synergy')
        if synergy is not None:
            summary['persona_temperature_synergy'] = {
                'r': synergy['correlation']['r'],
                'p_value': synergy['correlation']['p_value'],
                'significant': synergy['correlation']['significant']
            }

        # Poet vs. Robot
        poet_robot = self.results.get('poet_vs_robot')
        if poet_robot is not None:
            summary['poet_vs_robot_differences'] = {
                emotion: {
                    'difference': data['difference'],
                    'cohens_d': data['cohens_d'],
                    'significant': data['significant']
                }
                for emotion, data in poet_robot.items()
            }

        # Per-model persona effect sizes
        effect_sizes = self.results.get('persona_effect_sizes')
        if effect_sizes is not None and len(effect_sizes) > 0:
            summary['persona_effect_sizes'] = {
                'eta2_range': (
                    float(effect_sizes['eta2_M_persona'].min()),
                    float(effect_sizes['eta2_M_persona'].max())
                ),
                'n_models': int(len(effect_sizes))
            }

        return summary


if __name__ == "__main__":
    from .data_loader import load_and_validate_data

    print("Loading data...")
    df, _ = load_and_validate_data()

    print("\nRunning persona analysis...")
    analyzer = PersonaAnalyzer(df)
    results = analyzer.run_all_analyses()

    print("\n=== Summary Statistics ===")
    summary = analyzer.get_summary_statistics()

    if 'persona_temperature_synergy' in summary:
        pts = summary['persona_temperature_synergy']
        print(f"\nPersona-Temperature Synergy:")
        print(f"  r = {pts['r']:.3f}, p = {pts['p_value']:.4f}")
        print(f"  Significant: {pts['significant']}")
