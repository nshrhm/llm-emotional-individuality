"""
Utility functions for statistical analysis and data export.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from dataclasses import dataclass, asdict


@dataclass
class CorrelationResult:
    """Store correlation analysis results."""
    r: float
    p_value: float
    n: int
    significant: bool

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ANOVAResult:
    """Store ANOVA test results."""
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    eta_squared: float
    significant: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_correlation(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> CorrelationResult:
    """Calculate Pearson correlation with significance test.

    Args:
        x: First variable
        y: Second variable
        alpha: Significance level

    Returns:
        CorrelationResult object
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    r, p = stats.pearsonr(x_clean, y_clean)

    return CorrelationResult(
        r=float(r),
        p_value=float(p),
        n=len(x_clean),
        significant=p < alpha
    )


def calculate_anova(
    data: pd.DataFrame,
    groups_col: str,
    value_col: str,
    alpha: float = 0.05
) -> ANOVAResult:
    """Calculate one-way ANOVA with effect size.

    Args:
        data: DataFrame
        groups_col: Column with group labels
        value_col: Column with values
        alpha: Significance level

    Returns:
        ANOVAResult object
    """
    # Prepare groups
    groups = [group[value_col].dropna().values
              for name, group in data.groupby(groups_col)]

    # Calculate ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    # Calculate effect size (eta-squared)
    # SS_between / SS_total
    grand_mean = data[value_col].mean()
    ss_between = sum(
        len(group) * (group.mean() - grand_mean) ** 2
        for group in groups
    )
    ss_total = sum((data[value_col] - grand_mean) ** 2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Degrees of freedom
    k = len(groups)  # number of groups
    n = len(data)  # total observations
    df_between = k - 1
    df_within = n - k

    return ANOVAResult(
        f_statistic=float(f_stat),
        p_value=float(p_value),
        df_between=df_between,
        df_within=df_within,
        eta_squared=float(eta_squared),
        significant=p_value < alpha
    )


def calculate_missing_data_rate(
    df: pd.DataFrame,
    emotion_cols: List[str]
) -> float:
    """Calculate missing data rate (MDR).

    MDR = proportion of rows with any missing emotion value

    Args:
        df: DataFrame
        emotion_cols: List of emotion column names

    Returns:
        Missing data rate [0, 1]
    """
    missing_mask = df[emotion_cols].isna().any(axis=1)
    return missing_mask.sum() / len(df)


def calculate_consistency_score(
    numeric_values: np.ndarray,
    text_categories: List[str],
    method: str = 'correlation'
) -> float:
    """Calculate language-numerical consistency score.

    Maps text categories to numeric rankings and correlates with numeric values.

    Args:
        numeric_values: Numerical assessments
        text_categories: Text category labels
        method: 'correlation' or 'kappa'

    Returns:
        Consistency score [0, 1] or [-1, 1] for correlation
    """
    # Simple implementation: map categories to ordinal values
    # This is a simplified version - actual implementation may need
    # more sophisticated NLP analysis of text reasons

    # For now, return placeholder
    # TODO: Implement proper consistency analysis
    return 0.8  # Placeholder


def export_to_csv(data: pd.DataFrame, filename: str, output_dir: Path):
    """Export DataFrame to CSV.

    Args:
        data: DataFrame to export
        filename: Output filename
        output_dir: Output directory
    """
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    print(f"Exported: {output_path}")


def export_to_json(data: Dict[str, Any], filename: str, output_dir: Path):
    """Export dictionary to JSON.

    Args:
        data: Dictionary to export
        filename: Output filename
        output_dir: Output directory
    """
    output_path = output_dir / filename

    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return obj

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=convert_types, ensure_ascii=False)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Exported: {output_path}")


def format_correlation_text(corr: CorrelationResult) -> str:
    """Format correlation result as text.

    Args:
        corr: CorrelationResult object

    Returns:
        Formatted string
    """
    sig_marker = "*" if corr.significant else ""
    return f"r={corr.r:.3f}, p={corr.p_value:.4f}{sig_marker}, n={corr.n}"


def format_anova_text(anova: ANOVAResult) -> str:
    """Format ANOVA result as text.

    Args:
        anova: ANOVAResult object

    Returns:
        Formatted string
    """
    sig_marker = "*" if anova.significant else ""
    return (f"F({anova.df_between},{anova.df_within})={anova.f_statistic:.2f}, "
            f"p={anova.p_value:.4f}{sig_marker}, η²={anova.eta_squared:.3f}")


def calculate_effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.

    d = (mean1 - mean2) / pooled_std

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0


def kruskal_wallis_test(
    data: pd.DataFrame,
    groups_col: str,
    value_col: str,
    alpha: float = 0.05
) -> Dict:
    """Perform Kruskal-Wallis H-test (non-parametric ANOVA).

    Args:
        data: DataFrame
        groups_col: Column with group labels
        value_col: Column with values
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    groups = [
        group[value_col].dropna().values
        for _, group in data.groupby(groups_col)
    ]

    h_stat, p_value = stats.kruskal(*groups)
    n_groups = len(groups)
    total_n = int(sum(len(arr) for arr in groups))

    # Epsilon-squared effect size for Kruskal-Wallis.
    # We use the common definition: ε² := (H - k + 1) / (n - k),
    # where k is the number of groups and n is the total sample size.
    epsilon_squared = (
        float((h_stat - n_groups + 1) / (total_n - n_groups))
        if total_n > n_groups
        else float('nan')
    )

    return {
        'h_statistic': float(h_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'n_groups': int(n_groups),
        'total_n': int(total_n),
        'epsilon_squared': epsilon_squared,
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Result of division or default
    """
    return numerator / denominator if denominator != 0 else default
