"""
Fuzzy membership functions and Shannon-type fuzzy entropy calculations.

Implements the fuzzy-theoretic framework for emotional assessment as described
in Section 2.1.2 of the paper.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass

from .config import FUZZY_PARAMS


@dataclass
class FuzzyMembership:
    """Fuzzy membership values for a single assessment."""
    low: float
    medium: float
    high: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.low, self.medium, self.high])

    def entropy(self) -> float:
        """Calculate Shannon-type fuzzy entropy (membership entropy).

        H_f = -Σ μ_i log₂(μ_i)

        Returns:
            Fuzzy entropy value (for normalized memberships, in [0, log₂(3)];
            for our fixed membership-function design, the achievable maximum is
            ≈ 1.058).
        """
        memberships = self.to_array()
        # Handle zero values: 0 * log(0) = 0
        entropy = 0.0
        for mu in memberships:
            if mu > 0:
                entropy -= mu * np.log2(mu)
        return entropy

    def dominant_region(self) -> str:
        """Get dominant fuzzy region.

        Returns:
            'Low', 'Medium', or 'High'
        """
        memberships = {'Low': self.low, 'Medium': self.medium, 'High': self.high}
        return max(memberships, key=memberships.get)


def fuzzy_low(x: float) -> float:
    """Fuzzy membership function for Low intensity.

    μ_Low(x) = max(0, (30 - x) / 30) for x ∈ [0, 30]
             = 0                      for x > 30

    Args:
        x: Emotion value [0, 100]

    Returns:
        Membership degree [0, 1]
    """
    if x <= 0:
        return 1.0
    elif x <= 30:
        return max(0.0, (30 - x) / 30)
    else:
        return 0.0


def fuzzy_medium(x: float) -> float:
    """Fuzzy membership function for Medium intensity.

    Trapezoidal function:
    μ_Medium(x) = 0                if x ≤ 10
                = (x - 10) / 20    if 10 < x ≤ 30
                = 1                if 30 < x ≤ 50
                = (70 - x) / 20    if 50 < x ≤ 70
                = 0                if x > 70

    Args:
        x: Emotion value [0, 100]

    Returns:
        Membership degree [0, 1]
    """
    if x <= 10:
        return 0.0
    elif x <= 30:
        return (x - 10) / 20
    elif x <= 50:
        return 1.0
    elif x <= 70:
        return (70 - x) / 20
    else:
        return 0.0


def fuzzy_high(x: float) -> float:
    """Fuzzy membership function for High intensity.

    μ_High(x) = 0                  if x < 50
              = (x - 50) / 50      if 50 ≤ x ≤ 100
              = 1                  if x > 100

    Args:
        x: Emotion value [0, 100]

    Returns:
        Membership degree [0, 1]
    """
    if x < 50:
        return 0.0
    elif x <= 100:
        return (x - 50) / 50
    else:
        return 1.0


def calculate_fuzzy_membership(value: float) -> FuzzyMembership:
    """Calculate fuzzy membership for all three regions.

    Args:
        value: Emotion assessment value [0, 100]

    Returns:
        FuzzyMembership object with Low, Medium, High memberships
    """
    return FuzzyMembership(
        low=fuzzy_low(value),
        medium=fuzzy_medium(value),
        high=fuzzy_high(value)
    )


def calculate_fuzzy_entropy(value: float) -> float:
    """Calculate fuzzy entropy for a single emotion value.

    Args:
        value: Emotion assessment value [0, 100]

    Returns:
        Fuzzy entropy H_f ∈ [0, log₂(3)]
    """
    membership = calculate_fuzzy_membership(value)
    return membership.entropy()


def process_fuzzy_data(df: pd.DataFrame, emotion_col: str) -> pd.DataFrame:
    """Process emotion data to add fuzzy membership and entropy columns.

    Args:
        df: DataFrame with emotion values
        emotion_col: Column name (e.g., 'Q1value')

    Returns:
        DataFrame with added fuzzy columns:
        - {emotion_col}_fuzzy_low
        - {emotion_col}_fuzzy_medium
        - {emotion_col}_fuzzy_high
        - {emotion_col}_fuzzy_entropy
        - {emotion_col}_fuzzy_dominant
    """
    result_df = df.copy()

    # Calculate fuzzy memberships
    memberships = df[emotion_col].apply(calculate_fuzzy_membership)

    result_df[f'{emotion_col}_fuzzy_low'] = memberships.apply(lambda m: m.low)
    result_df[f'{emotion_col}_fuzzy_medium'] = memberships.apply(lambda m: m.medium)
    result_df[f'{emotion_col}_fuzzy_high'] = memberships.apply(lambda m: m.high)
    result_df[f'{emotion_col}_fuzzy_entropy'] = memberships.apply(lambda m: m.entropy())
    result_df[f'{emotion_col}_fuzzy_dominant'] = memberships.apply(
        lambda m: m.dominant_region()
    )

    return result_df


def calculate_vendor_fuzzy_distribution(
    df: pd.DataFrame,
    emotion_col: str,
    group_by: str = 'developer'
) -> pd.DataFrame:
    """Calculate fuzzy membership distribution by vendor.

    Args:
        df: DataFrame with fuzzy membership columns
        emotion_col: Emotion column name
        group_by: Grouping column (default: 'developer')

    Returns:
        DataFrame with columns:
        - group_by column
        - mean_score: Mean emotion value
        - std_score: Standard deviation
        - low_pct: Percentage in Low region
        - medium_pct: Percentage in Medium region
        - high_pct: Percentage in High region
        - mean_entropy: Mean fuzzy entropy
    """
    fuzzy_low_col = f'{emotion_col}_fuzzy_low'
    fuzzy_medium_col = f'{emotion_col}_fuzzy_medium'
    fuzzy_high_col = f'{emotion_col}_fuzzy_high'
    fuzzy_entropy_col = f'{emotion_col}_fuzzy_entropy'
    fuzzy_dominant_col = f'{emotion_col}_fuzzy_dominant'

    # Calculate statistics by group
    stats = []

    for group_name, group_df in df.groupby(group_by):
        # Mean and std of raw scores
        mean_score = group_df[emotion_col].mean()
        std_score = group_df[emotion_col].std()

        # Percentage in each dominant region
        dominant_counts = group_df[fuzzy_dominant_col].value_counts()
        total = len(group_df)

        low_pct = (dominant_counts.get('Low', 0) / total) * 100
        medium_pct = (dominant_counts.get('Medium', 0) / total) * 100
        high_pct = (dominant_counts.get('High', 0) / total) * 100

        # Mean entropy
        mean_entropy = group_df[fuzzy_entropy_col].mean()

        stats.append({
            group_by: group_name,
            'mean_score': mean_score,
            'std_score': std_score,
            'low_pct': low_pct,
            'medium_pct': medium_pct,
            'high_pct': high_pct,
            'mean_entropy': mean_entropy
        })

    return pd.DataFrame(stats)


def get_entropy_statistics(df: pd.DataFrame, emotion_col: str) -> Dict:
    """Get entropy statistics for an emotion dimension.

    Args:
        df: DataFrame with fuzzy entropy column
        emotion_col: Emotion column name

    Returns:
        Dictionary with entropy statistics
    """
    entropy_col = f'{emotion_col}_fuzzy_entropy'

    return {
        'mean': df[entropy_col].mean(),
        'std': df[entropy_col].std(),
        'min': df[entropy_col].min(),
        'max': df[entropy_col].max(),
        'median': df[entropy_col].median(),
        'q25': df[entropy_col].quantile(0.25),
        'q75': df[entropy_col].quantile(0.75)
    }


if __name__ == "__main__":
    # Test fuzzy membership functions
    test_values = [0, 15, 30, 40, 50, 60, 70, 85, 100]

    print("Testing Fuzzy Membership Functions")
    print("=" * 60)
    print(f"{'Value':<10} {'Low':<10} {'Medium':<10} {'High':<10} {'Entropy':<10}")
    print("-" * 60)

    for val in test_values:
        membership = calculate_fuzzy_membership(val)
        entropy = membership.entropy()
        print(f"{val:<10} {membership.low:<10.3f} {membership.medium:<10.3f} "
              f"{membership.high:<10.3f} {entropy:<10.3f}")

    # Verify theoretical properties
    print("\n=== Theoretical Property Verification ===")

    # Property 3: Coverage (at least one membership > 0)
    print("\nProperty 3 (Coverage): At least one membership > 0 for all x")
    for val in range(0, 101, 10):
        membership = calculate_fuzzy_membership(val)
        has_coverage = max(membership.to_array()) > 0
        print(f"  x={val}: {has_coverage}")

    # Maximum entropy point
    print("\nMaximum entropy occurs at equal membership (μ₁=μ₂=μ₃=1/3):")
    print(f"  Theoretical max: log₂(3) = {np.log2(3):.4f}")

    # Find empirical max
    entropies = [calculate_fuzzy_entropy(x) for x in range(0, 101)]
    max_entropy = max(entropies)
    max_entropy_value = np.argmax(entropies)
    print(f"  Empirical max: {max_entropy:.4f} at x={max_entropy_value}")
