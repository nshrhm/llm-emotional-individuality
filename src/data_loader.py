"""
Data loading and validation module.

Loads data_all.csv and performs validation checks to ensure data integrity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings

from .config import (
    DATA_PATH,
    EMOTION_COLUMNS,
    REASON_COLUMNS,
    EXPECTED_DATA_POINTS,
    VENDORS
)


class DataLoader:
    """Load and validate experimental data."""

    _INTEGER_TOLERANCE = 1e-9

    def __init__(self, data_path: Path = DATA_PATH):
        """Initialize data loader.

        Args:
            data_path: Path to data_all.csv file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.validation_report = {}

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file with validation.

        Returns:
            DataFrame with validated data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        print(f"Loaded {len(self.df)} rows")

        # Validate data structure
        self._validate_columns()
        self._validate_emotion_values()
        self._calculate_missing_rates()
        self._validate_data_dimensions()

        return self.df

    def _validate_columns(self):
        """Validate that required columns exist."""
        required_cols = [
            "timestamp", "text", "developer", "model", "persona",
            "temperature", "trial"
        ] + EMOTION_COLUMNS

        missing_required = set(required_cols) - set(self.df.columns)
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        # Reason columns (free-text justifications) are optional for reproducibility.
        # Public datasets may omit them due to redistribution constraints.
        missing_reasons = set(REASON_COLUMNS) - set(self.df.columns)
        if missing_reasons:
            warnings.warn(
                "Reason columns missing; continuing (expected for numeric-only public datasets): "
                f"{sorted(missing_reasons)}"
            )
            for col in missing_reasons:
                self.df[col] = ""

        # Ensure reason columns are always strings (avoid NaN surprises downstream).
        for col in REASON_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        print("✓ All required columns present")

    def _validate_emotion_values(self):
        """Validate emotion values are in valid range [0, 100]."""
        for col in EMOTION_COLUMNS:
            # Convert to numeric, coercing errors to NaN
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Enforce integer-valued scores for parsability (and report anomalies)
            valid_mask = self.df[col].notna()
            if valid_mask.any():
                values = self.df.loc[valid_mask, col].to_numpy(dtype=float)

                out_of_range = (values < 0) | (values > 100)
                out_of_range_count = int(out_of_range.sum())
                if out_of_range_count > 0:
                    warnings.warn(
                        f"{col}: {out_of_range_count} values out of range [0, 100]; "
                        f"clipping to [0, 100]"
                    )

                is_integer_like = np.isclose(
                    values,
                    np.round(values),
                    atol=self._INTEGER_TOLERANCE,
                    rtol=0.0,
                )
                non_integer_count = int((~is_integer_like).sum())
                if non_integer_count > 0:
                    warnings.warn(
                        f"{col}: {non_integer_count} non-integer values found; "
                        f"rounding to nearest integer and clipping to [0, 100]"
                    )

                # Round half-up (deterministic for non-negative values) and clip to valid range
                rounded = np.floor(values + 0.5)
                rounded = np.clip(rounded, 0, 100).astype(float)
                self.df.loc[valid_mask, col] = rounded

        print("✓ Emotion values validated")

    def _calculate_missing_rates(self):
        """Calculate missing data rates overall and by model."""
        # Overall missing rate
        total_trials = len(self.df)

        # Check if any emotion value is missing
        missing_mask = self.df[EMOTION_COLUMNS].isna().any(axis=1)
        total_missing = missing_mask.sum()

        overall_mdr = total_missing / total_trials

        valid_trials = total_trials - total_missing
        valid_score_rate = valid_trials / total_trials if total_trials > 0 else float("nan")
        submission_coverage = total_trials / EXPECTED_DATA_POINTS if EXPECTED_DATA_POINTS > 0 else float("nan")

        self.validation_report['overall_missing_rate'] = overall_mdr
        self.validation_report['total_trials'] = total_trials
        self.validation_report['missing_trials'] = total_missing
        self.validation_report['valid_trials'] = valid_trials
        self.validation_report['valid_score_rate'] = valid_score_rate
        self.validation_report['submission_coverage'] = submission_coverage

        print(f"\n=== Data Quality Report ===")
        print(f"Designed maximum trials: {EXPECTED_DATA_POINTS}")
        print(f"Collected trials: {total_trials}")
        print(f"Submission coverage (collected/designed): {submission_coverage * 100:.1f}%")
        print(f"Valid-score trials: {valid_trials}")
        print(f"Missing trials: {total_missing}")
        print(f"Valid-score rate (valid/collected): {valid_score_rate * 100:.1f}%")
        print(f"Missing-data rate (missing/collected): {overall_mdr * 100:.1f}%")

        # Missing rate by model
        model_mdr = self.df.groupby(['developer', 'model']).apply(
            lambda x: x[EMOTION_COLUMNS].isna().any(axis=1).sum() / len(x)
        ).reset_index(name='missing_rate')

        self.validation_report['model_missing_rates'] = model_mdr

        # Show top models with missing data
        if total_missing > 0:
            print(f"\nModels with missing data:")
            models_with_missing = model_mdr[model_mdr['missing_rate'] > 0].sort_values(
                'missing_rate', ascending=False
            )
            for _, row in models_with_missing.head(10).iterrows():
                print(f"  {row['developer']}/{row['model']}: "
                      f"{row['missing_rate']*100:.1f}%")

    def _validate_data_dimensions(self):
        """Validate expected data dimensions."""
        # Check unique counts
        n_models = self.df['model'].nunique()
        n_texts = self.df['text'].nunique()
        n_personas = self.df['persona'].nunique()
        n_vendors = self.df['developer'].nunique()

        print(f"\n=== Data Dimensions ===")
        print(f"Vendors: {n_vendors}")
        print(f"Models: {n_models}")
        print(f"Texts: {n_texts}")
        print(f"Personas: {n_personas}")

        # Expected: 36 models × 3 texts × 4 personas × 10 trials = 4320
        expected_from_dims = n_models * n_texts * n_personas * 10
        print(f"\nExpected trials from observed dimensions: {expected_from_dims}")
        print(f"Designed maximum trials (configured): {EXPECTED_DATA_POINTS}")
        print(f"Collected trials: {len(self.df)}")

        self.validation_report['dimensions'] = {
            'n_vendors': n_vendors,
            'n_models': n_models,
            'n_texts': n_texts,
            'n_personas': n_personas,
            'expected_trials': expected_from_dims,
            'actual_trials': len(self.df)
        }

    def get_valid_data(self) -> pd.DataFrame:
        """Get dataframe with only valid emotion values.

        Returns:
            DataFrame with complete emotion data
        """
        if self.df is None:
            self.load_data()

        # Remove rows with any missing emotion values
        valid_mask = ~self.df[EMOTION_COLUMNS].isna().any(axis=1)
        valid_df = self.df[valid_mask].copy()

        print(f"\nValid data: {len(valid_df)} / {len(self.df)} rows "
              f"({len(valid_df)/len(self.df)*100:.1f}%)")

        return valid_df

    def get_validation_report(self) -> Dict:
        """Get validation report summary.

        Returns:
            Dictionary with validation statistics
        """
        return self.validation_report


def load_and_validate_data(data_path: Path = DATA_PATH) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to load and validate data.

    Args:
        data_path: Path to data CSV file

    Returns:
        Tuple of (valid_dataframe, validation_report)
        valid_dataframe contains only rows without missing emotion values
    """
    loader = DataLoader(data_path)
    loader.load_data()
    valid_df = loader.get_valid_data()
    report = loader.get_validation_report()

    return valid_df, report


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw data including rows with missing values.
    
    Use this for reliability analysis (MDR calculation).
    For other analyses, use load_and_validate_data() to get valid data only.
    
    Args:
        data_path: Path to data CSV file
        
    Returns:
        DataFrame with all data including missing values
    """
    loader = DataLoader(data_path)
    loader.load_data()
    return loader.df


if __name__ == "__main__":
    # Test data loading
    df, report = load_and_validate_data()
    print("\n=== Data Loading Complete ===")
    print(f"Valid data shape: {df.shape}")
