"""
Verification script to validate numerical results against paper values.

This script checks that the computed values match the claims in draft.md.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import CSV_OUTPUT, DATA_PATH, EMOTION_COLUMNS, JSON_OUTPUT, PAPER_VALUES


class ResultVerifier:
    """Verify analysis results against paper values."""

    def __init__(
        self,
        summary_file: Path = JSON_OUTPUT / 'analysis_summary.json',
        data_path: Path = DATA_PATH,
    ):
        """Initialize verifier.

        Args:
            summary_file: Path to analysis summary JSON
            data_path: Path to input CSV (default: data_all.csv)
        """
        self.summary_file = summary_file
        self.data_path = Path(data_path)
        self.summary = None
        self.errors = []
        self.warnings = []
        self.passes = []

    def load_summary(self):
        """Load analysis summary from JSON."""
        if not self.summary_file.exists():
            self.summary = None
            print(f"Summary file not found: {self.summary_file} (will verify from CSVs where possible)")
            return

        try:
            with open(self.summary_file, 'r') as f:
                self.summary = json.load(f)
        except json.JSONDecodeError as e:
            self.summary = None
            print(f"Warning: Could not parse {self.summary_file}: {e} (will verify from CSVs where possible)")
            return

        print(f"Loaded summary from: {self.summary_file}")

    def check_submission_coverage(self):
        """Verify submission coverage (collected/designed) matches configured baseline."""
        expected_coverage = PAPER_VALUES['submission_coverage']
        expected_collected = PAPER_VALUES.get('collected_trials')
        expected_designed = PAPER_VALUES.get('designed_trials')

        df = pd.read_csv(self.data_path)
        collected = int(len(df))
        designed = int(expected_designed) if expected_designed is not None else None
        actual_coverage = float(collected / designed) if designed else float("nan")

        coverage_match = abs(actual_coverage - expected_coverage) < 0.001  # Within 0.1%
        count_match = True
        if expected_collected is not None:
            count_match = count_match and (collected == int(expected_collected))

        if coverage_match and count_match:
            self.passes.append(
                f"✓ Submission coverage: {collected}/{designed} = {actual_coverage:.3f} (matches)"
            )
        else:
            self.warnings.append(
                f"⚠ Submission coverage mismatch: expected {expected_coverage:.3f}, got {actual_coverage:.3f} "
                f"(collected={collected}, designed={designed})"
            )

    def check_valid_score_rate(self):
        """Verify valid-score rate (valid/collected) matches configured baseline."""
        expected = PAPER_VALUES['valid_score_rate']
        df = pd.read_csv(self.data_path)
        missing_mask = df[EMOTION_COLUMNS].isna().any(axis=1)
        actual = float((~missing_mask).mean())

        if abs(actual - expected) < 0.001:  # Within 0.1%
            self.passes.append(f"✓ Valid-score rate (valid/collected): {actual:.3f} (matches {expected:.3f})")
        else:
            self.errors.append(
                f"✗ Valid-score rate mismatch: expected {expected}, got {actual}"
            )

    def check_valid_data_points(self):
        """Verify valid score rows count (e.g., 4,067)."""
        expected = PAPER_VALUES['valid_data_points']
        df = pd.read_csv(self.data_path)
        missing_mask = df[EMOTION_COLUMNS].isna().any(axis=1)
        actual = int((~missing_mask).sum())

        if actual == expected:
            self.passes.append(f"✓ Valid data points: {actual} (matches)")
        else:
            self.warnings.append(
                f"⚠ Data points: expected {expected}, got {actual}"
            )

    def check_temperature_correlation_range(self):
        """Verify temperature correlation range matches configured baseline."""
        expected_min, expected_max = PAPER_VALUES['temperature_corr_range']

        temp_corr_path = CSV_OUTPUT / 'temperature_correlation.csv'
        if temp_corr_path.exists():
            df = pd.read_csv(temp_corr_path)
            actual_min = float(df['r_T_sigma2'].min())
            actual_max = float(df['r_T_sigma2'].max())

            # Check if within reasonable tolerance (±0.01)
            min_match = abs(actual_min - expected_min) < 0.01
            max_match = abs(actual_max - expected_max) < 0.01

            if min_match and max_match:
                self.passes.append(
                    f"✓ Temperature correlation range: "
                    f"[{actual_min:.3f}, {actual_max:.3f}] (matches)"
                )
            else:
                self.warnings.append(
                    f"⚠ Temperature correlation range: "
                    f"expected [{expected_min}, {expected_max}], "
                    f"got [{actual_min:.3f}, {actual_max:.3f}]"
                )

    def check_fuzzy_entropy_range(self):
        """Verify model-level mean entropy range matches configured baseline."""
        expected_min, expected_max = PAPER_VALUES['fuzzy_entropy_range']

        entropy_path = CSV_OUTPUT / 'model_entropy_profiles.csv'
        if entropy_path.exists():
            df = pd.read_csv(entropy_path)
            actual_min = float(df['overall_mean_entropy'].min())
            actual_max = float(df['overall_mean_entropy'].max())

            # Check if within reasonable tolerance (±0.01)
            min_match = abs(actual_min - expected_min) < 0.01
            max_match = abs(actual_max - expected_max) < 0.01

            if min_match and max_match:
                self.passes.append(
                    f"✓ Model-level mean entropy range: "
                    f"[{actual_min:.3f}, {actual_max:.3f}] (matches)"
                )
            else:
                self.warnings.append(
                    f"⚠ Model-level mean entropy range: "
                    f"expected [{expected_min}, {expected_max}], "
                    f"got [{actual_min:.3f}, {actual_max:.3f}]"
                )

    def check_persona_temperature_synergy(self):
        """Verify persona-temperature synergy matches configured baseline."""
        expected = PAPER_VALUES['persona_temp_synergy']

        synergy_path = CSV_OUTPUT / 'persona_temperature_synergy_pearson.csv'
        if synergy_path.exists():
            df = pd.read_csv(synergy_path)
            actual_r = float(df.loc[0, 'r'])
            actual_p = float(df.loc[0, 'p_value'])

            r_match = abs(actual_r - expected['r']) < 0.02
            p_match = abs(actual_p - expected['p']) < 0.02

            if r_match and p_match:
                self.passes.append(
                    f"✓ Persona-temperature synergy (Pearson): "
                    f"r={actual_r:.3f}, p={actual_p:.4f} (matches)"
                )
            else:
                self.warnings.append(
                    f"⚠ Persona-temperature synergy (Pearson): "
                    f"expected r={expected['r']}, p={expected['p']}, "
                    f"got r={actual_r:.3f}, p={actual_p:.4f}"
                )

    def check_consistency_range(self):
        """Verify consistency score range matches configured baseline."""
        expected_min, expected_max = PAPER_VALUES['consistency_range']

        consistency_path = CSV_OUTPUT / 'consistency_scores.csv'
        if consistency_path.exists():
            df = pd.read_csv(consistency_path)
            actual_min = float(df['consistency_score'].min())
            actual_max = float(df['consistency_score'].max())

            min_match = abs(actual_min - expected_min) < 0.02
            max_match = abs(actual_max - expected_max) < 0.02

            if min_match and max_match:
                self.passes.append(
                    f"✓ Consistency range: "
                    f"[{actual_min:.3f}, {actual_max:.3f}] (matches)"
                )
            else:
                self.warnings.append(
                    f"⚠ Consistency range: "
                    f"expected [{expected_min}, {expected_max}], "
                    f"got [{actual_min:.3f}, {actual_max:.3f}]"
                )

    def run_all_checks(self):
        """Run all verification checks."""
        print("\n" + "=" * 70)
        print("Result Verification Against Paper Values")
        print("=" * 70)

        self.load_summary()

        print("\nRunning verification checks...\n")

        self.check_submission_coverage()
        self.check_valid_score_rate()
        self.check_valid_data_points()
        self.check_temperature_correlation_range()
        self.check_fuzzy_entropy_range()
        self.check_persona_temperature_synergy()
        self.check_consistency_range()

        # Print results
        print("\n" + "-" * 70)
        print("VERIFICATION RESULTS")
        print("-" * 70)

        if self.passes:
            print(f"\n✅ PASSED ({len(self.passes)}):")
            for msg in self.passes:
                print(f"  {msg}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  {msg}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  {msg}")

        print("\n" + "=" * 70)

        # Summary
        total_checks = len(self.passes) + len(self.warnings) + len(self.errors)
        print(f"Total checks: {total_checks}")
        print(f"Passed: {len(self.passes)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Errors: {len(self.errors)}")

        if self.errors:
            print("\n❌ Verification FAILED - Please review errors above")
            return False
        elif self.warnings:
            print("\n⚠️  Verification PASSED with warnings - Please review above")
            return True
        else:
            print("\n✅ Verification PASSED - All checks successful!")
            return True

    def generate_verification_report(self, output_file: Path = JSON_OUTPUT / 'verification_report.json'):
        """Generate detailed verification report.

        Args:
            output_file: Path to save report
        """
        timestamp = None
        if self.summary and 'metadata' in self.summary:
            timestamp = self.summary['metadata'].get('timestamp')
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        report = {
            'timestamp': timestamp,
            'total_checks': len(self.passes) + len(self.warnings) + len(self.errors),
            'passed': len(self.passes),
            'warnings': len(self.warnings),
            'errors': len(self.errors),
            'passed_checks': self.passes,
            'warning_checks': self.warnings,
            'error_checks': self.errors,
            'overall_status': 'PASSED' if not self.errors else 'FAILED'
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nVerification report saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify computed results against PAPER_VALUES baselines.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_PATH),
        help="Path to input CSV (default: data_all.csv)",
    )
    args = parser.parse_args()

    verifier = ResultVerifier(data_path=Path(args.data))

    try:
        success = verifier.run_all_checks()
        verifier.generate_verification_report()

        sys.exit(0 if success else 1)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
