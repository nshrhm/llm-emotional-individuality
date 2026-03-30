"""
Revision-support exports for the public reproducibility package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CONTROLLABILITY_THRESHOLD, RANDOM_SEED, RELIABILITY_THRESHOLD
from .fuzzy_framework import calculate_fuzzy_entropy


RHO_THRESHOLD_GRID = (0.90, 0.95, 0.98)
KAPPA_THRESHOLD_GRID = (0.40, 0.50, 0.60)
SPEARMAN_BOOTSTRAP_SAMPLES = 5000


def _classify_quadrant(reliability: float, controllability: float, rho_threshold: float, kappa_threshold: float) -> str:
    if reliability > rho_threshold and controllability > kappa_threshold:
        return "Ideal"
    if reliability > rho_threshold and controllability <= kappa_threshold:
        return "Stable"
    if reliability <= rho_threshold and controllability > kappa_threshold:
        return "Controllable"
    return "Avoid"


def _merge_reliability_controllability(
    reliability_df: pd.DataFrame,
    temp_corr_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(
        reliability_df[["developer", "model", "reliability", "overall_mdr"]],
        temp_corr_df[["developer", "model", "r_T_sigma2", "controllability"]],
        on=["developer", "model"],
        how="left",
    )
    merged["controllability"] = merged["controllability"].fillna(0.0)
    merged["r_T_sigma2"] = merged["r_T_sigma2"].fillna(0.0)
    return merged


def build_quadrant_threshold_exports(
    reliability_df: pd.DataFrame,
    temp_corr_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = _merge_reliability_controllability(reliability_df, temp_corr_df)
    merged["base_quad"] = merged.apply(
        lambda row: _classify_quadrant(
            row["reliability"],
            row["controllability"],
            RELIABILITY_THRESHOLD,
            CONTROLLABILITY_THRESHOLD,
        ),
        axis=1,
    )

    sensitivity_rows = []
    reassignment_rows = []

    for rho_threshold in RHO_THRESHOLD_GRID:
        for kappa_threshold in KAPPA_THRESHOLD_GRID:
            current = merged.copy()
            current["new_quad"] = current.apply(
                lambda row: _classify_quadrant(
                    row["reliability"],
                    row["controllability"],
                    rho_threshold,
                    kappa_threshold,
                ),
                axis=1,
            )

            counts = current["new_quad"].value_counts()
            changed = current[current["new_quad"] != current["base_quad"]].copy()
            changed["rho_threshold"] = rho_threshold
            changed["kappa_threshold"] = kappa_threshold

            sensitivity_rows.append({
                "rho_threshold": rho_threshold,
                "kappa_threshold": kappa_threshold,
                "Ideal": int(counts.get("Ideal", 0)),
                "Stable": int(counts.get("Stable", 0)),
                "Controllable": int(counts.get("Controllable", 0)),
                "Avoid": int(counts.get("Avoid", 0)),
                "n_changed_from_base_0p95_0p5": int(len(changed)),
            })

            if len(changed) > 0:
                reassignment_rows.extend(
                    changed[
                        ["developer", "model", "base_quad", "new_quad", "rho_threshold", "kappa_threshold"]
                    ].to_dict(orient="records")
                )

    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(
        ["rho_threshold", "kappa_threshold"]
    ).reset_index(drop=True)
    reassign_df = pd.DataFrame(reassignment_rows)
    if len(reassign_df) > 0:
        reassign_df = reassign_df.sort_values(
            ["rho_threshold", "kappa_threshold", "developer", "model"]
        ).reset_index(drop=True)

    return sensitivity_df, reassign_df


def _choose_temperature_variance_representatives(temp_corr_df: pd.DataFrame) -> list[dict[str, object]]:
    representatives = []

    def add_pick(df: pd.DataFrame, label: str, sort_keys: list[str], ascending: list[bool]) -> None:
        used = {(item["developer"], item["model"]) for item in representatives}
        for row in df.sort_values(sort_keys, ascending=ascending).to_dict(orient="records"):
            key = (row["developer"], row["model"])
            if key not in used:
                representatives.append({
                    "profile_type": label,
                    "developer": row["developer"],
                    "model": row["model"],
                    "r_T_sigma2": float(row["r_T_sigma2"]),
                    "controllability": float(row["controllability"]),
                })
                return

    add_pick(temp_corr_df, "positive", ["r_T_sigma2", "developer", "model"], [False, True, True])
    add_pick(temp_corr_df, "negative", ["r_T_sigma2", "developer", "model"], [True, True, True])
    near_flat = temp_corr_df.assign(abs_signed_r=temp_corr_df["r_T_sigma2"].abs())
    add_pick(near_flat, "near_flat", ["abs_signed_r", "developer", "model"], [True, True, True])

    order = {"negative": 0, "positive": 1, "near_flat": 2}
    representatives.sort(key=lambda item: order[item["profile_type"]])
    return representatives


def build_temperature_variance_profiles_export(
    temp_corr_df: pd.DataFrame,
    temp_variance_profiles_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    representatives = _choose_temperature_variance_representatives(temp_corr_df)
    selected = pd.DataFrame(representatives)[["developer", "model", "r_T_sigma2", "controllability"]]

    profiles_df = temp_variance_profiles_df.merge(
        selected,
        on=["developer", "model"],
        how="inner",
    )
    profiles_df = profiles_df.sort_values(
        ["developer", "model", "temperature"]
    ).reset_index(drop=True)
    return profiles_df, representatives


def _pearson_ci(r_value: float, n: int) -> tuple[float, float]:
    if n <= 3 or abs(r_value) >= 1.0:
        return (float("nan"), float("nan"))
    z_value = np.arctanh(r_value)
    se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(z_value - 1.96 * se)), float(np.tanh(z_value + 1.96 * se)))


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    stats_boot = []
    n = len(x)

    for _ in range(SPEARMAN_BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, n, size=n)
        rho, _ = stats.spearmanr(x[indices], y[indices])
        stats_boot.append(float(rho))

    low, high = np.nanpercentile(stats_boot, [2.5, 97.5])
    return float(low), float(high)


def build_entropy_interpretation_anchors(entropy_df: pd.DataFrame) -> pd.DataFrame:
    entropies = np.array([calculate_fuzzy_entropy(x) for x in range(0, 101)], dtype=float)
    max_entropy = float(entropies.max())
    model_mean_entropy = entropy_df["overall_mean_entropy"].to_numpy(dtype=float)

    return pd.DataFrame([{
        "single_assessment_theoretical_max_entropy": max_entropy,
        "model_level_mean_entropy_min": float(model_mean_entropy.min()),
        "model_level_mean_entropy_q1": float(np.quantile(model_mean_entropy, 0.25)),
        "model_level_mean_entropy_median": float(np.median(model_mean_entropy)),
        "model_level_mean_entropy_q3": float(np.quantile(model_mean_entropy, 0.75)),
        "model_level_mean_entropy_max": float(model_mean_entropy.max()),
        "normalized_model_mean_entropy_min": float(model_mean_entropy.min() / max_entropy),
        "normalized_model_mean_entropy_max": float(model_mean_entropy.max() / max_entropy),
    }])


def build_entropy_reliability_robustness(
    entropy_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(
        entropy_df[["developer", "model", "overall_mean_entropy"]],
        reliability_df[["developer", "model", "overall_mdr"]],
        on=["developer", "model"],
        how="inner",
    ).sort_values(["developer", "model"])

    x = merged["overall_mean_entropy"].to_numpy(dtype=float)
    y = merged["overall_mdr"].to_numpy(dtype=float)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    pearson_low, pearson_high = _pearson_ci(float(pearson_r), len(merged))
    spearman_low, spearman_high = _bootstrap_spearman_ci(x, y)

    return pd.DataFrame([{
        "metric_pair": "overall_mean_entropy_vs_overall_mdr",
        "n_models": int(len(merged)),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "pearson_ci_low_95": pearson_low,
        "pearson_ci_high_95": pearson_high,
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "spearman_bootstrap_ci_low_95": spearman_low,
        "spearman_bootstrap_ci_high_95": spearman_high,
    }])


def build_pca_loadings_export(pca_results: dict) -> pd.DataFrame:
    feature_names = pca_results["feature_names"]
    components = pca_results["components"]
    variance_explained = pca_results["explained_variance_ratio"]

    rows = []
    for component_idx, component_name in enumerate(("PC1", "PC2", "PC3")):
        loadings = np.array(components[component_idx], dtype=float).copy()
        max_abs_idx = int(np.argmax(np.abs(loadings)))
        if loadings[max_abs_idx] < 0:
            loadings *= -1.0
        for feature_name, loading in zip(feature_names, loadings):
            persona, emotion = feature_name.split("_", 1)
            rows.append({
                "component": component_name,
                "persona": persona,
                "emotion": emotion,
                "loading": float(loading),
                "variance_explained": float(variance_explained[component_idx]),
            })

    return pd.DataFrame(rows)


def build_revision_support_exports(
    temp_results: dict,
    fuzzy_results: dict,
    persona_results: dict,
) -> dict:
    sensitivity_df, reassign_df = build_quadrant_threshold_exports(
        temp_results["model_reliability"],
        temp_results["temperature_correlation"],
    )
    variance_profiles_df = temp_results.get("temperature_variance_profiles_representative")
    representatives = []
    if variance_profiles_df is None or len(variance_profiles_df) == 0:
        variance_profiles_df, representatives = build_temperature_variance_profiles_export(
            temp_results["temperature_correlation"],
            temp_results["temperature_variance_profiles"],
        )
    else:
        representatives = (
            variance_profiles_df[
                ["developer", "model", "r_T_sigma2", "controllability"]
            ]
            .drop_duplicates()
            .sort_values(["developer", "model"])
            .to_dict(orient="records")
        )

    outputs = {
        "quadrant_threshold_sensitivity": sensitivity_df,
        "quadrant_threshold_reassignments": reassign_df,
        "temperature_variance_profiles_representative": variance_profiles_df,
        "temperature_variance_representatives": representatives,
        "entropy_interpretation_anchors": build_entropy_interpretation_anchors(
            fuzzy_results["model_entropy_profiles"]
        ),
        "entropy_reliability_robustness": build_entropy_reliability_robustness(
            fuzzy_results["model_entropy_profiles"],
            temp_results["model_reliability"],
        ),
        "pca_loadings_pc1_pc3": build_pca_loadings_export(persona_results["pca"]),
    }
    return outputs
