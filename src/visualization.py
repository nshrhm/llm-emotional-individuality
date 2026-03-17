"""
Visualization module for generating all figures and tables.

Generates publication-quality visualizations for all sections of the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .config import (
    FIGURES_OUTPUT,
    TABLES_OUTPUT,
    VENDOR_COLORS,
    FIGURE_DPI,
    FIGURE_SIZE,
    EMOTIONS,
    TEXTS,
    PERSONAS
)


# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = FIGURE_DPI
plt.rcParams['savefig.dpi'] = FIGURE_DPI
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


class Visualizer:
    """Generate all visualizations for the paper."""

    def __init__(self, results: Dict):
        """Initialize visualizer.

        Args:
            results: Dictionary containing all analysis results
        """
        self.results = results
        self.figures_output = FIGURES_OUTPUT
        self.tables_output = TABLES_OUTPUT

    def save_figure(self, fig, filename: str):
        """Save figure in multiple formats.

        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
        """
        for fmt in ['png', 'pdf']:
            filepath = self.figures_output / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=FIGURE_DPI)
            print(f"  Saved: {filepath}")

    def save_table(self, df: pd.DataFrame, filename: str):
        """Save table as CSV and LaTeX.

        Args:
            df: DataFrame to save
            filename: Base filename (without extension)
        """
        # CSV
        csv_path = self.tables_output / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # LaTeX
        latex_path = self.tables_output / f"{filename}.tex"
        df.to_latex(latex_path, index=False)
        print(f"  Saved: {latex_path}")

    def plot_temperature_entropy_relationship(self, model_temp_entropy_df: pd.DataFrame,
                                              temp_corr_df: pd.DataFrame):
        """Figure 2: Temperature-Fuzzy Entropy Relationship.

        Shows how mean fuzzy entropy changes with temperature for representative models
        with high, moderate, and low controllability.

        Args:
            model_temp_entropy_df: DataFrame with model-temperature-entropy profiles
            temp_corr_df: DataFrame with temperature correlation data (for controllability)
        """
        # Select top and bottom models by controllability
        # 1st: gpt-4o-mini (|r|=0.982), Last: gemma-3-1b-it (|r|=0.039)
        representative_models = {
            'gpt-4o-mini': ('OpenAI', 'High controllability (|r|=0.98)'),
            'gemma-3-1b-it': ('Google', 'Low controllability (|r|=0.04)')
        }

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each representative model
        for model_name, (vendor, label) in representative_models.items():
            model_data = model_temp_entropy_df[
                (model_temp_entropy_df['model'] == model_name)
            ].sort_values('temperature')

            if len(model_data) > 0:
                color = VENDOR_COLORS.get(vendor, '#333333')
                ax.plot(
                    model_data['temperature'],
                    model_data['mean_entropy'],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=f'{model_name} ({label})',
                    color=color,
                    alpha=0.8
                )

        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Mean Fuzzy Entropy', fontsize=12)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        # The revised paper treats this as an empirical profile, not a universal law.
        ax.text(0.02, 0.98,
               'Representative empirical profiles; entropy need not vary monotonically with temperature.',
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        self.save_figure(fig, 'figure2_temperature_entropy_relationship')
        plt.close(fig)

    def plot_temperature_variance_profiles_representative(
        self,
        representative_profiles_df: pd.DataFrame,
        representatives: list[dict[str, object]],
    ):
        """Figure 2b: Representative temperature-variance profiles."""
        if representative_profiles_df.empty or not representatives:
            return

        label_map = {
            "negative": "Strong negative",
            "positive": "Strong positive",
            "near_flat": "Near-flat",
        }
        order = ["negative", "positive", "near_flat"]
        meta_by_type = {item["profile_type"]: item for item in representatives}

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

        for ax, profile_type in zip(axes, order):
            meta = meta_by_type.get(profile_type)
            if meta is None:
                ax.axis("off")
                continue

            profile_df = representative_profiles_df[
                (representative_profiles_df["developer"] == meta["developer"])
                & (representative_profiles_df["model"] == meta["model"])
            ].sort_values("temperature")

            vendor = meta["developer"]
            color = VENDOR_COLORS.get(vendor, "#333333")
            ax.plot(
                profile_df["temperature"],
                profile_df["pooled_variance"],
                marker="o",
                linewidth=2,
                markersize=7,
                color=color,
            )
            ax.set_title(
                f"{label_map[profile_type]}\n{meta['model']}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Temperature", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.text(
                0.03,
                0.95,
                f"{vendor}\nr={meta['r_T_sigma2']:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        axes[0].set_ylabel("Pooled variance", fontsize=10)
        plt.tight_layout()
        self.save_figure(fig, "figure2b_temperature_variance_profiles_representative")
        plt.close(fig)

    def plot_reliability_controllability_matrix(self, classification_df: pd.DataFrame):
        """Figure 1: Reliability-Controllability Matrix.

        Args:
            classification_df: DataFrame with reliability and controllability
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot points by vendor
        for vendor, color in VENDOR_COLORS.items():
            vendor_data = classification_df[classification_df['developer'] == vendor]
            if len(vendor_data) > 0:
                ax.scatter(
                    vendor_data['controllability'],
                    vendor_data['reliability'],
                    c=color,
                    label=vendor,
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )

        # Draw quadrant lines
        ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Labels for quadrants
        ax.text(0.75, 0.98, 'Ideal', fontsize=12, ha='center', style='italic', alpha=0.6)
        ax.text(0.25, 0.98, 'Stable', fontsize=12, ha='center', style='italic', alpha=0.6)
        ax.text(0.75, 0.92, 'Controllable', fontsize=12, ha='center', style='italic', alpha=0.6)
        ax.text(0.25, 0.92, 'Avoid', fontsize=12, ha='center', style='italic', alpha=0.6)

        ax.set_xlabel('Controllability (κ = |r_T,σ²|)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reliability (ρ = 1 - MDR)', fontsize=12, fontweight='bold')
        
        # Legend positioned below the plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=vendor, alpha=0.7)
            for vendor, color in VENDOR_COLORS.items()
            if vendor in classification_df['developer'].unique()
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                 fontsize=10, ncol=7, title='Vendor', title_fontsize=11,
                 frameon=True, shadow=True)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.60, 1.005)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_figure(fig, 'figure1_reliability_controllability_matrix')
        plt.close(fig)

    def plot_vendor_fuzzy_distributions(self, vendor_profiles: Dict[str, pd.DataFrame]):
        """Figure 3: Vendor-Specific Fuzzy Distributions Across Emotions.

        Args:
            vendor_profiles: Dictionary mapping emotion -> vendor distribution DataFrame
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        emotions = ['Interest', 'Surprise', 'Sadness', 'Anger']

        for idx, emotion in enumerate(emotions):
            if emotion in vendor_profiles:
                df = vendor_profiles[emotion].sort_values('mean_score', ascending=False)

                ax = axes[idx]

                # Stacked bar chart
                vendors = df['developer'].values
                low_pct = df['low_pct'].values
                medium_pct = df['medium_pct'].values
                high_pct = df['high_pct'].values

                x = np.arange(len(vendors))
                width = 0.6

                ax.bar(x, low_pct, width, label='Low', color='#3498db', alpha=0.8)
                ax.bar(x, medium_pct, width, bottom=low_pct, label='Medium', color='#95a5a6', alpha=0.8)
                ax.bar(x, high_pct, width, bottom=low_pct+medium_pct, label='High', color='#e74c3c', alpha=0.8)

                ax.set_xlabel('Vendor', fontsize=10)
                ax.set_ylabel('Percentage (%)', fontsize=10)
                ax.set_title(f'{emotion}', fontsize=11, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(vendors, rotation=45, ha='right', fontsize=9)
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)

                if idx == 0:
                    ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        self.save_figure(fig, 'figure3_vendor_fuzzy_distributions')
        plt.close(fig)

    def plot_entropy_distribution_violin(self, df: pd.DataFrame, entropy_df: pd.DataFrame):
        """Figure 4: Fuzzy Entropy Distribution by Model.

        Violin plots showing the entropy distributions for representative models
        from each vendor, ordered by mean entropy with overlaid box plots.

        Args:
            df: Main DataFrame with fuzzy entropy columns for each assessment
            entropy_df: DataFrame with model-level entropy statistics
        """
        # Prepare data: collect all entropy values for each model
        from .config import EMOTION_COLUMNS
        
        entropy_cols = [f'{col}_fuzzy_entropy' for col in EMOTION_COLUMNS]
        
        # Select one representative model per vendor
        # Use specific preferred models where defined, otherwise use median
        preferred_models = {
            'Google': 'gemini-2.0-flash'
        }
        
        representative_models = []
        for vendor in entropy_df['developer'].unique():
            vendor_models = entropy_df[entropy_df['developer'] == vendor].copy()
            
            # Check if there's a preferred model for this vendor
            if vendor in preferred_models:
                preferred_model_name = preferred_models[vendor]
                preferred_model = vendor_models[vendor_models['model'] == preferred_model_name]
                
                if len(preferred_model) > 0:
                    representative = preferred_model.iloc[0]
                else:
                    # Fallback to median if preferred model not found
                    vendor_median = vendor_models['overall_mean_entropy'].median()
                    vendor_models['distance_to_median'] = abs(
                        vendor_models['overall_mean_entropy'] - vendor_median
                    )
                    representative = vendor_models.nsmallest(1, 'distance_to_median').iloc[0]
            else:
                # Select model closest to vendor's median entropy
                vendor_median = vendor_models['overall_mean_entropy'].median()
                vendor_models['distance_to_median'] = abs(
                    vendor_models['overall_mean_entropy'] - vendor_median
                )
                representative = vendor_models.nsmallest(1, 'distance_to_median').iloc[0]
            
            representative_models.append({
                'model': representative['model'],
                'developer': representative['developer'],
                'mean_entropy': representative['overall_mean_entropy']
            })
        
        representative_df = pd.DataFrame(representative_models)
        
        # Collect entropy data for representative models only
        model_data = []
        for _, row in representative_df.iterrows():
            model_name = row['model']
            developer = row['developer']
            mean_entropy = row['mean_entropy']
            
            # Get all entropy values for this model
            model_mask = df['model'] == model_name
            if model_mask.any():
                model_df = df[model_mask]
                # Flatten all entropy values across all emotions
                entropy_values = model_df[entropy_cols].values.flatten()
                # Remove NaN values
                entropy_values = entropy_values[~np.isnan(entropy_values)]
                
                for val in entropy_values:
                    model_data.append({
                        'model': model_name,
                        'developer': developer,
                        'entropy': val,
                        'mean_entropy': mean_entropy
                    })
        
        plot_df = pd.DataFrame(model_data)
        
        # Sort models by mean entropy
        model_order = representative_df.sort_values('mean_entropy')['model'].tolist()
        
        # Create figure with better size for fewer models
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create violin plot with vendor colors
        positions = np.arange(len(model_order))
        
        for idx, model in enumerate(model_order):
            model_subset = plot_df[plot_df['model'] == model]
            if len(model_subset) > 0:
                developer = model_subset['developer'].iloc[0]
                color = VENDOR_COLORS.get(developer, '#333333')
                
                # Violin plot
                parts = ax.violinplot(
                    [model_subset['entropy'].values],
                    positions=[idx],
                    widths=0.8,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False
                )
                
                # Color the violin
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.4)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1)
                
                # Overlay bee swarm plot (individual data points)
                entropy_values = model_subset['entropy'].values
                # Add small random jitter to x-position for better visibility
                np.random.seed(42)  # For reproducibility
                x_jitter = np.random.normal(idx, 0.04, size=len(entropy_values))
                ax.scatter(x_jitter, entropy_values, 
                          alpha=0.3, s=20, color=color, 
                          edgecolors='black', linewidth=0.3,
                          zorder=3)
                
                # Overlay box plot with transparent boxes
                bp = ax.boxplot(
                    [model_subset['entropy'].values],
                    positions=[idx],
                    widths=0.3,
                    showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.3, linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5)
                )
        
        # Customize plot
        ax.set_ylabel('Fuzzy Entropy $H_f$', fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        
        # Create labels with vendor name
        labels = []
        for model in model_order:
            developer = plot_df[plot_df['model'] == model]['developer'].iloc[0]
            labels.append(f'{model}\n({developer})')
        
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend for vendors below the plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=vendor, alpha=0.7)
            for vendor, color in VENDOR_COLORS.items()
            if vendor in plot_df['developer'].unique()
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25),
                 fontsize=10, ncol=7, title='Vendor', title_fontsize=11,
                 frameon=True, shadow=True)
        
        plt.tight_layout()
        self.save_figure(fig, 'figure4_entropy_distribution_violin')
        plt.close(fig)

    def plot_tsne_clustering(self, tsne_embedding: np.ndarray, cluster_labels: np.ndarray,
                            model_labels: List[str]):
        """Figure 5: t-SNE Visualization of LLM Emotion Space.

        Args:
            tsne_embedding: 2D t-SNE coordinates
            cluster_labels: Cluster assignments
            model_labels: Model identifiers
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Extract vendors from model labels
        vendors = [label.split('/')[0] for label in model_labels]

        # Plot points colored by vendor
        for vendor, color in VENDOR_COLORS.items():
            vendor_mask = np.array(vendors) == vendor
            if vendor_mask.any():
                ax.scatter(
                    tsne_embedding[vendor_mask, 0],
                    tsne_embedding[vendor_mask, 1],
                    c=color,
                    label=vendor,
                    s=150,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        self.save_figure(fig, 'figure5_tsne_clustering')
        plt.close(fig)

    def plot_correlation_matrices_by_text(self, correlation_matrices: Dict):
        """Figure 6: Genre-Specific Emotion Correlation Matrices.

        Args:
            correlation_matrices: Dictionary with correlation matrices by text
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        texts = ['t1', 't2', 't3']

        for idx, text_id in enumerate(texts):
            if text_id in correlation_matrices:
                corr_data = correlation_matrices[text_id]
                matrix = corr_data['matrix']
                genre = corr_data['genre']

                ax = axes[idx]
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    ax=ax,
                    cbar_kws={'label': 'Correlation'}
                )
                ax.set_title(f'{genre} (T{idx+1})', fontsize=11, fontweight='bold')

        plt.tight_layout()
        self.save_figure(fig, 'figure6_correlation_matrices')
        plt.close(fig)

    def create_table_model_reliability(self, reliability_df: pd.DataFrame):
        """Table 2: Response Reliability Analysis.

        Args:
            reliability_df: DataFrame with model reliability data
        """
        # Select top models with missing data
        table_df = reliability_df[['developer', 'model', 'overall_mdr',
                                   'Q1_mdr', 'Q2_mdr', 'Q3_mdr', 'Q4_mdr']].copy()

        # Format percentages
        for col in ['overall_mdr', 'Q1_mdr', 'Q2_mdr', 'Q3_mdr', 'Q4_mdr']:
            table_df[col] = (table_df[col] * 100).round(1)

        # Rename columns
        table_df.columns = ['Vendor', 'Model', 'Overall MDR', 'Q1 MDR', 'Q2 MDR', 'Q3 MDR', 'Q4 MDR']

        self.save_table(table_df.head(20), 'table2_model_reliability')

    def create_table_temperature_correlation(self, temp_corr_df: pd.DataFrame):
        """Table 3: Temperature Controllability.

        Args:
            temp_corr_df: DataFrame with temperature correlation data
        """
        table_df = temp_corr_df[['developer', 'model', 'r_T_sigma2', 'abs_r']].copy()
        table_df.columns = ['Vendor', 'Model', 'r(T,σ²)', '|r(T,σ²)|']

        # Round to 3 decimal places
        table_df['r(T,σ²)'] = table_df['r(T,σ²)'].round(3)
        table_df['|r(T,σ²)|'] = table_df['|r(T,σ²)|'].round(3)

        self.save_table(table_df.head(15), 'table3_temperature_correlation')

    def create_table_fuzzy_entropy(self, entropy_df: pd.DataFrame):
        """Table 6: Fuzzy Entropy Analysis.

        Args:
            entropy_df: DataFrame with model entropy profiles
        """
        table_df = entropy_df[['developer', 'model', 'overall_mean_entropy',
                               'overall_std_entropy']].copy()
        table_df.columns = ['Vendor', 'Model', 'Mean H_f', 'SD H_f']

        # Round to 2 decimal places
        table_df['Mean H_f'] = table_df['Mean H_f'].round(2)
        table_df['SD H_f'] = table_df['SD H_f'].round(2)

        # Show top and bottom models
        top_models = table_df.head(5)
        bottom_models = table_df.tail(5)

        combined = pd.concat([top_models, bottom_models])
        self.save_table(combined, 'table6_fuzzy_entropy')

    def generate_all_visualizations(self):
        """Generate all figures and tables."""
        print("\nGenerating visualizations...")

        # Temperature analysis (Section 3.1)
        if 'temperature' in self.results:
            print("  Section 3.1: Temperature Analysis")
            temp_results = self.results['temperature']

            if 'reliability_controllability' in temp_results:
                self.plot_reliability_controllability_matrix(
                    temp_results['reliability_controllability']
                )

            if 'model_reliability' in temp_results:
                self.create_table_model_reliability(temp_results['model_reliability'])

            if 'temperature_correlation' in temp_results:
                self.create_table_temperature_correlation(
                    temp_results['temperature_correlation']
                )

        # Figure 2: Temperature-Entropy Relationship (Section 3.1.2)
        # This requires data from both temperature and fuzzy analyses
        if 'fuzzy' in self.results and 'temperature' in self.results:
            fuzzy_results = self.results['fuzzy']
            temp_results = self.results['temperature']

            if ('model_temperature_entropy_profiles' in fuzzy_results and
                'temperature_correlation' in temp_results):
                print("  Section 3.1.2: Temperature-Entropy Relationship")
                self.plot_temperature_entropy_relationship(
                    fuzzy_results['model_temperature_entropy_profiles'],
                    temp_results['temperature_correlation']
                )

        if 'revision_support' in self.results:
            revision_support = self.results['revision_support']
            if ('temperature_variance_profiles_representative' in revision_support and
                    'temperature_variance_representatives' in revision_support):
                print("  Appendix support: Representative temperature-variance profiles")
                self.plot_temperature_variance_profiles_representative(
                    revision_support['temperature_variance_profiles_representative'],
                    revision_support['temperature_variance_representatives'],
                )

        # Fuzzy analysis (Section 3.2)
        if 'fuzzy' in self.results:
            print("  Section 3.2: Fuzzy Analysis")
            fuzzy_results = self.results['fuzzy']

            if 'vendor_fuzzy_profiles' in fuzzy_results:
                self.plot_vendor_fuzzy_distributions(
                    fuzzy_results['vendor_fuzzy_profiles']
                )

            # Figure 4: Entropy Distribution Violin Plot
            if ('model_entropy_profiles' in fuzzy_results and 
                'processed_df' in fuzzy_results):
                print("  Figure 4: Fuzzy Entropy Distribution")
                self.plot_entropy_distribution_violin(
                    fuzzy_results['processed_df'],
                    fuzzy_results['model_entropy_profiles']
                )

            if 'model_entropy_profiles' in fuzzy_results:
                self.create_table_fuzzy_entropy(fuzzy_results['model_entropy_profiles'])

        # Clustering analysis (Section 3.4)
        if 'clustering' in self.results:
            print("  Section 3.4: Clustering Analysis")
            cluster_results = self.results['clustering']

            if all(k in cluster_results for k in ['tsne_embedding', 'cluster_labels', 'model_labels']):
                self.plot_tsne_clustering(
                    cluster_results['tsne_embedding'],
                    cluster_results['cluster_labels'],
                    cluster_results['model_labels']
                )

        # Correlation analysis (Section 3.5)
        if 'correlation' in self.results:
            print("  Section 3.5: Correlation Analysis")
            corr_results = self.results['correlation']

            if 'correlation_matrices_by_text' in corr_results:
                self.plot_correlation_matrices_by_text(
                    corr_results['correlation_matrices_by_text']
                )

        print("✓ Visualization generation complete")


if __name__ == "__main__":
    print("This module should be run through the main pipeline.")
    print("Use: python main.py")
