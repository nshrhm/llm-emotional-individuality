"""
Visualization module for generating all figures and tables.

Generates publication-quality visualizations for all sections of the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import textwrap
from typing import Dict, List, Tuple, Any

from .config import (
    FIGURES_OUTPUT,
    TABLES_OUTPUT,
    VENDOR_COLORS,
    FIGURE_DPI,
    FIGURE_SIZE,
    FIGURE_FORMAT,
    FIGURE_FONT_FAMILY,
    FIGURE_FONT_SERIF,
    FIGURE_BASE_FONT_SIZE,
    FIGURE_AXIS_LABEL_SIZE,
    FIGURE_TICK_LABEL_SIZE,
    FIGURE_LEGEND_FONT_SIZE,
    FIGURE_LEGEND_TITLE_SIZE,
    FIGURE_TITLE_SIZE,
    FIGURE_ANNOTATION_SIZE,
    FIGURE_PDF_FONT_TYPE,
    FIGURE_PS_FONT_TYPE,
    FIGURE_MATHTEXT_FONTSET,
    EMOTIONS,
    TEXTS,
    PERSONAS
)


def apply_publication_style():
    """Set a manuscript-aligned global plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'font.family': FIGURE_FONT_FAMILY,
        'font.serif': FIGURE_FONT_SERIF,
        'font.size': FIGURE_BASE_FONT_SIZE,
        'axes.titlesize': FIGURE_TITLE_SIZE,
        'axes.labelsize': FIGURE_AXIS_LABEL_SIZE,
        'xtick.labelsize': FIGURE_TICK_LABEL_SIZE,
        'ytick.labelsize': FIGURE_TICK_LABEL_SIZE,
        'legend.fontsize': FIGURE_LEGEND_FONT_SIZE,
        'legend.title_fontsize': FIGURE_LEGEND_TITLE_SIZE,
        'pdf.fonttype': FIGURE_PDF_FONT_TYPE,
        'ps.fonttype': FIGURE_PS_FONT_TYPE,
        'mathtext.fontset': FIGURE_MATHTEXT_FONTSET,
    })


apply_publication_style()


def wrap_model_label(model_name: str, vendor_name: str, width: int = 14) -> str:
    """Wrap model labels to preserve readable font sizes in the paper."""
    normalized = model_name.replace('-', '- ').replace('_', '_ ')
    wrapped = textwrap.fill(normalized, width=width, break_long_words=False)
    wrapped = wrapped.replace('- ', '-').replace('_ ', '_')
    return f"{wrapped}\n({vendor_name})"


def classify_temperature_variance_profile(correlation: float) -> str:
    """Map correlation values to the Figure 2b narrative labels."""
    if correlation <= -0.7:
        return 'strong negative'
    if correlation >= 0.7:
        return 'strong positive'
    return 'near-flat'


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
        for fmt in FIGURE_FORMAT:
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

        fig, ax = plt.subplots(figsize=(7.75, 6.5))

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

        ax.set_xlabel('Temperature')
        ax.set_ylabel('Mean Fuzzy Entropy')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        # Add theoretical explanation text
        ax.text(0.02, 0.98,
               'Higher temperature → Greater diversity → Higher fuzzy entropy',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        fig.subplots_adjust(bottom=0.26)
        self.save_figure(fig, 'figure2_temperature_entropy_relationship')
        plt.close(fig)

    def plot_representative_temperature_variance_profiles(
        self,
        profile_df: pd.DataFrame
    ):
        """Figure 2b: Representative temperature-variance profiles."""
        if len(profile_df) == 0:
            return

        representative_order = [
            ('OpenAI', 'gpt-4o-mini'),
            ('Google', 'gemini-2.0-flash'),
            ('Google', 'gemma-3-1b-it'),
        ]
        profile_colors = {
            ('OpenAI', 'gpt-4o-mini'): '#1b9e77',
            ('Google', 'gemini-2.0-flash'): '#3b82f6',
            ('Google', 'gemma-3-1b-it'): '#d97706',
        }

        fig, axes = plt.subplots(1, 3, figsize=(8.25, 4.45), sharey=True)

        y_min = profile_df['pooled_variance'].min()
        y_max = profile_df['pooled_variance'].max()
        y_pad = (y_max - y_min) * 0.16 if y_max > y_min else 40

        for ax, key in zip(axes, representative_order):
            developer, model = key
            panel_df = profile_df[
                (profile_df['developer'] == developer) &
                (profile_df['model'] == model)
            ].sort_values('temperature')

            if len(panel_df) == 0:
                ax.set_visible(False)
                continue

            temperatures = panel_df['temperature'].to_numpy(dtype=float)
            variances = panel_df['pooled_variance'].to_numpy(dtype=float)
            counts = panel_df['n_values'].astype(int).tolist()
            correlation = float(panel_df['r_T_sigma2'].iloc[0])
            profile_label = classify_temperature_variance_profile(correlation)
            color = profile_colors[key]

            ax.plot(
                temperatures,
                variances,
                color=color,
                marker='o',
                linewidth=2.5,
                markersize=7,
                markeredgecolor='white',
                markeredgewidth=0.8,
            )

            if len(temperatures) >= 2:
                coeffs = np.polyfit(temperatures, variances, 1)
                trend_x = np.linspace(temperatures.min(), temperatures.max(), 100)
                trend_y = coeffs[0] * trend_x + coeffs[1]
                ax.plot(
                    trend_x,
                    trend_y,
                    color=color,
                    linestyle='--',
                    linewidth=1.6,
                    alpha=0.85,
                )

            label_offset = (y_max - y_min) * 0.035 if y_max > y_min else 10
            subtitle_y = 0.97
            if key == ('OpenAI', 'gpt-4o-mini'):
                label_offset *= 0.8
                subtitle_y = 520
            for temp, variance in zip(temperatures, variances):
                ax.text(
                    temp,
                    variance + label_offset,
                    f'{variance:.0f}',
                    ha='center',
                    va='bottom',
                )

            ax.set_title(model, fontweight='bold', pad=18)
            subtitle_kwargs = {
                'ha': 'center',
                'fontsize': FIGURE_TICK_LABEL_SIZE - 1,
                'bbox': dict(
                    boxstyle='round,pad=0.22',
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.82,
                ),
            }
            if key == ('OpenAI', 'gpt-4o-mini'):
                ax.text(
                    0.55,
                    subtitle_y,
                    f'{developer}\n$r = {correlation:.3f}$ ({profile_label})',
                    transform=ax.transData,
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.22',
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.90,
                    ),
                    ha='center',
                    fontsize=FIGURE_TICK_LABEL_SIZE - 1,
                )
            else:
                ax.text(
                    0.5,
                    subtitle_y,
                    f'{developer}\n$r = {correlation:.3f}$ ({profile_label})',
                    transform=ax.transAxes,
                    va='top',
                    **subtitle_kwargs,
                )
            ax.text(
                0.5,
                -0.28,
                'n per temperature: ' + ', '.join(str(n) for n in counts),
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=FIGURE_TICK_LABEL_SIZE - 1,
            )
            ax.set_xlabel('Assigned temperature', fontweight='bold')
            ax.set_xticks([0.1, 0.4, 0.7, 0.9])
            ax.set_xlim(0.05, 0.95)
            ax.set_ylim(y_min - y_pad * 0.25, y_max + y_pad)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel('Pooled score variance', fontweight='bold')

        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.29, top=0.83, wspace=0.14)
        self.save_figure(fig, 'figure2b_temperature_variance_profiles_representative')
        plt.close(fig)

    def plot_reliability_controllability_matrix(self, classification_df: pd.DataFrame):
        """Figure 1: Reliability-Controllability Matrix.

        Args:
            classification_df: DataFrame with reliability and controllability
        """
        fig, ax = plt.subplots(figsize=(7.75, 6.6))

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
        ax.text(0.75, 0.98, 'Ideal', ha='center', style='italic', alpha=0.6)
        ax.text(0.25, 0.98, 'Stable', ha='center', style='italic', alpha=0.6)
        ax.text(0.75, 0.92, 'Controllable', ha='center', style='italic', alpha=0.6)
        ax.text(0.25, 0.92, 'Avoid', ha='center', style='italic', alpha=0.6)

        ax.set_xlabel('Controllability (κ = |r_T,σ²|)', fontweight='bold')
        ax.set_ylabel('Reliability (ρ = 1 - MDR)', fontweight='bold')
        
        # Legend positioned below the plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=vendor, alpha=0.7)
            for vendor, color in VENDOR_COLORS.items()
            if vendor in classification_df['developer'].unique()
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.16),
                 ncol=4, title='Vendor', frameon=True, shadow=True)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.60, 1.005)
        ax.grid(True, alpha=0.3)

        fig.subplots_adjust(bottom=0.28)
        self.save_figure(fig, 'figure1_reliability_controllability_matrix')
        plt.close(fig)

    def plot_vendor_fuzzy_distributions(self, vendor_profiles: Dict[str, pd.DataFrame]):
        """Figure 3: Vendor-Specific Fuzzy Distributions Across Emotions.

        Args:
            vendor_profiles: Dictionary mapping emotion -> vendor distribution DataFrame
        """
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 10.0))
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

                ax.set_xlabel('Vendor')
                ax.set_ylabel('Percentage (%)')
                ax.set_title(f'{emotion}', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(vendors, rotation=55, ha='right')
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)

        fig.subplots_adjust(bottom=0.17, hspace=0.42, wspace=0.18)
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
        fig, ax = plt.subplots(figsize=(8.3, 8.8))
        
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
        ax.set_ylabel('Fuzzy Entropy $H_f$', fontweight='bold')
        ax.set_xticks(positions)
        
        # Create labels with vendor name
        labels = []
        for model in model_order:
            developer = plot_df[plot_df['model'] == model]['developer'].iloc[0]
            labels.append(wrap_model_label(model, developer, width=15))
        
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend for vendors below the plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=vendor, alpha=0.7)
            for vendor, color in VENDOR_COLORS.items()
            if vendor in plot_df['developer'].unique()
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.26),
                 ncol=2, title='Vendor', frameon=True, shadow=True)

        fig.subplots_adjust(bottom=0.34)
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
        fig, ax = plt.subplots(figsize=(8.2, 7.8))

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

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=True)
        ax.grid(True, alpha=0.3)

        fig.subplots_adjust(bottom=0.23)
        self.save_figure(fig, 'figure5_tsne_clustering')
        plt.close(fig)

    def plot_correlation_matrices_by_text(self, correlation_matrices: Dict):
        """Figure 6: Genre-Specific Emotion Correlation Matrices.

        Args:
            correlation_matrices: Dictionary with correlation matrices by text
        """
        fig = plt.figure(figsize=(8.6, 3.9))
        grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.09], wspace=0.32)
        axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
        colorbar_ax = fig.add_subplot(grid[0, 3])

        texts = ['t1', 't2', 't3']
        heatmap_mappable = None

        for idx, text_id in enumerate(texts):
            if text_id in correlation_matrices:
                corr_data = correlation_matrices[text_id]
                matrix = corr_data['matrix']
                genre = corr_data['genre']

                ax = axes[idx]
                heatmap = sns.heatmap(
                    matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    ax=ax,
                    cbar=False,
                    annot_kws={'size': FIGURE_ANNOTATION_SIZE}
                )
                ax.set_title(f'{genre} (T{idx+1})', fontweight='bold')
                if idx > 0:
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', length=0)
                heatmap_mappable = heatmap.collections[0]

        if heatmap_mappable is not None:
            colorbar = fig.colorbar(
                heatmap_mappable,
                cax=colorbar_ax
            )
            colorbar.set_label('Correlation', size=FIGURE_AXIS_LABEL_SIZE, labelpad=10)
            colorbar.ax.tick_params(labelsize=FIGURE_TICK_LABEL_SIZE, pad=2)

        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.13, top=0.90)
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

            if 'temperature_variance_profiles_representative' in temp_results:
                self.plot_representative_temperature_variance_profiles(
                    temp_results['temperature_variance_profiles_representative']
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
