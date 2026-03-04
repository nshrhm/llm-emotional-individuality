"""
Model Clustering and Consistency Analysis (Section 3.4).

Implements:
- t-SNE dimensional reduction
- Hierarchical clustering
- Language-numerical consistency scores
- Intra/inter-cluster distance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

from .config import EMOTION_COLUMNS, TSNE_PARAMS, RANDOM_SEED


class ClusteringAnalyzer:
    """Analyze model clustering and consistency."""

    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer.

        Args:
            df: DataFrame with emotion data
        """
        self.df = df
        self.results = {}

    @staticmethod
    def _cluster_labels_for_matrix(X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster the feature matrix using the same preprocessing as the main pipeline."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        return clustering.fit_predict(X_scaled)

    def prepare_model_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for clustering.

        Each model is represented by mean emotion scores across
        all texts and personas (48 dimensions: 3 texts × 4 personas × 4 emotions).

        Returns:
            Tuple of (feature_matrix, model_labels)
        """
        feature_data = []
        model_labels = []

        for (developer, model), group in self.df.groupby(['developer', 'model']):
            # Calculate mean scores for each text-persona-emotion combination
            row_features = []

            for text in sorted(self.df['text'].unique()):
                for persona in sorted(self.df['persona'].unique()):
                    subset = group[(group['text'] == text) & (group['persona'] == persona)]
                    if len(subset) > 0:
                        mean_emotions = subset[EMOTION_COLUMNS].mean()
                        row_features.extend(mean_emotions.values)
                    else:
                        row_features.extend([np.nan] * len(EMOTION_COLUMNS))

            # Only include models with complete data
            if not any(np.isnan(row_features)):
                feature_data.append(row_features)
                model_labels.append(f"{developer}/{model}")

        X = np.array(feature_data)

        self.results['feature_matrix'] = X
        self.results['model_labels'] = model_labels

        return X, model_labels

    def calculate_cluster_stability_ari(
        self,
        n_bootstrap: int = 1000,
        n_clusters: Optional[int] = None,
        random_seed: int = RANDOM_SEED
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Estimate clustering stability via bootstrap ARI.

        We bootstrap within each (model, text, persona) cell by resampling the
        valid assessments in that cell with replacement (keeping the cell size
        fixed), recompute the 48D model feature matrix, recluster, and compute
        the Adjusted Rand Index (ARI) against the baseline clustering.

        Returns:
            (df_samples, df_summary)
        """
        if n_clusters is None:
            n_clusters = int(self.results.get('n_clusters', 5))

        X_base = self.results.get('feature_matrix')
        model_labels = self.results.get('model_labels')
        if X_base is None or model_labels is None:
            X_base, model_labels = self.prepare_model_feature_matrix()

        baseline_labels = self.results.get('cluster_labels')
        if baseline_labels is None or int(self.results.get('n_clusters', n_clusters)) != n_clusters:
            baseline_labels = self._cluster_labels_for_matrix(X_base, n_clusters=n_clusters)

        texts = sorted(self.df['text'].unique())
        personas = sorted(self.df['persona'].unique())

        parsed_models: List[Tuple[str, str]] = [
            tuple(label.split('/', 1)) for label in model_labels
        ]

        cell_data: Dict[Tuple[str, str, str, str], np.ndarray] = {}
        for developer, model in parsed_models:
            model_df = self.df[(self.df['developer'] == developer) & (self.df['model'] == model)]
            for text in texts:
                for persona in personas:
                    subset = model_df[(model_df['text'] == text) & (model_df['persona'] == persona)]
                    values = subset[EMOTION_COLUMNS].dropna().to_numpy(dtype=float)
                    if len(values) == 0:
                        raise ValueError(
                            f"Missing cell data for {developer}/{model}, text={text}, persona={persona} "
                            f"(required for complete-case clustering)."
                        )
                    cell_data[(developer, model, text, persona)] = values

        rng = np.random.default_rng(random_seed)
        ari_values: List[float] = []

        for b in range(int(n_bootstrap)):
            feature_rows: List[List[float]] = []

            for developer, model in parsed_models:
                row_features: List[float] = []
                for text in texts:
                    for persona in personas:
                        values = cell_data[(developer, model, text, persona)]
                        n = values.shape[0]
                        indices = rng.integers(0, n, size=n)
                        row_features.extend(values[indices].mean(axis=0).tolist())
                feature_rows.append(row_features)

            X_boot = np.array(feature_rows, dtype=float)
            labels_boot = self._cluster_labels_for_matrix(X_boot, n_clusters=n_clusters)
            ari_values.append(float(adjusted_rand_score(baseline_labels, labels_boot)))

        df_samples = pd.DataFrame({
            'bootstrap_iteration': np.arange(1, len(ari_values) + 1, dtype=int),
            'ari': ari_values,
            'n_clusters': int(n_clusters),
            'n_models': int(len(parsed_models)),
        })

        mean_ari = float(np.mean(ari_values)) if ari_values else float('nan')
        ci_low, ci_high = (float('nan'), float('nan'))
        if ari_values:
            ci_low, ci_high = (
                float(np.quantile(ari_values, 0.025)),
                float(np.quantile(ari_values, 0.975)),
            )

        df_summary = pd.DataFrame([{
            'n_bootstrap': int(n_bootstrap),
            'n_clusters': int(n_clusters),
            'n_models': int(len(parsed_models)),
            'mean_ari': mean_ari,
            'ci_low_95': ci_low,
            'ci_high_95': ci_high,
            'ci_method': 'percentile',
            'bootstrap_strategy': 'within_cell_resample_fixed_n',
            'cell_definition': '(developer, model, text, persona)',
        }])

        self.results['clustering_stability_ari_samples'] = df_samples
        self.results['clustering_stability_ari_summary'] = df_summary

        return df_samples, df_summary

    def perform_tsne(self, X: np.ndarray = None) -> np.ndarray:
        """Perform t-SNE dimensional reduction.

        Args:
            X: Feature matrix (if None, use stored matrix)

        Returns:
            2D embedding coordinates
        """
        if X is None:
            X = self.results.get('feature_matrix')

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply t-SNE
        tsne = TSNE(**TSNE_PARAMS)
        X_embedded = tsne.fit_transform(X_scaled)

        self.results['tsne_embedding'] = X_embedded

        return X_embedded

    def perform_hierarchical_clustering(
        self,
        X: np.ndarray = None,
        n_clusters: int = 7
    ) -> np.ndarray:
        """Perform hierarchical clustering.

        Args:
            X: Feature matrix (if None, use stored matrix)
            n_clusters: Number of clusters

        Returns:
            Cluster labels
        """
        if X is None:
            X = self.results.get('feature_matrix')

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(X_scaled)

        self.results['cluster_labels'] = labels
        self.results['n_clusters'] = n_clusters

        return labels

    def calculate_cluster_distances(
        self,
        X: np.ndarray = None,
        labels: np.ndarray = None
    ) -> Dict:
        """Calculate intra-cluster and inter-cluster distances.

        Args:
            X: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary with distance statistics
        """
        if X is None:
            X = self.results.get('feature_matrix')
        if labels is None:
            labels = self.results.get('cluster_labels')

        # Calculate pairwise distances
        distances = squareform(pdist(X, metric='euclidean'))

        # Intra-cluster distances
        intra_distances = {}
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 1:
                cluster_dists = distances[np.ix_(cluster_indices, cluster_indices)]
                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices_from(cluster_dists, k=1)
                intra_dists = cluster_dists[triu_indices]

                intra_distances[f'Cluster {cluster_id}'] = {
                    'mean': float(np.mean(intra_dists)),
                    'std': float(np.std(intra_dists)),
                    'min': float(np.min(intra_dists)),
                    'max': float(np.max(intra_dists)),
                    'n_models': int(np.sum(cluster_mask))
                }

        # Inter-cluster distances
        inter_distances = {}
        cluster_ids = np.unique(labels)
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                c1_indices = np.where(labels == c1)[0]
                c2_indices = np.where(labels == c2)[0]

                inter_dists = distances[np.ix_(c1_indices, c2_indices)].flatten()

                inter_distances[f'Cluster {c1} - Cluster {c2}'] = {
                    'mean': float(np.mean(inter_dists)),
                    'std': float(np.std(inter_dists)),
                    'min': float(np.min(inter_dists)),
                    'max': float(np.max(inter_dists))
                }

        distance_stats = {
            'intra_cluster': intra_distances,
            'inter_cluster': inter_distances
        }

        self.results['cluster_distances'] = distance_stats

        return distance_stats

    def calculate_consistency_scores(self) -> pd.DataFrame:
        """Calculate language-numerical consistency scores.

        This is a simplified implementation. In a full implementation,
        you would analyze the correlation between numerical scores and
        sentiment/emotion expressed in the text reasons.

        Returns:
            DataFrame with consistency scores by model
        """
        consistency_scores = []

        for (developer, model), group in self.df.groupby(['developer', 'model']):
            # For now, calculate a proxy consistency score based on
            # variability of responses (lower variance = more consistent)

            # Calculate coefficient of variation for each emotion
            cvs = []
            for col in EMOTION_COLUMNS:
                mean_val = group[col].mean()
                std_val = group[col].std()
                if mean_val > 0:
                    cv = std_val / mean_val
                    cvs.append(cv)

            # Consistency score: inverse of mean CV (normalized to [0, 1])
            if cvs:
                mean_cv = np.mean(cvs)
                # Transform CV to consistency score (lower CV = higher consistency)
                consistency = 1 / (1 + mean_cv)  # Maps [0, inf) -> [0, 1]
            else:
                consistency = 0

            consistency_scores.append({
                'developer': developer,
                'model': model,
                'consistency_score': consistency,
                'mean_cv': np.mean(cvs) if cvs else np.nan
            })

        df_consistency = pd.DataFrame(consistency_scores)
        df_consistency = df_consistency.sort_values('consistency_score', ascending=False)

        self.results['consistency_scores'] = df_consistency

        return df_consistency

    def calculate_vendor_consistency(self) -> pd.DataFrame:
        """Calculate vendor-level consistency statistics.

        Returns:
            DataFrame with vendor consistency metrics
        """
        consistency_df = self.results.get('consistency_scores')

        if consistency_df is None:
            consistency_df = self.calculate_consistency_scores()

        vendor_stats = consistency_df.groupby('developer').agg({
            'consistency_score': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()

        vendor_stats.columns = [
            'developer', 'mean_consistency', 'std_consistency',
            'min_consistency', 'max_consistency', 'n_models'
        ]

        vendor_stats = vendor_stats.sort_values('mean_consistency', ascending=False)

        self.results['vendor_consistency'] = vendor_stats

        return vendor_stats

    def assign_clusters_to_vendors(
        self,
        labels: np.ndarray = None,
        model_labels: List[str] = None
    ) -> pd.DataFrame:
        """Assign cluster labels to each model and analyze vendor distribution.

        Args:
            labels: Cluster labels
            model_labels: Model identifiers

        Returns:
            DataFrame with model-cluster assignments
        """
        if labels is None:
            labels = self.results.get('cluster_labels')
        if model_labels is None:
            model_labels = self.results.get('model_labels')

        # Parse developer and model from labels
        assignments = []
        for model_label, cluster in zip(model_labels, labels):
            developer, model = model_label.split('/', 1)
            assignments.append({
                'developer': developer,
                'model': model,
                'cluster': cluster
            })

        df_assignments = pd.DataFrame(assignments)
        self.results['cluster_assignments'] = df_assignments

        return df_assignments

    def run_all_analyses(self) -> Dict:
        """Run all clustering and consistency analyses.

        Returns:
            Dictionary with all analysis results
        """
        print("Running Clustering Analysis (Section 3.4)...")

        print("  - Preparing feature matrix...")
        X, model_labels = self.prepare_model_feature_matrix()
        print(f"    Feature matrix: {X.shape}")

        print("  - Performing t-SNE...")
        self.perform_tsne(X)

        print("  - Performing hierarchical clustering...")
        labels = self.perform_hierarchical_clustering(X, n_clusters=5)

        print("  - Calculating cluster distances...")
        self.calculate_cluster_distances(X, labels)

        print("  - Estimating clustering stability (bootstrap ARI)...")
        self.calculate_cluster_stability_ari(n_bootstrap=1000, n_clusters=self.results.get('n_clusters', 5))

        print("  - Assigning clusters to vendors...")
        self.assign_clusters_to_vendors(labels, model_labels)

        print("  - Calculating consistency scores...")
        self.calculate_consistency_scores()

        print("  - Calculating vendor consistency...")
        self.calculate_vendor_consistency()

        print("✓ Clustering analysis complete")

        return self.results

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for clustering analysis.

        Returns:
            Dictionary with key findings
        """
        summary = {}

        # Cluster statistics
        if 'cluster_distances' in self.results:
            cluster_dists = self.results['cluster_distances']
            intra = cluster_dists['intra_cluster']

            summary['clustering'] = {
                'n_clusters': self.results.get('n_clusters', 0),
                'mean_intra_distance': np.mean([v['mean'] for v in intra.values()]),
                'cluster_sizes': {k: v['n_models'] for k, v in intra.items()}
            }

        # Consistency statistics
        consistency_df = self.results.get('consistency_scores')
        if consistency_df is not None:
            summary['consistency'] = {
                'mean': consistency_df['consistency_score'].mean(),
                'std': consistency_df['consistency_score'].std(),
                'min': consistency_df['consistency_score'].min(),
                'max': consistency_df['consistency_score'].max(),
                'range': (
                    consistency_df['consistency_score'].min(),
                    consistency_df['consistency_score'].max()
                )
            }

        return summary


if __name__ == "__main__":
    from .data_loader import load_and_validate_data

    print("Loading data...")
    df, _ = load_and_validate_data()

    print("\nRunning clustering analysis...")
    analyzer = ClusteringAnalyzer(df)
    results = analyzer.run_all_analyses()

    print("\n=== Summary Statistics ===")
    summary = analyzer.get_summary_statistics()

    if 'clustering' in summary:
        print(f"\nClustering:")
        print(f"  Number of clusters: {summary['clustering']['n_clusters']}")
        print(f"  Mean intra-cluster distance: {summary['clustering']['mean_intra_distance']:.3f}")

    if 'consistency' in summary:
        cons = summary['consistency']
        print(f"\nConsistency:")
        print(f"  Range: {cons['range'][0]:.3f} - {cons['range'][1]:.3f}")
        print(f"  Mean: {cons['mean']:.3f}")
