"""Nested DBSCAN discovery followed by centroid assignment and noise flagging.

This experimental method is a heuristic for exploratory cohort analysis, not a
clinically validated model. Density groups are found in a PCA projection; their
centroids seed K-Means in the full standardized feature space.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    labels_: np.ndarray
    stage1_labels_: np.ndarray
    stage2_labels_: np.ndarray
    centroids_: np.ndarray
    n_cohorts_: int
    n_noise_: int


class NestedDBSCANCentroidKMeans:
    """Two density passes seed a full-space centroid refinement."""

    def __init__(self, eps1=0.6, min_samples1=12, eps2=0.25,
                 min_samples2=10, pca_components=2, noise_reassign_std=2.5,
                 scale=True, random_state=42):
        self.eps1 = eps1
        self.min_samples1 = min_samples1
        self.eps2 = eps2
        self.min_samples2 = min_samples2
        self.pca_components = pca_components
        self.noise_reassign_std = noise_reassign_std
        self.scale = scale
        self.random_state = random_state
        self.scaler_ = None
        self.pca_ = None
        self.result_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise ValueError("X must be a 2D array with at least two rows and one feature.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values; impute missing values first.")
        if self.eps1 <= 0 or self.eps2 <= 0:
            raise ValueError("DBSCAN radii must be greater than zero.")
        if self.min_samples1 < 2 or self.min_samples2 < 2:
            raise ValueError("DBSCAN min_samples values must be at least two.")
        if self.noise_reassign_std < 0:
            raise ValueError("Noise rejection tolerance cannot be negative.")

        n = X.shape[0]
        if self.scale:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            Xs = X.copy()
        components = min(int(self.pca_components), Xs.shape[1], Xs.shape[0] - 1)
        if components < 1:
            raise ValueError("PCA needs at least one component.")
        self.pca_ = PCA(n_components=components, random_state=self.random_state)
        Xp = self.pca_.fit_transform(Xs)

        stage1_labels = DBSCAN(eps=self.eps1, min_samples=self.min_samples1).fit_predict(Xp)
        survivor_mask = stage1_labels != -1
        stage2_labels_full = np.full(n, -1, dtype=int)
        if survivor_mask.sum() >= self.min_samples2:
            stage2_labels_full[survivor_mask] = DBSCAN(
                eps=self.eps2, min_samples=self.min_samples2
            ).fit_predict(Xp[survivor_mask])

        clean_ids = sorted(set(stage2_labels_full.tolist()) - {-1})
        if not clean_ids:
            self.result_ = ClusteringResult(np.full(n, -1, dtype=int), stage1_labels,
                stage2_labels_full, np.empty((0, Xs.shape[1])), 0, n)
            return self

        init_centroids = np.vstack([Xs[stage2_labels_full == cid].mean(axis=0) for cid in clean_ids])
        km = KMeans(n_clusters=len(clean_ids), init=init_centroids, n_init=1,
                    random_state=self.random_state)
        km_labels = km.fit_predict(Xs)
        centroids = km.cluster_centers_
        distances = np.linalg.norm(Xs - centroids[km_labels], axis=1)
        final_labels = km_labels.copy()

        for k, cid in enumerate(clean_ids):
            clean_mask = stage2_labels_full == cid
            clean_distances = np.linalg.norm(Xs[clean_mask] - centroids[k], axis=1)
            # A minimum tolerance avoids rejecting the seed cohort when it is tiny.
            threshold = clean_distances.mean() + self.noise_reassign_std * clean_distances.std()
            threshold = max(float(threshold), float(np.quantile(clean_distances, 0.95)), 1e-12)
            final_labels[(km_labels == k) & (distances > threshold)] = -1

        self.result_ = ClusteringResult(final_labels, stage1_labels, stage2_labels_full,
            centroids, len(clean_ids), int(np.sum(final_labels == -1)))
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.result_.labels_

    def inverse_transform_centroids(self):
        if self.result_ is None:
            raise ValueError("Call fit before requesting centroids.")
        if self.scaler_ is not None:
            return self.scaler_.inverse_transform(self.result_.centroids_)
        return self.result_.centroids_
