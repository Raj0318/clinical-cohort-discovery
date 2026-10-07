"""
clustering.py
-------------
Core contribution: NestedDBSCANCentroidKMeans

A two-stage, noise-robust clustering algorithm for clinical patient
cohort discovery.

WHY THIS DESIGN
----------------
Clinical feature vectors (Age, BMI, BP, Glucose, Cholesterol, ...) are
high-dimensional and correlated, which flattens local density contrast
and makes plain DBSCAN unstable (curse of dimensionality: in 8-D
standardized space, "dense" and "sparse" regions become hard to tell
apart, so DBSCAN either lumps everything into one giant cluster or
shatters it into many tiny ones).

Fix: use PCA to obtain a low-dimensional "density-estimation" view of
the data purely for the two DBSCAN passes (this is where density
actually means something geometrically), while keeping ALL original
clinical features for centroid computation and the final K-Means
assignment (so cohort profiles stay clinically interpretable in real
units, not PCA components).

STAGE 1 - Nested DBSCAN (coarse -> fine), run in PCA-reduced space
    Level 1 (coarse): large eps1/min_samples1 DBSCAN strips gross
        global outliers.
    Level 2 (fine): smaller eps2/min_samples2 DBSCAN, run ONLY on
        Level-1 survivors, discovers the fine-grained sub-cohorts.
    Nesting two passes (coarse -> fine) prevents a handful of noisy
    points from bridging two real clusters together.

STAGE 2 - Centroid K-Means (refinement + noise rescue), run in the
    FULL original (scaled) feature space
    Stage-2 cluster centroids are computed in the ORIGINAL feature
    space and used as the initial centroids for K-Means run over the
    FULL dataset (including Stage-1 noise). This "rescues" borderline
    points while points far from every centroid (beyond
    noise_reassign_std standard deviations) are kept flagged as noise.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class ClusteringResult:
    labels_: np.ndarray            # final cohort labels (-1 = noise)
    stage1_labels_: np.ndarray     # coarse DBSCAN labels (PCA space)
    stage2_labels_: np.ndarray     # fine DBSCAN labels (PCA space, on stage1 survivors)
    centroids_: np.ndarray         # final K-Means centroids (full scaled feature space)
    n_cohorts_: int
    n_noise_: int


class NestedDBSCANCentroidKMeans:
    def __init__(
        self,
        eps1: float = 0.6,
        min_samples1: int = 12,
        eps2: float = 0.25,
        min_samples2: int = 10,
        pca_components: int = 2,
        noise_reassign_std: float = 2.5,
        scale: bool = True,
        random_state: int = 42,
    ):
        """
        eps1, min_samples1 : coarse (Level-1) DBSCAN params, applied in
            PCA-reduced space - strips gross outliers.
        eps2, min_samples2 : fine (Level-2) DBSCAN params, applied in
            PCA-reduced space on Level-1 survivors - discovers cohorts.
        pca_components : components used ONLY for the DBSCAN density
            passes. Centroids and K-Means always use full features.
        noise_reassign_std : a point stays flagged noise if its
            distance to its assigned centroid exceeds
            (mean + noise_reassign_std * std) of that cohort's
            original clean-point spread. Lower = stricter.
        scale : standardize features before clustering (recommended).
        """
        self.eps1 = eps1
        self.min_samples1 = min_samples1
        self.eps2 = eps2
        self.min_samples2 = min_samples2
        self.pca_components = pca_components
        self.noise_reassign_std = noise_reassign_std
        self.scale = scale
        self.random_state = random_state

        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.result_: Optional[ClusteringResult] = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if self.scale:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            Xs = X.copy()

        self.pca_ = PCA(n_components=self.pca_components, random_state=self.random_state)
        Xp = self.pca_.fit_transform(Xs)

        # ---- Stage 1: coarse DBSCAN (global outlier removal) ----
        stage1 = DBSCAN(eps=self.eps1, min_samples=self.min_samples1)
        stage1_labels = stage1.fit_predict(Xp)
        survivor_mask = stage1_labels != -1

        # ---- Stage 2: fine DBSCAN on survivors only ----
        stage2_labels_full = np.full(n, -1)
        if survivor_mask.sum() >= self.min_samples2:
            stage2 = DBSCAN(eps=self.eps2, min_samples=self.min_samples2)
            fine_labels = stage2.fit_predict(Xp[survivor_mask])
            stage2_labels_full[survivor_mask] = fine_labels

        clean_cluster_ids = sorted(set(stage2_labels_full) - {-1})
        n_cohorts = len(clean_cluster_ids)

        if n_cohorts == 0:
            self.result_ = ClusteringResult(
                labels_=np.full(n, -1),
                stage1_labels_=stage1_labels,
                stage2_labels_=stage2_labels_full,
                centroids_=np.empty((0, Xs.shape[1])),
                n_cohorts_=0,
                n_noise_=n,
            )
            return self

        # ---- centroids in FULL feature space ----
        init_centroids = np.array([
            Xs[stage2_labels_full == cid].mean(axis=0)
            for cid in clean_cluster_ids
        ])

        # ---- Stage 3: Centroid-initialized K-Means on FULL data ----
        kmeans = KMeans(
            n_clusters=n_cohorts,
            init=init_centroids,
            n_init=1,
            random_state=self.random_state,
        )
        km_labels = kmeans.fit_predict(Xs)
        centroids = kmeans.cluster_centers_

        # ---- Stage 4: noise-robust reassignment / rejection ----
        final_labels = km_labels.copy()
        dists_to_own_centroid = np.linalg.norm(Xs - centroids[km_labels], axis=1)

        for k, cid in enumerate(clean_cluster_ids):
            clean_mask = stage2_labels_full == cid
            if clean_mask.sum() < 2:
                continue
            clean_dists = np.linalg.norm(Xs[clean_mask] - centroids[k], axis=1)
            thresh = clean_dists.mean() + self.noise_reassign_std * clean_dists.std()
            this_cluster_mask = km_labels == k
            too_far = this_cluster_mask & (dists_to_own_centroid > thresh)
            final_labels[too_far] = -1

        self.result_ = ClusteringResult(
            labels_=final_labels,
            stage1_labels_=stage1_labels,
            stage2_labels_=stage2_labels_full,
            centroids_=centroids,
            n_cohorts_=n_cohorts,
            n_noise_=int((final_labels == -1).sum()),
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.result_.labels_

    def inverse_transform_centroids(self) -> np.ndarray:
        if self.scaler_ is not None:
            return self.scaler_.inverse_transform(self.result_.centroids_)
        return self.result_.centroids_
