"""
pipeline.py
-----------
Pure (no file I/O, no plotting) pipeline functions shared by the
Streamlit app and any CLI/script usage. Keeping this separate from
app.py means the ML logic can be unit-tested without Streamlit
installed.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

from .clustering import NestedDBSCANCentroidKMeans


@dataclass
class PipelineOutput:
    features: list
    Xs: np.ndarray                  # standardized features
    X_2d: np.ndarray                # PCA(2) projection, for plotting
    true_labels: np.ndarray
    labels: dict = field(default_factory=dict)       # method name -> label array
    metrics_df: pd.DataFrame = None
    cohort_profile_df: pd.DataFrame = None


def safe_internal_metrics(X, labels, name):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    row = {"method": name, "n_clusters_found": n_clusters,
           "n_noise_points": int((labels == -1).sum())}
    if n_clusters >= 2 and mask.sum() > n_clusters:
        row["silhouette"] = round(silhouette_score(X[mask], labels[mask]), 4)
        row["davies_bouldin"] = round(davies_bouldin_score(X[mask], labels[mask]), 4)
        row["calinski_harabasz"] = round(calinski_harabasz_score(X[mask], labels[mask]), 2)
    else:
        row["silhouette"] = row["davies_bouldin"] = row["calinski_harabasz"] = np.nan
    return row


def external_metrics(true_labels, pred_labels, name, has_ground_truth: bool):
    if not has_ground_truth:
        return {"method": name, "ARI_vs_ground_truth": np.nan, "NMI_vs_ground_truth": np.nan}
    return {
        "method": name,
        "ARI_vs_ground_truth": round(adjusted_rand_score(true_labels, pred_labels), 4),
        "NMI_vs_ground_truth": round(normalized_mutual_info_score(true_labels, pred_labels), 4),
    }


def run_pipeline(
    df: pd.DataFrame,
    features: list,
    true_label_col: str | None,
    kmeans_k: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    nested_eps1: float,
    nested_min_samples1: int,
    nested_eps2: float,
    nested_min_samples2: int,
    nested_pca_components: int,
    nested_noise_std: float,
    random_state: int = 42,
) -> PipelineOutput:
    """Run Plain K-Means, Plain DBSCAN, and the Proposed method on `df`,
    and compute internal (+ optional external) validity metrics."""

    X = df[features].values
    has_ground_truth = true_label_col is not None and true_label_col in df.columns
    true_labels = df[true_label_col].values if has_ground_truth else np.full(len(df), -1)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- Plain K-Means ----
    km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=random_state)
    km_labels = km.fit_predict(Xs)

    # ---- Plain DBSCAN ----
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    db_labels = db.fit_predict(Xs)

    # ---- Proposed: Nested DBSCAN + Centroid K-Means ----
    proposed = NestedDBSCANCentroidKMeans(
        eps1=nested_eps1, min_samples1=nested_min_samples1,
        eps2=nested_eps2, min_samples2=nested_min_samples2,
        pca_components=nested_pca_components,
        noise_reassign_std=nested_noise_std,
        scale=False,  # already scaled above
        random_state=random_state,
    )
    proposed_labels = proposed.fit_predict(Xs)

    labels = {
        "Plain K-Means": km_labels,
        "Plain DBSCAN": db_labels,
        "Proposed: Nested DBSCAN + Centroid K-Means": proposed_labels,
    }

    # ---- metrics ----
    rows = []
    for name, lab in labels.items():
        internal = safe_internal_metrics(Xs, lab, name)
        external = external_metrics(true_labels, lab, name, has_ground_truth)
        rows.append({**internal, **external})
    metrics_df = pd.DataFrame(rows)

    # ---- cohort clinical profile table (proposed method) ----
    profile_df = df.copy()
    profile_df["proposed_cohort"] = proposed_labels
    cohort_profile_df = profile_df.groupby("proposed_cohort")[features].mean().round(2)
    cohort_profile_df["n_patients"] = profile_df.groupby("proposed_cohort").size()

    # ---- 2D projection for visualization ----
    pca2 = PCA(n_components=2, random_state=random_state)
    X_2d = pca2.fit_transform(Xs)

    return PipelineOutput(
        features=features,
        Xs=Xs,
        X_2d=X_2d,
        true_labels=true_labels if has_ground_truth else None,
        labels=labels,
        metrics_df=metrics_df,
        cohort_profile_df=cohort_profile_df,
    )
