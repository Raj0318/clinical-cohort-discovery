"""Validated orchestration and descriptive metrics for cohort exploration."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from .clustering import NestedDBSCANCentroidKMeans


@dataclass
class PipelineOutput:
    features: list[str]
    Xs: np.ndarray
    X_2d: np.ndarray
    true_labels: np.ndarray | None
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cohort_profile_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def safe_internal_metrics(X: np.ndarray, labels: np.ndarray, name: str) -> dict:
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = np.unique(labels[mask]).size
    row = {"method": name, "n_clusters_found": int(n_clusters),
           "n_noise_points": int((~mask).sum())}
    if n_clusters >= 2 and mask.sum() > n_clusters:
        row["silhouette"] = round(float(silhouette_score(
            X[mask], labels[mask], sample_size=min(5_000, int(mask.sum())), random_state=42
        )), 4)
        row["davies_bouldin"] = round(float(davies_bouldin_score(X[mask], labels[mask])), 4)
        row["calinski_harabasz"] = round(float(calinski_harabasz_score(X[mask], labels[mask])), 2)
    else:
        row.update(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return row


def external_metrics(true_labels, predicted, name: str, enabled: bool) -> dict:
    if not enabled:
        return {"method": name, "ARI_vs_ground_truth": np.nan, "NMI_vs_ground_truth": np.nan}
    return {"method": name,
            "ARI_vs_ground_truth": round(float(adjusted_rand_score(true_labels, predicted)), 4),
            "NMI_vs_ground_truth": round(float(normalized_mutual_info_score(true_labels, predicted)), 4)}


def run_pipeline(df: pd.DataFrame, features: list[str], true_label_col: str | None,
                 kmeans_k: int, dbscan_eps: float, dbscan_min_samples: int,
                 nested_eps1: float, nested_min_samples1: int, nested_eps2: float,
                 nested_min_samples2: int, nested_pca_components: int,
                 nested_noise_std: float, random_state: int = 42) -> PipelineOutput:
    """Run three clustering comparisons after validating and preparing features."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Provide a non-empty patient table.")
    if not features or len(set(features)) != len(features):
        raise ValueError("Select one or more distinct numeric features.")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Selected feature columns are missing: {', '.join(missing)}.")
    if len(df) < 3:
        raise ValueError("At least three records are required for cohort analysis.")
    if len(features) < 2:
        raise ValueError("Select at least two numeric features so the projection is meaningful.")
    if not 2 <= int(kmeans_k) <= len(df):
        raise ValueError("K-Means cohort count must be between 2 and the record count.")
    try:
        X = df.loc[:, features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every selected feature must contain numeric values.") from exc
    X[~np.isfinite(X)] = np.nan
    all_missing = [features[i] for i in range(X.shape[1]) if np.isnan(X[:, i]).all()]
    if all_missing:
        raise ValueError(f"These selected features contain no usable values: {', '.join(all_missing)}.")
    X = SimpleImputer(strategy="median").fit_transform(X)
    varying = np.nanstd(X, axis=0) > 1e-12
    X, used_features = X[:, varying], [name for name, keep in zip(features, varying) if keep]
    if len(used_features) < 2:
        raise ValueError("At least two selected features must contain varying values.")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if not np.isfinite(Xs).all():
        raise ValueError("Feature preparation produced invalid values; check the uploaded data.")

    km_labels = KMeans(n_clusters=kmeans_k, n_init=10, random_state=random_state).fit_predict(Xs)
    db_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(Xs)
    proposed = NestedDBSCANCentroidKMeans(
        eps1=nested_eps1, min_samples1=nested_min_samples1,
        eps2=nested_eps2, min_samples2=nested_min_samples2,
        pca_components=min(int(nested_pca_components), Xs.shape[1], Xs.shape[0] - 1),
        noise_reassign_std=nested_noise_std, scale=False, random_state=random_state)
    proposed_labels = proposed.fit_predict(Xs)
    labels = {"Plain K-Means": km_labels, "Plain DBSCAN": db_labels,
              "Proposed: Nested DBSCAN + Centroid K-Means": proposed_labels}

    has_truth = true_label_col is not None and true_label_col in df.columns
    truth = df[true_label_col].to_numpy() if has_truth else None
    rows = []
    for name, predicted in labels.items():
        rows.append({**safe_internal_metrics(Xs, predicted, name),
                     **external_metrics(truth, predicted, name, has_truth)})
    metric_df = pd.DataFrame(rows)

    profile = df.loc[:, used_features].copy()
    profile["proposed_cohort"] = proposed_labels
    profile_df = profile.groupby("proposed_cohort", sort=True)[used_features].mean().round(2)
    profile_df["n_patients"] = profile.groupby("proposed_cohort").size()

    projection = PCA(n_components=2, random_state=random_state).fit_transform(Xs)
    return PipelineOutput(used_features, Xs, projection, truth, labels, metric_df, profile_df)
