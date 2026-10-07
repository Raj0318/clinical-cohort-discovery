"""
app.py
------
Streamlit UI for the Noise-Robust Clinical Patient Cohort Discovery
System (Nested DBSCAN + Centroid K-Means).

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

from src.data_generator import generate_clinical_dataset, FEATURES as DEFAULT_FEATURES
from src.pipeline import run_pipeline

st.set_page_config(page_title="Clinical Cohort Discovery", layout="wide")

# ----------------------------------------------------------------------
# Sidebar - data source + algorithm parameters
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

data_source = st.sidebar.radio("Data source", ["Synthetic demo data", "Upload my own CSV"])

if data_source == "Synthetic demo data":
    st.sidebar.subheader("Synthetic data settings")
    n_samples = st.sidebar.slider("Number of patients", 200, 3000, 1200, step=100)
    noise_fraction = st.sidebar.slider("Noise fraction", 0.0, 0.3, 0.08, step=0.01)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    df = generate_clinical_dataset(n_samples=n_samples, noise_fraction=noise_fraction, random_state=int(seed))
    features = DEFAULT_FEATURES
    true_label_col = "true_cohort_id"
else:
    uploaded = st.sidebar.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is None:
        st.info("⬅️ Upload a CSV from the sidebar to get started, or switch to synthetic demo data.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.sidebar.subheader("Select numeric feature columns")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = st.sidebar.multiselect("Features to cluster on", numeric_cols, default=numeric_cols)
    has_truth = st.sidebar.checkbox("I have a ground-truth cohort column (for evaluation only)")
    true_label_col = None
    if has_truth:
        true_label_col = st.sidebar.selectbox("Ground-truth column", df.columns.tolist())
    if not features:
        st.warning("Select at least one numeric feature column to continue.")
        st.stop()

st.sidebar.subheader("Baseline: Plain K-Means")
kmeans_k = st.sidebar.slider("k (number of clusters)", 2, 10, 5)

st.sidebar.subheader("Baseline: Plain DBSCAN")
dbscan_eps = st.sidebar.slider("eps", 0.1, 3.0, 0.9, step=0.05)
dbscan_min_samples = st.sidebar.slider("min_samples", 2, 30, 10)

st.sidebar.subheader("Proposed: Nested DBSCAN + Centroid K-Means")
with st.sidebar.expander("Stage 1 — coarse DBSCAN (outlier strip)", expanded=False):
    eps1 = st.slider("eps1", 0.1, 3.0, 0.6, step=0.05, key="eps1")
    min_samples1 = st.slider("min_samples1", 2, 30, 12, key="ms1")
with st.sidebar.expander("Stage 2 — fine DBSCAN (cohort discovery)", expanded=False):
    eps2 = st.slider("eps2", 0.05, 2.0, 0.25, step=0.05, key="eps2")
    min_samples2 = st.slider("min_samples2", 2, 30, 10, key="ms2")
pca_components = st.sidebar.slider("PCA components (density view only)", 2, min(len(features), 6), 2)
noise_std = st.sidebar.slider("Noise rejection strictness (std)", 1.0, 4.0, 2.5, step=0.1,
                               help="Lower = more points kept as noise. Higher = more points rescued into a cohort.")

run_btn = st.sidebar.button("🚀 Run clustering", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏥 Noise-Robust Clinical Patient Cohort Discovery")
st.caption("Nested DBSCAN + Centroid-Initialized K-Means — a noise-robust hybrid clustering pipeline")

with st.expander("ℹ️ How the proposed algorithm works", expanded=False):
    st.markdown("""
1. **Standardize** features so BP, Glucose, Age etc. are comparable.
2. **PCA (density view only)** — DBSCAN struggles in high dimensions (curse of
   dimensionality), so a low-dimensional PCA projection is used *only* to help
   DBSCAN judge density correctly. Centroids and final assignment still use
   all original features, so results stay clinically interpretable.
3. **Stage 1 — coarse DBSCAN**: strips gross outliers using a wide `eps1`.
4. **Stage 2 — fine DBSCAN**: run only on Stage-1 survivors with a tighter
   `eps2`, discovering the fine-grained patient cohorts.
5. **Centroid K-Means**: the Stage-2 cluster centroids (computed in full
   feature space) seed a K-Means pass over the *entire* dataset — rescuing
   borderline noisy points into their correct cohort.
6. **Noise-robust rejection**: points still far from every centroid
   (beyond a configurable number of standard deviations) are kept flagged
   as noise rather than force-fit into a cohort.
    """)

st.subheader("📋 Data preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

if not run_btn:
    st.info("Adjust parameters in the sidebar, then click **🚀 Run clustering**.")
    st.stop()

# ----------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------
with st.spinner("Running Plain K-Means, Plain DBSCAN, and the Proposed method..."):
    out = run_pipeline(
        df=df,
        features=features,
        true_label_col=true_label_col,
        kmeans_k=kmeans_k,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        nested_eps1=eps1,
        nested_min_samples1=min_samples1,
        nested_eps2=eps2,
        nested_min_samples2=min_samples2,
        nested_pca_components=pca_components,
        nested_noise_std=noise_std,
    )

st.success("Done!")

# ----------------------------------------------------------------------
# Metrics table
# ----------------------------------------------------------------------
st.subheader("📊 Clustering quality comparison")
st.dataframe(out.metrics_df, use_container_width=True)
if true_label_col is None:
    st.caption("No ground-truth column provided — ARI/NMI columns are not applicable (NaN).")


def scatter_by_label(ax, X_2d, labels, title):
    unique_labels = sorted(set(labels))
    colors = colormaps["tab10"].resampled(max(len(unique_labels), 1))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        if lab == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=14, c="lightgrey", marker="x",
                       label="Noise", alpha=0.7)
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=14, c=[colors(i)], marker="o",
                       label=f"Cohort {lab}", alpha=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(fontsize=7, loc="best")


# ----------------------------------------------------------------------
# Cluster visualizations
# ----------------------------------------------------------------------
st.subheader("🗺️ Cluster visualizations (PCA 2D projection)")

n_panels = len(out.labels) + (1 if out.true_labels is not None else 0)
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

panel_idx = 0
if out.true_labels is not None:
    scatter_by_label(axes[panel_idx], out.X_2d, out.true_labels, "Ground Truth")
    panel_idx += 1
for name, lab in out.labels.items():
    scatter_by_label(axes[panel_idx], out.X_2d, lab, name)
    panel_idx += 1

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Cohort clinical profiles (proposed method)
# ----------------------------------------------------------------------
st.subheader("🩺 Discovered cohort clinical profiles (Proposed method)")
st.caption("Row `-1` = patients flagged as noise/outliers and not assigned to any cohort.")
st.dataframe(out.cohort_profile_df, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(min(1.2 * len(out.features) + 2, 14), 0.6 * len(out.cohort_profile_df) + 2))
profile_vals = out.cohort_profile_df[out.features]
z = (profile_vals - profile_vals.mean()) / profile_vals.std().replace(0, 1)
im = ax2.imshow(z.values, cmap="coolwarm", aspect="auto")
ax2.set_xticks(range(len(out.features)))
ax2.set_xticklabels(out.features, rotation=45, ha="right")
ax2.set_yticks(range(len(out.cohort_profile_df)))
ax2.set_yticklabels(out.cohort_profile_df.index.tolist())
ax2.set_ylabel("Cohort ID (-1 = noise)")
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax2.text(j, i, f"{profile_vals.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
fig2.colorbar(im, ax=ax2, label="Z-score vs other cohorts")
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)

# ----------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------
st.subheader("⬇️ Downloads")
col1, col2, col3 = st.columns(3)

result_df = df.copy()
for name, lab in out.labels.items():
    result_df[f"cluster__{name.split(':')[0].strip().replace(' ', '_')}"] = lab

with col1:
    st.download_button("Download clustered dataset (CSV)",
                        result_df.to_csv(index=False).encode(),
                        file_name="clustered_patients.csv", mime="text/csv",
                        use_container_width=True)
with col2:
    st.download_button("Download metrics comparison (CSV)",
                        out.metrics_df.to_csv(index=False).encode(),
                        file_name="metrics_comparison.csv", mime="text/csv",
                        use_container_width=True)
with col3:
    st.download_button("Download cohort profiles (CSV)",
                        out.cohort_profile_df.to_csv().encode(),
                        file_name="cohort_clinical_profiles.csv", mime="text/csv",
                        use_container_width=True)
