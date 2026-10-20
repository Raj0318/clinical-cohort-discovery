"""Clinical Cohort Studio — privacy-conscious exploratory cohort analysis."""

from __future__ import annotations

import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps

from src.data_generator import FEATURES as DEFAULT_FEATURES, generate_clinical_dataset
from src.pipeline import run_pipeline

st.set_page_config(page_title="Cohort Studio | Clinical Discovery", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e9f1ff; --muted:#8ea2bd; --panel:rgba(15,27,48,.78); --line:rgba(159,190,233,.13); }
html, body, [class*="css"] { font-family:Manrope, sans-serif; }
.stApp { color:var(--ink); background:radial-gradient(ellipse at 12% 6%,rgba(26,113,152,.24),transparent 38%),radial-gradient(ellipse at 85% 20%,rgba(108,67,181,.21),transparent 35%),linear-gradient(145deg,#07111f 0%,#0a1425 50%,#11162a 100%); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:rgba(7,15,28,.91); border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
.hero { padding:30px 32px; border:1px solid var(--line); border-radius:24px; background:linear-gradient(115deg,rgba(21,48,73,.88),rgba(25,26,52,.77)); box-shadow:0 24px 80px rgba(0,0,0,.2); margin:12px 0 26px; }
.eyebrow { color:#70e0d0; text-transform:uppercase; letter-spacing:.17em; font:500 11px 'DM Mono',monospace; }
.hero h1 { font-size:clamp(32px,5vw,54px); line-height:1.07; letter-spacing:-.055em; margin:12px 0; }
.hero p { color:#a9bbd1; font-size:15px; max-width:730px; margin:0; }
.section-kicker { color:#70e0d0; font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.15em; }
div[data-testid="stDataFrame"], .stPlotlyChart { border:1px solid var(--line); border-radius:15px; overflow:hidden; }
button[kind="primary"] { border-radius:12px; background:linear-gradient(100deg,#14a99a,#5976db); border:0; }
.fineprint { color:#8092aa; font-size:12px; }
</style>
""", unsafe_allow_html=True)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "method"


def csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Neutralize spreadsheet formula prefixes in untrusted text cells."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r") else value
        )
    return safe


st.sidebar.markdown("## ◈ COHORT STUDIO")
st.sidebar.caption("Clinical discovery workspace · local session")
source = st.sidebar.radio("DATA SOURCE", ["Synthetic demo", "Upload CSV"], label_visibility="collapsed")
df: pd.DataFrame | None = None
features: list[str] = []
truth_col: str | None = None

if source == "Synthetic demo":
    n = st.sidebar.slider("Patient records", 200, 3000, 1200, 100)
    noise = st.sidebar.slider("Injected noise", 0.0, 0.30, 0.08, 0.01)
    seed = int(st.sidebar.number_input("Reproducibility seed", min_value=0, value=42, step=1))
    df = generate_clinical_dataset(n_samples=n, noise_fraction=noise, random_state=seed)
    features = DEFAULT_FEATURES
    truth_col = "true_cohort_id"
else:
    upload = st.sidebar.file_uploader("Choose a patient CSV", type=["csv"], help="Processed in this app session. Avoid uploading direct identifiers.")
    if upload is not None:
        try:
            if upload.size > 25 * 1024 * 1024:
                st.error("This upload exceeds the 25 MB limit.")
                st.stop()
            df = pd.read_csv(upload, nrows=15_001)
            if len(df) > 15_000:
                st.error("CSV is limited to 15,000 records per run to keep density clustering responsive.")
                st.stop()
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        features = st.sidebar.multiselect("Numeric analysis features", numeric, default=numeric[:min(8, len(numeric))])
        if st.sidebar.checkbox("Include ground truth for evaluation"):
            truth_col = st.sidebar.selectbox("Ground-truth column", [None, *df.columns.tolist()])

if df is None:
    st.markdown('<div class="hero"><div class="eyebrow">EXPLORATORY CLINICAL ANALYTICS</div><h1>Find the signal<br>inside the noise.</h1><p>Discover interpretable patient cohorts with a transparent, density-aware workflow. Start with synthetic data or bring a de-identified CSV.</p></div>', unsafe_allow_html=True)
    st.info("Choose a data source in the left panel to begin.")
    st.stop()

if source == "Upload CSV" and not features:
    st.warning("Select at least one numeric analysis feature to continue.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("BASELINES")
kmeans_k = st.sidebar.slider("K-Means cohorts", 2, 10, 5)
db_eps = st.sidebar.slider("DBSCAN radius", 0.1, 3.0, 0.9, 0.05)
db_min = st.sidebar.slider("DBSCAN min samples", 2, 30, 10)
with st.sidebar.expander("PROPOSED · Nested density + centroid refinement"):
    eps1 = st.slider("Coarse radius", 0.1, 3.0, 0.6, 0.05)
    min1 = st.slider("Coarse min samples", 2, 30, 12)
    eps2 = st.slider("Fine radius", 0.05, 2.0, 0.25, 0.05)
    min2 = st.slider("Fine min samples", 2, 30, 10)
    pca_n = st.slider("Density projection dimensions", 2, min(max(len(features), 2), 6), 2)
    reject_std = st.slider("Noise rejection tolerance", 1.0, 4.0, 2.5, 0.1)
run = st.sidebar.button("Run cohort analysis", type="primary", use_container_width=True)

st.markdown('<div class="hero"><div class="eyebrow">EXPLORATORY CLINICAL ANALYTICS · RESEARCH PROTOTYPE</div><h1>Find the signal<br>inside the noise.</h1><p>Compare density discovery with centroid-based baselines. Inspect each cohort in clinical units, understand the method’s limits, and export a reproducible analysis.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Dataset snapshot</div>', unsafe_allow_html=True)
a, b, c = st.columns(3)
a.metric("Records", f"{len(df):,}")
b.metric("Selected features", str(len(features)))
c.metric("Missing feature values", f"{int(df[features].isna().sum().sum()):,}" if features else "—")
st.dataframe(df.head(8), use_container_width=True, hide_index=True)
st.caption("Use de-identified data only. This research tool does not provide diagnoses or treatment recommendations.")

with st.expander("Method notes and interpretation"):
    st.markdown("""
**Proposed method: nested density discovery with centroid refinement.** Numeric features are median-imputed and standardized. PCA is used only for the two DBSCAN density passes. The surviving density groups seed K-Means in the full standardized feature space; a distance threshold then flags distant assignments as noise. Feature profiles are reported in original units.

This is an experimental unsupervised workflow, not a validated clinical algorithm. PCA can hide structure; DBSCAN is sensitive to parameters; centroid refinement can change density-discovered memberships. Compare baselines, inspect profiles, and validate externally before drawing conclusions. Synthetic demo data is illustrative only.
""")

if not run:
    st.info("Tune the parameters, then choose **Run cohort analysis**.")
    st.stop()

try:
    with st.spinner("Preparing features and comparing clustering methods…"):
        out = run_pipeline(df, features, truth_col, kmeans_k, db_eps, db_min,
                           eps1, min1, eps2, min2, pca_n, reject_std)
except (ValueError, KeyError, FloatingPointError) as exc:
    st.error(f"Analysis could not run: {exc}")
    st.stop()

st.markdown('<div class="section-kicker">02 / Model comparison</div>', unsafe_allow_html=True)
st.dataframe(out.metrics_df, use_container_width=True, hide_index=True)
st.caption("Internal scores exclude points labeled noise. Silhouette is higher-is-better; Davies–Bouldin is lower-is-better. Metrics are descriptive, not clinical validation.")

st.markdown('<div class="section-kicker">03 / Cohort landscape</div>', unsafe_allow_html=True)
fig, axes = plt.subplots(1, len(out.labels) + (out.true_labels is not None), figsize=(5.2 * (len(out.labels) + (out.true_labels is not None)), 4.5), squeeze=False)
panels = []
if out.true_labels is not None:
    panels.append(("Ground truth", out.true_labels))
panels.extend(out.labels.items())
for ax, (name, labels) in zip(axes[0], panels):
    labs = np.asarray(labels)
    unique = sorted(np.unique(labs).tolist())
    cmap = colormaps["turbo"].resampled(max(1, len(unique)))
    for i, lab in enumerate(unique):
        mask = labs == lab
        ax.scatter(out.X_2d[mask, 0], out.X_2d[mask, 1], s=11, alpha=.72,
                   c="#9aa8b8" if lab == -1 else [cmap(i)], marker="x" if lab == -1 else "o",
                   label="Noise" if lab == -1 else f"Cohort {lab}", linewidths=.6)
    ax.set_title(name, color="#e9f1ff", fontsize=10, fontweight="bold")
    ax.set_xlabel("Projection 1", color="#9aabc0"); ax.set_ylabel("Projection 2", color="#9aabc0")
    ax.set_facecolor("#0b1728"); ax.tick_params(colors="#8497ae", labelsize=8)
    for edge in ax.spines.values(): edge.set_color("#24354a")
    ax.legend(fontsize=7, loc="best", frameon=False, labelcolor="#c2d0e0")
fig.patch.set_facecolor("#0b1422"); fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown('<div class="section-kicker">04 / Clinical profile · proposed method</div>', unsafe_allow_html=True)
st.dataframe(out.cohort_profile_df, use_container_width=True)
if not out.cohort_profile_df.empty:
    vals = out.cohort_profile_df[features]
    z = (vals - vals.mean()) / vals.std(ddof=0).replace(0, 1)
    fig2, ax2 = plt.subplots(figsize=(min(1.1 * len(features) + 2, 15), max(2.4, .55 * len(z) + 1.2)))
    im = ax2.imshow(z.fillna(0).values, cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5)
    ax2.set_xticks(range(len(features)), labels=features, rotation=40, ha="right", color="#c2d0e0")
    ax2.set_yticks(range(len(z)), labels=["Noise" if x == -1 else f"Cohort {x}" for x in z.index], color="#c2d0e0")
    ax2.set_facecolor("#0b1728"); fig2.patch.set_facecolor("#0b1422")
    fig2.colorbar(im, ax=ax2, label="Standardized profile", shrink=.8)
    fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

st.markdown('<div class="section-kicker">05 / Export</div>', unsafe_allow_html=True)
result = df.copy()
for name, labels in out.labels.items():
    result[f"cluster__{safe_name(name)}"] = labels
csv_data = csv_safe(result).to_csv(index=False).encode("utf-8")
metrics_data = out.metrics_df.to_csv(index=False).encode("utf-8")
profiles_data = out.cohort_profile_df.to_csv(index=True).encode("utf-8")
x, y, zcol = st.columns(3)
x.download_button("Download assignments", csv_data, "cohort_assignments.csv", "text/csv", use_container_width=True)
y.download_button("Download comparison", metrics_data, "model_comparison.csv", "text/csv", use_container_width=True)
zcol.download_button("Download profiles", profiles_data, "cohort_profiles.csv", "text/csv", use_container_width=True)
st.markdown('<p class="fineprint">No patient data is persisted by this app code. Hosting providers, reverse proxies, and uploaded input remain subject to their own logging and retention settings.</p>', unsafe_allow_html=True)
