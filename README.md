# Cohort Studio

**Noise-aware clinical cohort discovery — an exploratory research prototype.**

Cohort Studio compares K-Means and DBSCAN baselines with a proposed two-pass density discovery and centroid-refinement heuristic. It includes a synthetic demonstration dataset, cohort summaries in original feature units, projection plots, and downloadable analysis outputs.

> **Clinical and privacy boundary:** This software is not a medical device, does not produce diagnoses or treatment recommendations, and has not been clinically validated. Use synthetic or properly de-identified data. Do not upload direct identifiers or protected health information to an unapproved deployment. Hosting, reverse-proxy, and browser infrastructure may retain logs outside this application.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Start with synthetic data to explore the workflow. For CSVs, select at least two varying numeric columns. Missing numeric values are median-imputed for analysis; source rows are retained in the assignment export.

## What the proposed method does

1. Median-impute missing and non-finite feature values, remove constant selected columns, and standardize the remaining numeric features.
2. Project standardized features with PCA for the density passes.
3. Run a coarse DBSCAN pass, then a finer DBSCAN pass on its non-noise survivors.
4. Use the fine density groups to seed K-Means in the full standardized feature space.
5. Flag assignments beyond a cohort-derived distance threshold as noise.

The final labels can differ from the original density memberships because the centroid refinement reassigns points. PCA dimensionality, DBSCAN radii, scale, and cohort shape all affect results. The method is a heuristic, and its output needs independent validation before research or operational use. Internal metrics summarize geometric separation; they do not establish clinical validity or causality.

## Data handling and security

- CSV uploads are limited to 25 MB and 15,000 records to bound interactive density-clustering costs. The app processes data in the current Streamlit session and does not write uploads to disk. Silhouette is sampled for larger datasets.
- The app code does not persist patient data. This is not an end-to-end privacy guarantee: Streamlit hosting, proxies, observability, backups, and user devices have their own retention and access controls.
- Do not commit secrets, raw patient exports, or `.streamlit/secrets.toml`. Use the deployment provider's secret store for credentials.
- CSV downloads are for analysis. Treat exported files as sensitive and store/share them according to your organization's policy.
- For a real deployment, add identity-aware access control, TLS, audit logging, retention controls, dependency scanning, and an approved data-processing environment before using sensitive data.

## Container deployment

Build and run the non-root container:

```bash
docker build -t cohort-studio .
docker run --rm -p 8501:8501 cohort-studio
```

Set `STREAMLIT_SERVER_ADDRESS=0.0.0.0` and `STREAMLIT_SERVER_PORT=8501` when deploying behind a managed TLS reverse proxy. Keep the application private until authentication and hosting controls meet the needs of your environment. The included container health check verifies the Streamlit health endpoint; it is not a clinical service-level guarantee.

## Project layout

```text
app.py                  Streamlit interface
src/clustering.py       Proposed clustering estimator
src/pipeline.py         Validation, preprocessing, metrics, profiles
src/data_generator.py   Synthetic cohort data generator
.streamlit/config.toml   Streamlit security and upload defaults
Dockerfile              Non-root container baseline
```

## Resume and portfolio framing

**Problem:** density methods can flag outliers but may become hard to tune on correlated clinical features; centroid methods assign every point and can absorb distant records.

**Approach:** compare density-based cohort seeding with full-feature centroid refinement and an explicit noise flag, while keeping cohort summaries interpretable in source units.

**Engineering decisions:** input validation, missing-value handling, deterministic seeds, bounded upload size, no app-level upload persistence, a non-root container, and explicit research limitations.

**Next validation steps:** benchmark on documented public datasets, report parameter sensitivity and stability across seeds, compare against domain-reviewed labels, add privacy and threat-model review, and obtain clinical/regulatory review before any consequential use.
