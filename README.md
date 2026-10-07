# Noise-Robust Clinical Patient Cohort Discovery (Streamlit App)
### Nested DBSCAN + Centroid-Initialized K-Means

An interactive web app that discovers patient cohorts from clinical
vitals/labs while staying robust to noisy or erroneous records.

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
This opens the app in your browser (usually `http://localhost:8501`).

## 2. Project Structure
```
.
├── app.py                 # Streamlit UI — entry point
├── src/
│   ├── data_generator.py  # synthetic clinical dataset (5 cohorts + noise)
│   ├── clustering.py      # NestedDBSCANCentroidKMeans (core algorithm)
│   └── pipeline.py        # runs baselines + proposed method, computes metrics
├── data/                  # (gitignored) place your own CSVs here, or use the built-in generator
├── outputs/                # (gitignored contents) for anything you export manually
├── requirements.txt
└── .gitignore
```

## 3. Using the App
- **Sidebar → Data source**: use the built-in synthetic demo data, or
  upload your own patient CSV (select which numeric columns are the
  clustering features, and optionally a ground-truth cohort column if
  you have one, purely for evaluation).
- **Sidebar → Algorithm parameters**: tune Plain K-Means (`k`), Plain
  DBSCAN (`eps`, `min_samples`), and the Proposed method's two DBSCAN
  stages + noise-rejection strictness.
- Click **🚀 Run clustering**.
- The app shows: a metrics comparison table, PCA-2D cluster plots for
  every method side by side, a clinical cohort-profile table +
  heatmap for the proposed method's discovered cohorts, and CSV
  download buttons for all of the above.

## 4. The Algorithm — What Is What
**Problem it solves:** K-Means forces every point into a cluster (one
bad outlier can drag a centroid off target). Plain DBSCAN correctly
flags noise, but breaks down in high-dimensional clinical feature
space (8+ correlated features flatten density contrast — "the curse
of dimensionality").

**Pipeline (`src/clustering.py`):**
1. **Standardize** features (`StandardScaler`).
2. **PCA (2 components by default)** — used ONLY to give the DBSCAN
   stages a space where "density" is geometrically meaningful again.
   Centroids and the final K-Means step still use ALL original
   features, so results stay in real clinical units.
3. **Stage 1 — coarse DBSCAN** (wide `eps1`): strips gross outliers
   from the whole population.
4. **Stage 2 — fine DBSCAN** (tight `eps2`), run only on Stage-1
   survivors: discovers the fine-grained patient cohorts. Running two
   DBSCAN passes nested like this ("Nested DBSCAN") stops a handful of
   noisy points from bridging two real clusters into one.
5. **Centroid computation**: for each Stage-2 cluster, its mean is
   computed in the full feature space — these become the initial
   centroids for K-Means (instead of random/`k-means++` init).
6. **Centroid K-Means**: run over the ENTIRE dataset (including
   Stage-1 noise) — this "rescues" borderline noisy points into their
   correct cohort while giving stable, repeatable cluster centers.
7. **Noise-robust rejection**: any point still farther than
   `mean + noise_std * std` from its assigned centroid (measured
   against that cohort's original clean-point spread) is kept flagged
   as noise (`-1`) rather than forced into a cohort.

## 5. Evaluation Metrics Shown
- **Internal** (always available): Silhouette Score, Davies-Bouldin
  Index, Calinski-Harabasz Index.
- **External** (only if you provide/use a ground-truth column):
  Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).

## 6. Tuning Tips (if clusters look wrong on your own data)
- If the Proposed method finds **0 or 1 cohort**: `eps2` is too large
  for your data's scale — lower it (try halving it) and/or increase
  `pca_components`.
- If it finds **too many tiny fragments**: raise `eps2` slightly or
  lower `min_samples2`.
- If **too many points are flagged noise**: raise the noise-rejection
  strictness slider, or `eps1`.
- A good starting point: generate a k-distance plot for your scaled
  data (`k = min_samples`), and set `eps` near the "elbow" of the
  sorted k-distance curve. This is standard DBSCAN tuning practice.
