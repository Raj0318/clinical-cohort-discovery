"""
data_generator.py
------------------
Generates a synthetic (but clinically realistic) patient dataset with
several underlying cohorts (ground-truth clusters) plus injected noise
and outlier points, to simulate messy real-world EHR data.

Features generated (per patient):
    Age, BMI, Systolic_BP, Diastolic_BP, Glucose, Cholesterol,
    Heart_Rate, Creatinine

Ground-truth cohorts:
    0 - Healthy
    1 - Pre-Diabetic
    2 - Hypertensive
    3 - Diabetic + Hypertensive (high risk)
    4 - Elderly / Renal-risk

A configurable fraction of points is replaced with uniform random noise
scattered across the feature space (simulating data-entry errors,
mixed/rare conditions, or measurement noise) and is NOT part of any
ground-truth cohort (label = -1).
"""

import numpy as np
import pandas as pd

FEATURES = ["Age", "BMI", "Systolic_BP", "Diastolic_BP",
            "Glucose", "Cholesterol", "Heart_Rate", "Creatinine"]

COHORT_SPECS = [
    dict(name="Healthy",
         mean=[35, 22, 115, 75, 90, 180, 72, 0.9],
         std=[8, 2, 8, 6, 8, 15, 6, 0.1]),
    dict(name="Pre-Diabetic",
         mean=[48, 27, 128, 82, 118, 205, 78, 1.0],
         std=[9, 2.5, 10, 7, 10, 18, 7, 0.12]),
    dict(name="Hypertensive",
         mean=[55, 29, 152, 96, 100, 210, 84, 1.1],
         std=[10, 3, 12, 8, 12, 20, 8, 0.15]),
    dict(name="Diabetic_Hypertensive",
         mean=[60, 33, 160, 100, 175, 245, 90, 1.3],
         std=[9, 3.5, 14, 9, 20, 25, 9, 0.18]),
    dict(name="Elderly_RenalRisk",
         mean=[70, 26, 140, 88, 110, 200, 76, 1.8],
         std=[7, 2.5, 11, 7, 14, 18, 7, 0.25]),
]

CLIP_BOUNDS = {
    "Age": (18, 95), "BMI": (14, 55), "Systolic_BP": (80, 220),
    "Diastolic_BP": (50, 140), "Glucose": (60, 350),
    "Cholesterol": (120, 350), "Heart_Rate": (45, 140),
    "Creatinine": (0.4, 4.0),
}


def generate_clinical_dataset(
    n_samples: int = 1200,
    noise_fraction: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    n_noise = int(n_samples * noise_fraction)
    n_clustered = n_samples - n_noise
    n_cohorts = len(COHORT_SPECS)
    base_per_cohort = n_clustered // n_cohorts
    remainder = n_clustered - base_per_cohort * n_cohorts

    rows, labels, cohort_names = [], [], []

    for i, spec in enumerate(COHORT_SPECS):
        n_this = base_per_cohort + (1 if i < remainder else 0)
        mean = np.array(spec["mean"])
        std = np.array(spec["std"])
        samples = rng.normal(loc=mean, scale=std, size=(n_this, len(mean)))
        rows.append(samples)
        labels.extend([i] * n_this)
        cohort_names.extend([spec["name"]] * n_this)

    all_means = np.array([s["mean"] for s in COHORT_SPECS])
    lo = all_means.min(axis=0) * 0.7
    hi = all_means.max(axis=0) * 1.3
    noise_samples = rng.uniform(low=lo, high=hi, size=(n_noise, len(FEATURES)))
    rows.append(noise_samples)
    labels.extend([-1] * n_noise)
    cohort_names.extend(["Noise/Outlier"] * n_noise)

    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=FEATURES)
    for col, (lo_c, hi_c) in CLIP_BOUNDS.items():
        df[col] = df[col].clip(lo_c, hi_c)

    df["true_cohort_id"] = labels
    df["true_cohort_name"] = cohort_names

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df.insert(0, "patient_id", [f"P{idx:05d}" for idx in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate_clinical_dataset()
    df.to_csv("../data/synthetic_patients.csv", index=False)
    print(df["true_cohort_name"].value_counts())
