import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between reference and target distributions.
    Expands boundary endpoints to +/- infinity to capture extreme drift.
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    # Expand boundaries to infinity to handle out-of-bounds values in future drift days
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pcts = expected_counts / len(expected)
    actual_pcts = actual_counts / len(actual)

    # Laplace smoothing to protect against log(0) and zero-division
    eps = 1e-4
    expected_pcts = np.where(expected_pcts == 0, eps, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, eps, actual_pcts)

    psi_val = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi_val)


def compute_feature_drift(
    base_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    feature_cols: list, 
    max_samples: int = 100000
) -> pd.DataFrame:
    """
    Calculates PSI and Wasserstein Distance per feature between base and target DataFrames.
    Subsamples up to max_samples per feature for rapid Wasserstein sorting.
    """
    records = []

    for col in feature_cols:
        base_dist = base_df[col].values
        target_dist = target_df[col].values

        # Downsample for fast sorting performance if array size exceeds threshold
        if len(base_dist) > max_samples:
            base_dist = np.random.choice(base_dist, max_samples, replace=False)
        if len(target_dist) > max_samples:
            target_dist = np.random.choice(target_dist, max_samples, replace=False)

        psi = calculate_psi(base_dist, target_dist)
        wd = float(wasserstein_distance(base_dist, target_dist))

        records.append({
            "Feature": col,
            "PSI": psi,
            "Wasserstein_Distance": wd
        })

    return pd.DataFrame(records)


def run_static_drift_analysis(project_folder: str) -> pd.DataFrame:
    """
    Executes benchmark static drift analysis comparing baseline (Feb 14-16)
    against all future timeline days (Feb 20 through Mar 02).
    """
    output_csv_path = os.path.join(project_folder, "static_drift_metrics.csv")
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Master scaler or feature list missing! Run baseline preprocessing first.")

    global_scaler = joblib.load(scaler_path)
    global_feature_columns = joblib.load(features_path)

    print("=" * 80)
    print("RUNNING BENCHMARK STATIC DRIFT ANALYSIS (WD & PSI)")
    print("=" * 80)

    # Load and combine raw baseline files (Feb 14, 15, 16) in memory
    baseline_days = ["feb14", "feb15", "feb16"]
    baseline_dfs = []
    for day in baseline_days:
        file_p = os.path.join(project_folder, f"{day}_raw_clean.csv")
        if os.path.exists(file_p):
            baseline_dfs.append(pd.read_csv(file_p))

    df_base_raw = pd.concat(baseline_dfs, axis=0, ignore_index=True)
    X_base_raw = df_base_raw[global_feature_columns]

    # Scale baseline into [0, 1] range using fitted global baseline scaler
    df_base_scaled = pd.DataFrame(global_scaler.transform(X_base_raw), columns=global_feature_columns)

    future_days = {
        "Feb_20": "feb20_raw_clean.csv",
        "Feb_21": "feb21_raw_clean.csv",
        "Feb_22": "feb22_raw_clean.csv",
        "Feb_23": "feb23_raw_clean.csv",
        "Feb_28": "feb28_raw_clean.csv",
        "Mar_01": "mar01_raw_clean.csv",
        "Mar_02": "mar02_raw_clean.csv"
    }

    all_drift_records = []

    for day_label, filename in future_days.items():
        file_path = os.path.join(project_folder, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Asset for {day_label} missing at {file_path}. Skipping.")
            continue

        df_day = pd.read_csv(file_path)
        X_day_raw = df_day[global_feature_columns]

        # Transform future day using baseline scaling bounds
        X_day_scaled = pd.DataFrame(global_scaler.transform(X_day_raw), columns=global_feature_columns)

        # Compute metrics across all features
        df_metrics = compute_feature_drift(df_base_scaled, X_day_scaled, global_feature_columns)

        mean_psi = df_metrics["PSI"].mean()
        mean_wd = df_metrics["Wasserstein_Distance"].mean()
        print(f"-> {day_label} Drift Metrics: Mean PSI = {mean_psi:.4f} | Mean WD = {mean_wd:.4f}")

        # ONLY append the aggregated means for the day, ignore feature-wise rows
        all_drift_records.append({
            "Day": day_label,
            "Mean_PSI": mean_psi,
            "Mean_WD": mean_wd
        })

    # Create the final dataframe from just the daily summaries
    df_final_drift = pd.DataFrame(all_drift_records)
    df_final_drift.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS! Static drift metrics committed to Drive: {output_csv_path}")
    print("=" * 80)

    return df_final_drift