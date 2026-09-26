import os
import joblib
from pathlib import Path
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


def run_static_drift_analysis_unsw(project_folder: str) -> pd.DataFrame:
    """
    Executes benchmark static drift analysis comparing baseline (Blocks 1-3)
    against all future timeline blocks (Blocks 4 through 10).
    """
    output_csv_path = os.path.join(project_folder, "static_drift_metrics_unsw.csv")
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Master scaler or feature list missing! Run baseline preprocessing first.")

    global_scaler = joblib.load(scaler_path)
    global_feature_columns = joblib.load(features_path)

    print("=" * 80)
    print("RUNNING BENCHMARK STATIC DRIFT ANALYSIS (WD & PSI) FOR UNSW-NB15")
    print("=" * 80)

    # Load and combine raw baseline blocks (1, 2, 3) in memory
    baseline_blocks = ["block_1", "block_2", "block_3"]
    baseline_dfs = []
    for block in baseline_blocks:
        file_p = os.path.join(project_folder, f"{block}_raw_clean.csv")
        if os.path.exists(file_p):
            baseline_dfs.append(pd.read_csv(file_p))

    df_base_raw = pd.concat(baseline_dfs, axis=0, ignore_index=True)
    X_base_raw = df_base_raw[global_feature_columns]

    # Scale baseline into [0, 1] range using fitted global baseline scaler
    df_base_scaled = pd.DataFrame(global_scaler.transform(X_base_raw), columns=global_feature_columns)

    future_blocks = {
        "Block_4": "block_4_raw_clean.csv",
        "Block_5": "block_5_raw_clean.csv",
        "Block_6": "block_6_raw_clean.csv",
        "Block_7": "block_7_raw_clean.csv",
        "Block_8": "block_8_raw_clean.csv",
        "Block_9": "block_9_raw_clean.csv",
        "Block_10": "block_10_raw_clean.csv"
    }

    all_drift_records = []

    for block_label, filename in future_blocks.items():
        file_path = os.path.join(project_folder, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Asset for {block_label} missing at {file_path}. Skipping.")
            continue

        df_block = pd.read_csv(file_path)
        X_block_raw = df_block[global_feature_columns]

        # Transform future block using baseline scaling bounds
        X_block_scaled = pd.DataFrame(global_scaler.transform(X_block_raw), columns=global_feature_columns)

        # Compute metrics across all features
        df_metrics = compute_feature_drift(df_base_scaled, X_block_scaled, global_feature_columns)

        mean_psi = df_metrics["PSI"].mean()
        mean_wd = df_metrics["Wasserstein_Distance"].mean()
        print(f"-> {block_label} Drift Metrics: Mean PSI = {mean_psi:.4f} | Mean WD = {mean_wd:.4f}")

        # ONLY append the aggregated means for the block
        all_drift_records.append({
            "Block": block_label,
            "Mean_PSI": mean_psi,
            "Mean_WD": mean_wd
        })

    # Create the final dataframe from just the block summaries
    df_final_drift = pd.DataFrame(all_drift_records)
    df_final_drift.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS! Static drift metrics saved to: {output_csv_path}")
    print("=" * 80)

    return df_final_drift


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    df_drift = run_static_drift_analysis_unsw(project_folder=data_dir)
    print("\nCalculated Static Drift Metrics Summary (UNSW-NB15):")
    print(df_drift.to_string(index=False))