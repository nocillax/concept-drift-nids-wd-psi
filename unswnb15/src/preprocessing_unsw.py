import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# using the first 3 sequential blocks as the institutional baseline
BASELINE_BLOCKS = ["block_1", "block_2", "block_3"]

def setup_baseline_preprocessing_unsw(project_folder: str):
    print("Initializing UNSW-NB15 Baseline Preprocessing Layer...")

    baseline_dfs = []
    for block in BASELINE_BLOCKS:
        file_path = os.path.join(project_folder, f"{block}_raw_clean.csv")
        if os.path.exists(file_path):
            print(f"-> Loading records from: {file_path}")
            baseline_dfs.append(pd.read_csv(file_path))
        else:
            print(f"Error: {file_path} not found. Run the data loader first.")

    if not baseline_dfs:
        raise FileNotFoundError("No baseline blocks found. Cannot fit scaler.")

    df_baseline_raw = pd.concat(baseline_dfs, axis=0, ignore_index=True)

    # isolate features
    X_baseline_raw = df_baseline_raw.drop(columns=['Label'])
    
    # Invariant Variance Filtering: drop columns that have zero variance in the baseline
    selector = X_baseline_raw.var() != 0
    global_feature_columns = X_baseline_raw.loc[:, selector].columns.tolist()

    print(f"Features after zero-variance filtering: {len(global_feature_columns)}")

    # lock the global scaler limits based ONLY on these first 3 blocks
    global_scaler = MinMaxScaler()
    global_scaler.fit(X_baseline_raw[global_feature_columns])

    # Save fitted transformers to the unsw project folder
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")
    
    joblib.dump(global_scaler, scaler_path)
    joblib.dump(global_feature_columns, features_path)

    print(f"Saved: {scaler_path}")
    print(f"Saved: {features_path}")

    return global_scaler, global_feature_columns


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    print(f"Executing UNSW-NB15 Preprocessing pointing to: {data_dir}")
    setup_baseline_preprocessing_unsw(project_folder=data_dir)