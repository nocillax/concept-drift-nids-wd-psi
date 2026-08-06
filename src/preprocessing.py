import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASELINE_DAYS = ["feb14", "feb15", "feb16"]

def setup_baseline_preprocessing(project_folder: str):
    """
    Loads Feb 14-16 in memory, extracts non-zero variance feature list,
    and fits global MinMaxScaler without saving redundant combined CSVs.
    """
    print("Initializing Baseline Preprocessing Layer...")

    baseline_dfs = []
    for day in BASELINE_DAYS:
        file_path = os.path.join(project_folder, f"{day}_raw_clean.csv")
        if os.path.exists(file_path):
            print(f"-> Loading records from: {file_path}")
            baseline_dfs.append(pd.read_csv(file_path))

    df_baseline_raw = pd.concat(baseline_dfs, axis=0, ignore_index=True)

    X_baseline_raw = df_baseline_raw.drop(columns=['Label'])
    selector = X_baseline_raw.var() != 0
    global_feature_columns = X_baseline_raw.loc[:, selector].columns.tolist()

    global_scaler = MinMaxScaler()
    global_scaler.fit(X_baseline_raw[global_feature_columns])

    # Save fitted transformers to Drive
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")
    
    joblib.dump(global_scaler, scaler_path)
    joblib.dump(global_feature_columns, features_path)

    print(f"Saved: {scaler_path}")
    print(f"Saved: {features_path}")
    print(f"Retained global features count: {len(global_feature_columns)}")

    return global_scaler, global_feature_columns