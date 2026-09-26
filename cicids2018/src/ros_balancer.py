import os
import sys
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

def balance_dataset_day(
    target_day: str, 
    project_folder: str, 
    target_count: int = 100000
) -> None:
    clean_raw_path = os.path.join(project_folder, f"{target_day}_raw_clean.csv")
    balanced_out_path = os.path.join(project_folder, f"{target_day}_ros_balanced.csv")
    
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Master scaler or feature column map not found! Run baseline preprocessing first.")
    
    global_feature_columns = joblib.load(features_path)

    if os.path.exists(balanced_out_path):
        print(f"[SKIP] Target day '{target_day}' is already ROS balanced at: {balanced_out_path}")
        return

    print("=" * 80)
    print(f"RUNNING RANDOM OVERSAMPLING (ROS) ENGINE FOR TARGET DAY: {target_day.upper()}")
    print(f"Reading Clean Source: {clean_raw_path}")
    print("=" * 80)

    df_day = pd.read_csv(clean_raw_path)
    class_counts = df_day['Label'].value_counts()
    
    print("\nInitial Class Distribution Matrix:")
    print(class_counts)

    balanced_class_dfs = []

    for label, count in class_counts.items():
        print(f"\nProcessing class footprint for: [{label}]")
        df_class = df_day[df_day['Label'] == label]

        # Scenario A: Majority class -> Downsampling (matches WGAN exactly)
        if count >= target_count:
            print(f"-> Class [{label}] meets target limit ({count} rows). Downsampling to {target_count}...")
            balanced_class_dfs.append(df_class.sample(n=target_count, random_state=42))

        # Scenario B: Minority class -> ROS Duplication (keeps all real rows + appends shortfall)
        else:
            needed_rows = target_count - count
            print(f"-> Class [{label}] is starved ({count} rows). Duplicating {needed_rows} rows via ROS...")
            
            # Sample only the missing count to preserve 100% of original minority rows
            df_oversampled = df_class.sample(n=needed_rows, replace=True, random_state=42)
            df_balanced_class = pd.concat([df_class, df_oversampled], axis=0, ignore_index=True)
            balanced_class_dfs.append(df_balanced_class)

    df_final_balanced = pd.concat(balanced_class_dfs, axis=0, ignore_index=True)

    # Ensure consistent column ordering
    output_columns = global_feature_columns + ['Label']
    df_final_balanced = df_final_balanced[output_columns]

    df_final_balanced.to_csv(balanced_out_path, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS! ROS Balanced file committed to: {balanced_out_path}")
    print("=" * 80)
    print(f"Verified Class Distribution Matrix for {target_day.upper()}:")
    print(df_final_balanced['Label'].value_counts())


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    if len(sys.argv) > 1:
        target_day = sys.argv[1].lower()
        balance_dataset_day(target_day=target_day, project_folder=data_dir)
    else:
        DAYS_TO_BALANCE = [
            "feb14", "feb15", "feb16", "feb20", "feb21", 
            "feb22", "feb23", "feb28", "mar01", "mar02"
        ]
        for day in DAYS_TO_BALANCE:
            balance_dataset_day(target_day=day, project_folder=data_dir)