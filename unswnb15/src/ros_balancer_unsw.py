import os
import sys
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def get_device():
    """Selects Intel Arc (xpu), CUDA, or CPU fallback."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def balance_dataset_block(
    target_block: str, 
    project_folder: str, 
    target_count: int = 100000
) -> None:
    device = get_device()
    clean_raw_path = os.path.join(project_folder, f"{target_block}_raw_clean.csv")
    balanced_out_path = os.path.join(project_folder, f"{target_block}_ros_balanced.csv")
    
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Master scaler or feature column map not found! Run baseline preprocessing first.")
    
    global_scaler = joblib.load(scaler_path)
    global_feature_columns = joblib.load(features_path)

    if os.path.exists(balanced_out_path):
        print(f"[SKIP] Target block '{target_block}' is already ROS balanced at: {balanced_out_path}")
        return

    print("=" * 80)
    print(f"RUNNING RANDOM OVERSAMPLING (ROS) ENGINE FOR TARGET BLOCK: {target_block.upper()}")
    print(f"Hardware Device: {device} ({torch.xpu.get_device_name(0) if device.type == 'xpu' else 'CPU/CUDA'})")
    print(f"Reading Clean Source: {clean_raw_path}")
    print("=" * 80)

    df_block = pd.read_csv(clean_raw_path)
    class_counts = df_block['Label'].value_counts()
    
    print("\nInitial Class Distribution Matrix:")
    print(class_counts)

    # Transform features into [0,1] space using saved global baseline scaler
    X_block_raw = df_block[global_feature_columns]
    y_block_raw = df_block['Label'].values

    X_block_scaled = global_scaler.transform(X_block_raw)
    X_block_scaled = np.clip(X_block_scaled, 0.0, 1.0)
    X_block_scaled = pd.DataFrame(X_block_scaled, columns=global_feature_columns)
    
    df_working = X_block_scaled.copy()
    df_working['Label'] = y_block_raw

    balanced_class_dfs = []

    for label, count in class_counts.items():
        print(f"\nProcessing class footprint for: [{label}]")
        df_class = df_working[df_working['Label'] == label]

        # Scenario A: Majority class -> Downsampling
        if count >= target_count:
            print(f"-> Class [{label}] meets target limit ({count} rows). Downsampling to {target_count}...")
            balanced_class_dfs.append(df_class.sample(n=target_count, random_state=42))

        # Scenario B: Minority class -> ROS Duplication
        else:
            needed_rows = target_count - count
            print(f"-> Class [{label}] is starved ({count} rows). Duplicating {needed_rows} rows via ROS...")
            
            # Sample only the missing count to preserve 100% of original minority rows
            df_oversampled = df_class.sample(n=needed_rows, replace=True, random_state=42)
            df_balanced_class = pd.concat([df_class, df_oversampled], axis=0, ignore_index=True)
            balanced_class_dfs.append(df_balanced_class)

    df_final_balanced_scaled = pd.concat(balanced_class_dfs, axis=0, ignore_index=True)
    X_balanced_scaled = df_final_balanced_scaled[global_feature_columns]
    y_balanced_final = df_final_balanced_scaled['Label'].values

    print("\nInverse scaling balanced distributions back to raw feature scales...")
    X_balanced_raw = pd.DataFrame(global_scaler.inverse_transform(X_balanced_scaled), columns=global_feature_columns)
    df_final_balanced = X_balanced_raw.copy()
    df_final_balanced['Label'] = y_balanced_final

    # Round categoricals back to integers (Specific to UNSW-NB15)
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        if col in df_final_balanced.columns:
            df_final_balanced[col] = df_final_balanced[col].round().astype(int)

    # Ensure consistent column ordering
    output_columns = global_feature_columns + ['Label']
    df_final_balanced = df_final_balanced[output_columns]

    df_final_balanced.to_csv(balanced_out_path, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS! ROS Balanced file committed to: {balanced_out_path}")
    print("=" * 80)
    print(f"Verified Class Distribution Matrix for {target_block.upper()}:")
    print(df_final_balanced['Label'].value_counts())


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    if len(sys.argv) > 1:
        target_block = sys.argv[1].lower()
        balance_dataset_block(target_block=target_block, project_folder=data_dir)
    else:
        BLOCKS_TO_BALANCE = [
            "block_1", "block_2", "block_3", "block_4", "block_5", 
            "block_6", "block_7", "block_8", "block_9", "block_10"
        ]
        for block in BLOCKS_TO_BALANCE:
            balance_dataset_block(target_block=block, project_folder=data_dir)