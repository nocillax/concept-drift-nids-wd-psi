import os
import gc
import time
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import wasserstein_distance

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

def get_device():
    """Selects Intel Arc (xpu), CUDA, or CPU fallback."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# =====================================================================
# 1. CORE CONFIGURATIONS
# =====================================================================
def run_adaptive_experiment_unsw_no_reservoir(project_folder):
    device = get_device()
    output_adaptive_path = os.path.join(project_folder, "adaptive_no_reservoir_results_unsw.csv")

    print("="*80)
    print("PHASE 5: ADAPTIVE SLIDING WINDOW (NO RESERVOIR ABLATION) - UNSW-NB15")
    print(f"Device: {device} ({torch.xpu.get_device_name(0) if device.type == 'xpu' else 'CPU/CUDA'})")
    print("="*80)

    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Scaler or feature columns missing. Run preprocessing once to generate .pkl files.")

    global_scaler = joblib.load(scaler_path)
    global_feature_columns = joblib.load(features_path)

    # =====================================================================
    # 2. HELPER FUNCTIONS
    # =====================================================================
    def compute_window_psi(expected, actual, num_bins=10):
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(expected, percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        expected_pcts = expected_counts / len(expected)
        actual_pcts = actual_counts / len(actual)
        eps = 1e-4
        expected_pcts = np.where(expected_pcts == 0, eps, expected_pcts)
        actual_pcts = np.where(actual_pcts == 0, eps, actual_pcts)
        return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))

    def print_day_header(day_name, day_num, total_days):
        print("\n" + "="*80)
        print(f"BLOCK {day_num}/{total_days}: {day_name.upper()}")
        print("="*80)

    def print_training_summary(window_rows, total_rows):
        print(f"  Sliding Window : {window_rows:,} rows")
        print(f"  History Anchor : 0 rows (Ablation Mode: Disabled)")
        print(f"  Total Training : {total_rows:,} rows")

    def print_model_result(model_name, acc, rec, f1, fpr, elapsed):
        print(f"    {model_name:12s} | Acc={acc:.4f} | DR={rec:.4f} | F1={f1:.4f} | FPR={fpr:.4f} | {elapsed:.1f}s")

    def print_day_summary(day_name, results):
        print(f"\n  --- {day_name} Summary ---")
        print(f"    {'Model':<12s} | {'Accuracy':>8s} | {'Det.Rate':>8s} | {'F1':>8s} | {'FPR':>8s}")
        print(f"    {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for r in results:
            print(f"    {r['model']:<12s} | {r['acc']:>8.4f} | {r['rec']:>8.4f} | {r['f1']:>8.4f} | {r['fpr']:>8.4f}")

    # =====================================================================
    # 2.5 MODEL ARCHITECTURES
    # =====================================================================
    class StaticMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.network(x)

    class Static1DCNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, input_dim)
                dummy_output = self.conv(dummy_input)
                computed_flatten_size = dummy_output.shape[1] * dummy_output.shape[2]

            self.fc = nn.Sequential(
                nn.Linear(computed_flatten_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class StaticLSTM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        def forward(self, x):
            x = x.unsqueeze(1)
            lstm_out, _ = self.lstm(x)
            final_timestep = lstm_out[:, -1, :]
            return self.fc(final_timestep)

    # =====================================================================
    # 3. TIMELINE SCHEDULE (UNSW-NB15 MAPPED)
    # =====================================================================
    timeline_schedule = {
        "Block_4": {
            "raw_test_file": os.path.join(project_folder, "block_4_raw_clean.csv"),
            "training_window_blocks": ["block_1_balanced.csv", "block_2_balanced.csv", "block_3_balanced.csv"],
            "drift_history_raw": ["block_1_raw_clean.csv", "block_2_raw_clean.csv", "block_3_raw_clean.csv"]
        },
        "Block_5": {
            "raw_test_file": os.path.join(project_folder, "block_5_raw_clean.csv"),
            "training_window_blocks": ["block_2_balanced.csv", "block_3_balanced.csv", "block_4_balanced.csv"],
            "drift_history_raw": ["block_2_raw_clean.csv", "block_3_raw_clean.csv", "block_4_raw_clean.csv"]
        },
        "Block_6": {
            "raw_test_file": os.path.join(project_folder, "block_6_raw_clean.csv"),
            "training_window_blocks": ["block_3_balanced.csv", "block_4_balanced.csv", "block_5_balanced.csv"],
            "drift_history_raw": ["block_3_raw_clean.csv", "block_4_raw_clean.csv", "block_5_raw_clean.csv"]
        },
        "Block_7": {
            "raw_test_file": os.path.join(project_folder, "block_7_raw_clean.csv"),
            "training_window_blocks": ["block_4_balanced.csv", "block_5_balanced.csv", "block_6_balanced.csv"],
            "drift_history_raw": ["block_4_raw_clean.csv", "block_5_raw_clean.csv", "block_6_raw_clean.csv"]
        },
        "Block_8": {
            "raw_test_file": os.path.join(project_folder, "block_8_raw_clean.csv"),
            "training_window_blocks": ["block_5_balanced.csv", "block_6_balanced.csv", "block_7_balanced.csv"],
            "drift_history_raw": ["block_5_raw_clean.csv", "block_6_raw_clean.csv", "block_7_raw_clean.csv"]
        },
        "Block_9": {
            "raw_test_file": os.path.join(project_folder, "block_9_raw_clean.csv"),
            "training_window_blocks": ["block_6_balanced.csv", "block_7_balanced.csv", "block_8_balanced.csv"],
            "drift_history_raw": ["block_6_raw_clean.csv", "block_7_raw_clean.csv", "block_8_raw_clean.csv"]
        },
        "Block_10": {
            "raw_test_file": os.path.join(project_folder, "block_10_raw_clean.csv"),
            "training_window_blocks": ["block_7_balanced.csv", "block_8_balanced.csv", "block_9_balanced.csv"],
            "drift_history_raw": ["block_7_raw_clean.csv", "block_8_raw_clean.csv", "block_9_raw_clean.csv"]
        }
    }

    adaptive_metrics_master = []
    total_days = len(timeline_schedule)

    # =====================================================================
    # 4. MAIN CHRONOLOGICAL LOOP
    # =====================================================================
    for day_idx, (day_name, stage) in enumerate(timeline_schedule.items(), 1):
        day_start = time.time()
        print_day_header(day_name, day_idx, total_days)

        # --- STEP A: ACTIVE DRIFT (Raw Sliding Window vs Raw Test) ---
        print("  [Drift] Computing PSI & WD (raw window vs raw test)...")
        t0 = time.time()

        hist_raw_dfs = []
        for f in stage["drift_history_raw"]:
            fp = os.path.join(project_folder, f)
            if os.path.exists(fp):
                hist_raw_dfs.append(pd.read_csv(fp, low_memory=False))

        df_hist_raw = pd.concat(hist_raw_dfs, axis=0, ignore_index=True)
        df_target_raw = pd.read_csv(stage["raw_test_file"], low_memory=False)

        X_hist_scaled = pd.DataFrame(global_scaler.transform(df_hist_raw[global_feature_columns]), columns=global_feature_columns)
        X_target_scaled = pd.DataFrame(global_scaler.transform(df_target_raw[global_feature_columns]), columns=global_feature_columns)

        psi_list, wd_list = [], []
        for col in global_feature_columns:
            psi_list.append(compute_window_psi(X_hist_scaled[col].values, X_target_scaled[col].values))
            wd_list.append(float(wasserstein_distance(X_hist_scaled[col].values, X_target_scaled[col].values)))

        mean_active_psi = np.mean(psi_list)
        mean_active_wd = np.mean(wd_list)
        print(f"  [Drift] PSI={mean_active_psi:.4f} | WD={mean_active_wd:.4f} ({time.time()-t0:.1f}s)")

        del df_hist_raw, df_target_raw, X_hist_scaled, X_target_scaled, hist_raw_dfs
        gc.collect()

        # --- STEP B: ASSEMBLE TRAINING POOL (Sliding Window Only) ---
        print("  [Train] Assembling sliding window (no history anchor)...")
        t0 = time.time()

        pool_dfs = []
        for block in stage["training_window_blocks"]:
            pool_dfs.append(pd.read_csv(os.path.join(project_folder, block), low_memory=False))

        df_assembled_pool = pd.concat(pool_dfs, axis=0, ignore_index=True)

        active_X_train = np.clip(global_scaler.transform(df_assembled_pool[global_feature_columns]), 0.0, 1.0).astype(np.float32)
        active_y_train = np.where(df_assembled_pool['Label'] == 'Benign', 0, 1)

        window_rows = len(df_assembled_pool)
        total_train_rows = len(df_assembled_pool)

        print_training_summary(window_rows, total_train_rows)
        print(f"  [Train] Assembly done ({time.time()-t0:.1f}s)")

        del df_assembled_pool, pool_dfs
        gc.collect()

        # --- STEP C: PREPARE EVALUATION DATA ---
        df_eval_raw = pd.read_csv(stage["raw_test_file"], low_memory=False)
        X_eval_scaled = np.clip(global_scaler.transform(df_eval_raw[global_feature_columns]), 0.0, 1.0).astype(np.float32)
        y_eval_binary = np.where(df_eval_raw['Label'] == 'Benign', 0, 1)
        print(f"  [Eval]  Test set: {len(y_eval_binary):,} rows")

        # --- STEP D: SETUP PYTORCH LOADER ---
        t_X_train = torch.tensor(active_X_train, dtype=torch.float32)
        t_y_train = torch.tensor(active_y_train, dtype=torch.float32).unsqueeze(1)
        
        active_loader = DataLoader(
            TensorDataset(t_X_train, t_y_train), 
            batch_size=2048, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=False
        )
        criterion = nn.BCEWithLogitsLoss()

        # --- STEP E: MODEL TRAINING & EVALUATION ---
        print("  [Models] Training & evaluating...")
        day_results = []
        models_to_run = ["RandomForest", "MLP", "1D-CNN", "LSTM"]

        for model_name in models_to_run:
            t0 = time.time()
            
            if model_name == "RandomForest":
                current_rf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)
                current_rf.fit(active_X_train, active_y_train)
                preds = current_rf.predict(X_eval_scaled)
                
            else:
                if model_name == "MLP":
                    model = StaticMLP(input_dim=len(global_feature_columns)).to(device)
                elif model_name == "1D-CNN":
                    model = Static1DCNN(input_dim=len(global_feature_columns)).to(device)
                elif model_name == "LSTM":
                    model = StaticLSTM(input_dim=len(global_feature_columns)).to(device)
                
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                model.train()
                
                for epoch in range(5):
                    for bx, by in active_loader:
                        bx, by = bx.to(device), by.to(device)
                        opt.zero_grad()
                        with torch.autocast(device_type=device.type, enabled=(device.type in ["cuda", "xpu"])):
                            loss = criterion(model(bx), by)
                        loss.backward()
                        opt.step()
                
                model.eval()
                with torch.no_grad():
                    t_ev = torch.tensor(X_eval_scaled, dtype=torch.float32)
                    all_preds = []
                    eval_batch_size = 4096 
                    
                    for i in range(0, len(t_ev), eval_batch_size):
                        batch_ev = t_ev[i : i + eval_batch_size].to(device)
                        with torch.autocast(device_type=device.type, enabled=(device.type in ["cuda", "xpu"])):
                            probs = torch.sigmoid(model(batch_ev)).cpu().numpy()
                        all_preds.extend((probs >= 0.5).astype(int).flatten())
                        
                    preds = np.array(all_preds)

            acc = accuracy_score(y_eval_binary, preds)
            prec = precision_score(y_eval_binary, preds, zero_division=0)
            rec = recall_score(y_eval_binary, preds, zero_division=0)
            f1 = f1_score(y_eval_binary, preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_eval_binary, preds, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            elapsed = time.time() - t0

            print_model_result(model_name, acc, rec, f1, fpr, elapsed)

            day_results.append({'model': model_name, 'acc': acc, 'rec': rec, 'f1': f1, 'fpr': fpr})
            adaptive_metrics_master.append({
                "Model": f"{model_name}_Adaptive_NoReservoir", "Deployment_Day": day_name,
                "Accuracy": acc, "Precision": prec, "Recall_DetectionRate": rec,
                "F1_Score": f1, "False_Positive_Rate": fpr,
                "Active_Window_PSI": mean_active_psi, "Active_Window_WD": mean_active_wd
            })

        print_day_summary(day_name, day_results)

        # Cleanup
        del df_eval_raw, X_eval_scaled, y_eval_binary
        del active_X_train, active_y_train, t_X_train, t_y_train, active_loader
        gc.collect()

        print(f"  [Done] Block completed in {time.time()-day_start:.1f}s")

    # =====================================================================
    # 5. EXPORT RESULTS
    # =====================================================================
    df_adaptive_performance = pd.DataFrame(adaptive_metrics_master)
    df_adaptive_performance.to_csv(output_adaptive_path, index=False)

    print("\n" + "="*80)
    print("ALL BLOCKS COMPLETE (NO RESERVOIR ABLATION)")
    print(f"Results saved to: {output_adaptive_path}")
    print("="*80)

    print("\nFINAL RESULTS SUMMARY")
    print(f"{'Block':<10s} | {'Model':<26s} | {'Accuracy':>8s} | {'Det.Rate':>8s} | {'F1':>8s} | {'FPR':>8s}")
    print("-" * 85)
    for _, row in df_adaptive_performance.iterrows():
        print(f"{row['Deployment_Day']:<10s} | {row['Model']:<26s} | {row['Accuracy']:>8.4f} | {row['Recall_DetectionRate']:>8.4f} | {row['F1_Score']:>8.4f} | {row['False_Positive_Rate']:>8.4f}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    run_adaptive_experiment_unsw_no_reservoir(project_folder=data_dir)