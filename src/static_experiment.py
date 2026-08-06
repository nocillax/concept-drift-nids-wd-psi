import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import torch.amp as amp

# Optional: Intel scikit-learn acceleration (Turbo mode for RF)
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel scikit-learn extension activated.")
except ImportError:
    pass

# =====================================================================
# STEP 1: ALIGNED STATIC CONTROL GROUP DATA TENSORS
# =====================================================================

def prepare_static_data(project_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print(f"STEP 1: COMPILING STATIC CONTROL GROUP TENSORS")
    print(f"Hardware Target: {device}")
    print("="*80)

    # Reconstruct Global Scaling Bounds
    baseline_raw_files = [
        os.path.join(project_folder, "feb14_raw_clean.csv"),
        os.path.join(project_folder, "feb15_raw_clean.csv"),
        os.path.join(project_folder, "feb16_raw_clean.csv"),
    ]
    
    df_raw_pool = pd.concat([pd.read_csv(f) for f in baseline_raw_files if os.path.exists(f)], axis=0, ignore_index=True)
    X_raw_base = df_raw_pool.drop(columns=['Label'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0.0)
    selector = X_raw_base.var() != 0
    global_feature_cols = X_raw_base.loc[:, selector].columns.tolist()

    static_scaler = MinMaxScaler()
    static_scaler.fit(X_raw_base[global_feature_cols])
    
    # Load Master Training Balanced Pool & Map to Binary Space
    baseline_balanced_files = [
        os.path.join(project_folder, "feb14_balanced.csv"),
        os.path.join(project_folder, "feb15_balanced.csv"),
        os.path.join(project_folder, "feb16_balanced.csv")
    ]
    df_train = pd.concat([pd.read_csv(f) for f in baseline_balanced_files if os.path.exists(f)], axis=0, ignore_index=True)
    X_train_raw = df_train[global_feature_cols]
    y_train_static = np.where(df_train['Label'] == 'Benign', 0, 1).astype(np.int32)
    
    # Optimization: Cast to float32 to prevent hidden memory copying during training
    X_train_static = static_scaler.transform(X_train_raw).astype(np.float32)

    # Ingest Future Deployment Test Days
    future_test_files = {
        "Feb_20": os.path.join(project_folder, "feb20_raw_clean.csv"),
        "Feb_21": os.path.join(project_folder, "feb21_raw_clean.csv"),
        "Feb_22": os.path.join(project_folder, "feb22_raw_clean.csv"),
        "Feb_23": os.path.join(project_folder, "feb23_raw_clean.csv"),
        "Feb_28": os.path.join(project_folder, "feb28_raw_clean.csv"),
        "Mar01": os.path.join(project_folder, "mar01_raw_clean.csv"),
        "Mar02": os.path.join(project_folder, "mar02_raw_clean.csv")
    }

    test_days_tensors = {}
    for day_name, file_path in future_test_files.items():
        if os.path.exists(file_path):
            df_day = pd.read_csv(file_path)
            X_day_raw = df_day[global_feature_cols]
            y_day_binary = np.where(df_day['Label'] == 'Benign', 0, 1).astype(np.int32)
            X_day_scaled = static_scaler.transform(X_day_raw).astype(np.float32)
            test_days_tensors[day_name] = (X_day_scaled, y_day_binary)

    return X_train_static, y_train_static, test_days_tensors, global_feature_cols, device

# =====================================================================
# STEP 2: RANDOM FOREST CONTROL ENSEMBLE
# =====================================================================

def train_rf_static(X_train_static, y_train_static, test_days_tensors, static_metrics_master):
    print("\n" + "="*80)
    print("STEP 2: TRAINING PILLAR 1 - RANDOM FOREST CONTROL ENSEMBLE")
    print("="*80)

    # Optimization: max_depth=25 speeds up training and limits overfitting
    rf_model = RandomForestClassifier(n_estimators=0, max_depth=25, warm_start=True, random_state=42, n_jobs=-1)

    total_trees = 100
    step_size = 10
    chunks = total_trees // step_size

    for i in tqdm(range(chunks), desc="Growing Random Forest Trees"):
        rf_model.n_estimators += step_size
        rf_model.fit(X_train_static, y_train_static)

    print("\nDeploying Random Forest against future timelines...")
    for day_name, (X_test, y_test) in test_days_tensors.items():
        preds = rf_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        static_metrics_master.append({
            "Model": "RandomForest_Static", "Deployment_Day": day_name,
            "Accuracy": acc, "Precision": prec, "Recall_DetectionRate": rec, "F1_Score": f1, "False_Positive_Rate": fpr
        })

# =====================================================================
# DEEP LEARNING ARCHITECTURES
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
        
        # FIX: Dynamically compute the flatten size using a dummy tensor
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
# STEP 3: DEEP LEARNING EVALUATION PIPELINE
# =====================================================================

def train_dl_model(model, model_name, train_loader, test_days_tensors, static_metrics_master, device):
    print("\n" + "="*80)
    print(f"TRAINING PILLAR: {model_name}")
    print("="*80)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/5", unit="batch", leave=True)

        for batch_X, batch_y in batch_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Optimization: Mixed Precision Training (autocast)
            with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            batch_bar.set_postfix(batch_loss=f"{current_loss:.4f}")

    # FIX: Optimized Batched Evaluation to prevent OOM
    model.eval()
    print(f"\nDeploying {model_name} against future timelines...")
    with torch.no_grad():
        for day_name, (X_test, y_test) in test_days_tensors.items():
            t_X_test = torch.tensor(X_test, dtype=torch.float32) # Kept on CPU initially
            all_preds = []
            eval_batch_size = 4096 
            
            for i in range(0, len(t_X_test), eval_batch_size):
                batch_ev = t_X_test[i : i + eval_batch_size].to(device)
                
                # Optimization: Mixed precision inference
                with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(batch_ev)
                    
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend((probs >= 0.5).astype(int).flatten())
                
            preds = np.array(all_preds)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            static_metrics_master.append({
                "Model": model_name, "Deployment_Day": day_name,
                "Accuracy": acc, "Precision": prec, "Recall_DetectionRate": rec, "F1_Score": f1, "False_Positive_Rate": fpr
            })

# =====================================================================
# STEP 4: ORCHESTRATION AND EXPORT
# =====================================================================

def run_static_experiment(project_folder):
    static_metrics_master = []
    
    X_train_static, y_train_static, test_days_tensors, global_feature_cols, device = prepare_static_data(project_folder)

    # Train Random Forest
    train_rf_static(X_train_static, y_train_static, test_days_tensors, static_metrics_master)

    # Prepare PyTorch DataLoader with optimizations (larger batch, pinning memory)
    t_X_train = torch.tensor(X_train_static, dtype=torch.float32)
    t_y_train = torch.tensor(y_train_static, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(t_X_train, t_y_train), 
        batch_size=2048, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    # Initialize PyTorch Models
    input_dim = len(global_feature_cols)
    
    mlp_model = StaticMLP(input_dim).to(device)
    cnn_model = Static1DCNN(input_dim).to(device)
    lstm_model = StaticLSTM(input_dim).to(device)

    # FIX: Fail-safe PyTorch 2.0 Compiler
    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            mlp_model = torch.compile(mlp_model)
            cnn_model = torch.compile(cnn_model)
            lstm_model = torch.compile(lstm_model)
        except Exception as e:
            print(f"\n[Notice] torch.compile skipped due to environment incompatibility: {e}\n")

    # Train Deep Learning Models
    train_dl_model(mlp_model, "MLP_Static", train_loader, test_days_tensors, static_metrics_master, device)
    train_dl_model(cnn_model, "1D-CNN_Static", train_loader, test_days_tensors, static_metrics_master, device)
    train_dl_model(lstm_model, "LSTM_Static", train_loader, test_days_tensors, static_metrics_master, device)

    # Export Metrics
    print("\n" + "="*80)
    print("STEP 4: METRICS CONSOLIDATION & PERSISTENT DRIVE EXPORT")
    print("="*80)
    
    output_results_path = os.path.join(project_folder, "static_performance_results.csv")
    df_static_performance = pd.DataFrame(static_metrics_master)
    df_static_performance.to_csv(output_results_path, index=False)
    
    print(f"\nSUCCESS! Static Group Master Performance File written to:\n{output_results_path}")
    print("\nAGGREGATED PERFORMANCE MATRIX OVERVIEW (STATIC BENCHMARK CONTROL GROUP):")
    print(df_static_performance.to_markdown(index=False))
