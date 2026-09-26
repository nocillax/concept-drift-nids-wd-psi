import os
import sys
import joblib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm

LATENT_DIM = 32
LAMBDA_GP = 10.0


def get_device():
    """Selects Intel Arc (xpu), CUDA, or CPU fallback."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WGAN_Generator(nn.Module):
    def __init__(self, input_dim=LATENT_DIM, output_dim=41):
        super(WGAN_Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, output_dim),
            nn.Sigmoid()  # Outputs bounded in [0, 1] matching MinMaxScaler range
        )

    def forward(self, z):
        return self.model(z)


class WGAN_Critic(nn.Module):
    def __init__(self, input_dim=41):
        super(WGAN_Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 1)  # Unconstrained score output
        )

    def forward(self, x):
        return self.model(x)


def calculate_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates 1-Lipschitz continuity penalty for WGAN-GP training."""
    alpha = torch.rand((real_samples.size(0), 1), device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = critic(interpolates)
    fake_grad_outputs = torch.ones(d_interpolates.size(), device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean() * LAMBDA_GP


def balance_dataset_block(
    target_block: str, 
    project_folder: str, 
    target_count: int = 100000, 
    epochs: int = 500,
    n_critic: int = 5
) -> None:
    """
    Balances a specific dataset block (downsamples majority classes to target_count
    and trains class-wise WGAN-GPs to generate synthetic samples for minority classes).
    """
    device = get_device()
    clean_raw_path = os.path.join(project_folder, f"{target_block}_raw_clean.csv")
    balanced_out_path = os.path.join(project_folder, f"{target_block}_balanced.csv")
    
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Master scaler or feature column map not found! Run baseline preprocessing first.")
    
    global_scaler = joblib.load(scaler_path)
    global_feature_columns = joblib.load(features_path)
    num_features = len(global_feature_columns)

    if os.path.exists(balanced_out_path):
        print(f"[SKIP] Target block '{target_block}' is already balanced at: {balanced_out_path}")
        return

    print("=" * 80)
    print(f"RUNNING GENERATIVE BALANCING ENGINE FOR TARGET BLOCK: {target_block.upper()}")
    print(f"Hardware Device: {device} ({torch.xpu.get_device_name(0) if device.type == 'xpu' else 'CPU/CUDA'})")
    print(f"Reading Clean Source: {clean_raw_path}")
    print("=" * 80)

    df_block = pd.read_csv(clean_raw_path)
    X_block_raw = df_block[global_feature_columns]
    y_block_raw = df_block['Label'].values

    class_counts = df_block['Label'].value_counts()
    print("\nInitial Class Distribution Matrix:")
    print(class_counts)

    # Transform features into [0,1] space using saved global baseline scaler
    # FIX: Clip the real data so the Critic does not easily spot unbounded outliers
    X_block_scaled = global_scaler.transform(X_block_raw)
    X_block_scaled = np.clip(X_block_scaled, 0.0, 1.0)
    X_block_scaled = pd.DataFrame(X_block_scaled, columns=global_feature_columns)
    
    df_working = X_block_scaled.copy()
    df_working['Label'] = y_block_raw

    balanced_class_dfs = []

    for label, count in class_counts.items():
        print(f"\nProcessing class footprint for: [{label}]")
        df_class = df_working[df_working['Label'] == label]

        # Scenario A: Over-abundant class -> Strict downsampling
        if count >= target_count:
            print(f"-> Class [{label}] meets target limit ({count} rows). Downsampling to {target_count}...")
            balanced_class_dfs.append(df_class.sample(n=target_count, random_state=42))

        # Scenario B: Starved minority class -> Train WGAN-GP synthetic generator
        else:
            needed_rows = target_count - count
            print(f"-> Class [{label}] is starved ({count} rows). Training WGAN-GP layer for {needed_rows} rows...")

            X_minority = df_class.drop(columns=['Label']).values
            minority_tensor = torch.tensor(X_minority, dtype=torch.float32).to(device)

            batch_size = 64 if len(X_minority) > 64 else len(X_minority)
            loader = DataLoader(
                TensorDataset(minority_tensor), 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=(len(X_minority) > 64)
            )

            gen_net = WGAN_Generator(input_dim=LATENT_DIM, output_dim=num_features).to(device)
            crit_net = WGAN_Critic(input_dim=num_features).to(device)

            optimizer_G = optim.Adam(gen_net.parameters(), lr=0.0001, betas=(0.0, 0.9))
            optimizer_C = optim.Adam(crit_net.parameters(), lr=0.0001, betas=(0.0, 0.9))

            gen_net.train()
            crit_net.train()

            epoch_pbar = tqdm(range(epochs), desc=f"Optimizing {str(label)[:15]} Landscape")

            for epoch in epoch_pbar:
                for i, (real_samples,) in enumerate(loader):
                    real_samples = real_samples.to(device)

                    # === Train Critic Matrix ===
                    optimizer_C.zero_grad()
                    noise = torch.randn(real_samples.size(0), LATENT_DIM, device=device)
                    fake_samples = gen_net(noise).detach()

                    loss_C = -torch.mean(crit_net(real_samples)) + torch.mean(crit_net(fake_samples))
                    gp = calculate_gradient_penalty(crit_net, real_samples, fake_samples, device)
                    (loss_C + gp).backward()
                    optimizer_C.step()

                    # === Train Generator Matrix every n_critic batches ===
                    if i % n_critic == 0:
                        optimizer_G.zero_grad()
                        gen_samples = gen_net(torch.randn(real_samples.size(0), LATENT_DIM, device=device))
                        loss_G = -torch.mean(crit_net(gen_samples))
                        loss_G.backward()
                        optimizer_G.step()

            # Mint synthetic feature arrays
            gen_net.eval()
            with torch.no_grad():
                noise = torch.randn(needed_rows, LATENT_DIM, device=device)
                generated_features = gen_net(noise).cpu().numpy()

            # Enforce strict [0.0, 1.0] MinMaxScaler bounds
            generated_features = np.clip(generated_features, 0.0, 1.0)

            df_synthetic = pd.DataFrame(generated_features, columns=global_feature_columns)
            df_synthetic['Label'] = label

            df_balanced_class = pd.concat([df_class, df_synthetic], axis=0, ignore_index=True)
            balanced_class_dfs.append(df_balanced_class)

    # Inverse scale back to raw metric feature bounds
    df_block_balanced_scaled = pd.concat(balanced_class_dfs, axis=0, ignore_index=True)
    X_balanced_scaled = df_block_balanced_scaled[global_feature_columns]
    y_balanced_final = df_block_balanced_scaled['Label'].values

    print("\nInverse scaling balanced distributions back to raw feature scales...")
    X_balanced_raw = pd.DataFrame(global_scaler.inverse_transform(X_balanced_scaled), columns=global_feature_columns)
    df_final_balanced = X_balanced_raw.copy()
    df_final_balanced['Label'] = y_balanced_final

    # FIX: Round categoricals back to integers
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        if col in df_final_balanced.columns:
            df_final_balanced[col] = df_final_balanced[col].round().astype(int)

    df_final_balanced.to_csv(balanced_out_path, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS! Balanced file saved to: {balanced_out_path}")
    print("=" * 80)
    print(f"Verified Class Distribution Matrix for {target_block.upper()}:")
    print(df_final_balanced['Label'].value_counts())


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    # Allows passing a specific block via terminal (e.g. python -m unswnb15.src.wgan_balancer_unsw block_1)    if len(sys.argv) > 1:
        target_block = sys.argv[1].lower()
        balance_dataset_block(target_block=target_block, project_folder=data_dir)
    else:
        BLOCKS_TO_BALANCE = [
            "block_1", "block_2", "block_3", "block_4", "block_5", 
            "block_6", "block_7", "block_8", "block_9", "block_10"
        ]
        for block in BLOCKS_TO_BALANCE:
            balance_dataset_block(target_block=block, project_folder=data_dir)