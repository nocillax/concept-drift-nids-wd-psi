import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm

LATENT_DIM = 32
LAMBDA_GP = 10.0


class WGAN_Generator(nn.Module):
    def __init__(self, input_dim=LATENT_DIM, output_dim=68):
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
    def __init__(self, input_dim=68):
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


def balance_dataset_day(
    target_day: str, 
    project_folder: str, 
    target_count: int = 100000, 
    epochs: int = 500,
    n_critic: int = 5
) -> None:
    """
    Balances a specific dataset day (downsamples majority classes to target_count
    and trains class-wise WGAN-GPs to generate synthetic samples for minority classes).
    
    Args:
        target_day: Target day key (e.g., 'feb14', 'feb28', mar02).
        project_folder: Path to persistent Google Drive folder.
        target_count: Target uniform sample limit per class (default: 100k).
        epochs: WGAN-GP training iterations (default: 500).
        n_critic: Number of critic updates per generator update (default: 5).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_raw_path = os.path.join(project_folder, f"{target_day}_raw_clean.csv")
    balanced_out_path = os.path.join(project_folder, f"{target_day}_balanced.csv")
    
    scaler_path = os.path.join(project_folder, "global_scaler.pkl")
    features_path = os.path.join(project_folder, "global_feature_columns.pkl")

    # Load fitted global baseline scaler and feature column order
    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Master scaler or feature column map not found! Run baseline preprocessing first.")
    
    global_scaler = joblib.load(scaler_path)
    global_feature_columns = joblib.load(features_path)
    num_features = len(global_feature_columns)

    if os.path.exists(balanced_out_path):
        print(f"[SKIP] Target day '{target_day}' is already balanced at: {balanced_out_path}")
        return

    print("=" * 80)
    print(f"RUNNING GENERATIVE BALANCING ENGINE FOR TARGET DAY: {target_day.upper()}")
    print(f"Hardware Device: {device}")
    print(f"Reading Clean Source: {clean_raw_path}")
    print("=" * 80)

    df_day = pd.read_csv(clean_raw_path)
    X_day_raw = df_day[global_feature_columns]
    y_day_raw = df_day['Label'].values

    class_counts = df_day['Label'].value_counts()
    print("\nInitial Class Distribution Matrix:")
    print(class_counts)

    # Transform features into [0,1] space using saved global baseline scaler
    X_day_scaled = pd.DataFrame(global_scaler.transform(X_day_raw), columns=global_feature_columns)
    df_working = X_day_scaled.copy()
    df_working['Label'] = y_day_raw

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

            epoch_pbar = tqdm(range(epochs), desc=f"Optimizing {label[:15]} Landscape")

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

                    # === Train Generator Matrix every n_critic batches (Corrected Batch Index Check) ===
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
    df_day_balanced_scaled = pd.concat(balanced_class_dfs, axis=0, ignore_index=True)
    X_balanced_scaled = df_day_balanced_scaled[global_feature_columns]
    y_balanced_final = df_day_balanced_scaled['Label'].values

    print("\nInverse scaling balanced distributions back to raw feature scales...")
    X_balanced_raw = pd.DataFrame(global_scaler.inverse_transform(X_balanced_scaled), columns=global_feature_columns)
    df_final_balanced = X_balanced_raw.copy()
    df_final_balanced['Label'] = y_balanced_final

    df_final_balanced.to_csv(balanced_out_path, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS! Balanced file committed to Drive: {balanced_out_path}")
    print("=" * 80)
    print(f"Verified Class Distribution Matrix for {target_day.upper()}:")
    print(df_final_balanced['Label'].value_counts())