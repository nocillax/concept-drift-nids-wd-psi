import os
import gc
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm

DATASET_S3_FILES = {
    "feb14": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb15": "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb16": "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb20": "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb21": "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb22": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb23": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "feb28": "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "mar01": "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "mar02": "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
}

S3_BUCKET_BASE = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms"

LABEL_STANDARDIZATION_MAP = {
    'Benign': 'Benign',
    'FTP-BruteForce': 'FTP-BruteForce',
    'SSH-Bruteforce': 'SSH-Bruteforce',
    'DoS attacks-GoldenEye': 'DoS attacks-GoldenEye',
    'DoS attacks-Slowloris': 'DoS attacks-Slowloris',
    'DoS attacks-Hulk': 'DoS attacks-Hulk',
    'DoS attacks-SlowHTTPTest': 'DoS attacks-SlowHTTPTest',
    'DDoS attacks-LOIC-HTTP': 'DDoS attacks-LOIC-HTTP',
    'DDOS attack-HOIC': 'DDoS attacks-HOIC',
    'DDOS attack-LOIC-UDP': 'DDoS attacks-LOIC-UDP',
    'Brute Force -Web': 'Brute Force-Web',
    'Brute Force -XSS': 'Brute Force-XSS',
    'SQL Injection': 'SQL-Injection',
    'Infilteration': 'Infiltration',
    'Infiltration': 'Infiltration',
    'Bot': 'Bot'
}

def fetch_and_clean_day(
    day_key: str, 
    filename: str, 
    output_dir: str, 
    target_rows: int = 200000, 
    chunksize: int = 100000, 
    sample_frac: float = 0.30
) -> None:
    raw_output_path = os.path.join(output_dir, f"{day_key}_raw.csv")
    clean_output_path = os.path.join(output_dir, f"{day_key}_raw_clean.csv")
    temp_local_file = f"/content/temp_{day_key}.csv"

    # Step A: Download & Sample Raw Data if not present
    if not os.path.exists(raw_output_path):
        print(f"\nFetching timeline segment from S3: {day_key.upper()}")
        s3_url = f"{S3_BUCKET_BASE}/{filename}"
        cmd = f'aws s3 cp "{s3_url}" "{temp_local_file}" --no-sign-request'
        subprocess.run(cmd, shell=True, check=True)

        chunk_list = []
        with tqdm(unit="chunk") as pbar:
            for chunk in pd.read_csv(temp_local_file, chunksize=chunksize, low_memory=False):
                sampled_chunk = chunk.sample(frac=sample_frac, random_state=42)
                chunk_list.append(sampled_chunk)
                pbar.update(1)

        df_raw = pd.concat(chunk_list, axis=0)
        df_raw.columns = df_raw.columns.str.strip()
        df_raw = df_raw.replace([np.inf, -np.inf], np.nan).dropna()

        df_final = df_raw.sample(n=target_rows, random_state=42, replace=(len(df_raw) < target_rows))
        df_final.to_csv(raw_output_path, index=False)

        if os.path.exists(temp_local_file):
            os.remove(temp_local_file)
        del df_raw, df_final, chunk_list
        gc.collect()

    # Step B: Sanitize Threat Labels & Structure
    if not os.path.exists(clean_output_path):
        print(f"Sanitizing {day_key.upper()} dataset labels...")
        df = pd.read_csv(raw_output_path, low_memory=False)

        df = df[df['Label'].astype(str).str.strip() != 'Label']
        if 'Timestamp' in df.columns:
            df = df.drop(columns=['Timestamp'])

        df['Label'] = df['Label'].astype(str).str.strip().map(LABEL_STANDARDIZATION_MAP)
        df = df.dropna(subset=['Label'])

        feature_cols = [c for c in df.columns if c != 'Label']
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df.to_csv(clean_output_path, index=False)
        print(f"-> Saved clean asset: {clean_output_path}")


def run_full_data_pipeline(project_folder: str) -> None:
    subprocess.run("pip install awscli --quiet", shell=True)
    for day_key, file_name in DATASET_S3_FILES.items():
        frac = 0.15 if day_key == "feb20" else 0.30
        fetch_and_clean_day(
            day_key=day_key,
            filename=file_name,
            output_dir=project_folder,
            target_rows=200000,
            sample_frac=frac
        )