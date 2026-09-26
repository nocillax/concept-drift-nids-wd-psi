import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path

# columns we do not need for NIDS training (like IPs and timestamps)
DROP_COLUMNS = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime']

# Standard 49 features for raw UNSW-NB15 CSVs
UNSW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth', 'res_bdy_len', 
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 
    'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
]

def process_and_chunk_unsw(project_folder: str, block_size: int = 200000, num_blocks: int = 7) -> None:
    raw_dir = os.path.join(project_folder, "raw_csvs")
    
    if not os.path.exists(raw_dir):
        print(f"Directory not found: {raw_dir}")
        print("Please create a 'raw_csvs' folder inside your data folder and place the 4 raw UNSW-NB15 CSV files there.")
        return

    raw_files = [os.path.join(raw_dir, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]
    
    print("Loading all raw UNSW-NB15 files into memory...")
    df_list = []
    for file in raw_files:
        if os.path.exists(file):
            df_list.append(pd.read_csv(file, low_memory=False, header=None))
        else:
            print(f"Missing file: {file}")
            
    if not df_list:
        print("No raw data to process. Exiting.")
        return

    df_full = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list
    gc.collect()

    # FIX 1: Assign the proper headers to the dataframe
    if len(df_full.columns) == len(UNSW_COLUMNS):
        df_full.columns = UNSW_COLUMNS
    else:
        print(f"Warning: Column count mismatch! Expected {len(UNSW_COLUMNS)}, got {len(df_full.columns)}")
        return

    # Just in case the user downloaded a version where the 1st row accidentally has headers
    if isinstance(df_full.iloc[0, 0], str) and 'srcip' in df_full.iloc[0].values:
        df_full = df_full[1:].reset_index(drop=True)

    print("Sorting continuous stream by timestamp (Stime)...")
    df_full['Stime'] = pd.to_numeric(df_full['Stime'], errors='coerce')
    df_full = df_full.dropna(subset=['Stime'])
    df_full = df_full.sort_values(by='Stime').reset_index(drop=True)

    print("Standardizing labels and categoricals...")
    
    # FIX 2: Drop the binary 'Label' FIRST to avoid naming collisions
    if 'Label' in df_full.columns:
         df_full = df_full.drop(columns=['Label']) 

    # Now standardize 'attack_cat' and rename it to 'Label' safely
    if 'attack_cat' in df_full.columns:
        df_full['attack_cat'] = df_full['attack_cat'].fillna('Benign')
        df_full['attack_cat'] = df_full['attack_cat'].astype(str).str.strip()
        df_full['attack_cat'] = df_full['attack_cat'].replace({'Normal': 'Benign'})
        df_full = df_full.rename(columns={'attack_cat': 'Label'})

    # handle categorical string features (proto, service, state) by converting them to numeric codes
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        if col in df_full.columns:
            df_full[col] = pd.factorize(df_full[col])[0]

    # drop IP addresses and timestamps
    cols_to_drop = [c for c in DROP_COLUMNS if c in df_full.columns]
    df_full = df_full.drop(columns=cols_to_drop)

    # ensure everything else is numeric
    feature_cols = [c for c in df_full.columns if c != 'Label']
    for col in feature_cols:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')

    df_full[feature_cols] = df_full[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"Dataset sorted and cleaned. Total rows available: {len(df_full):,}")
    print(f"Slicing stream into {num_blocks} blocks of {block_size:,} rows...")

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        
        if start_idx >= len(df_full):
            print(f"Ran out of data at block {i+1}.")
            break
            
        df_block = df_full.iloc[start_idx:end_idx]
        
        block_name = f"block_{i+1}"
        out_path = os.path.join(project_folder, f"{block_name}_raw_clean.csv")
        
        df_block.to_csv(out_path, index=False)
        print(f"-> Saved {block_name}: {out_path} ({len(df_block):,} rows)")

    del df_full
    gc.collect()
    print("Pipeline complete.")

def run_unsw_pipeline(project_folder: str) -> None:
    process_and_chunk_unsw(project_folder)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = str(PROJECT_ROOT / "data")
    
    print(f"Executing UNSW-NB15 Pipeline pointing to: {data_dir}")
    run_unsw_pipeline(project_folder=data_dir)