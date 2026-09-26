# Concept Drift Monitoring and Continual Adaptation in Streaming Network Intrusion Detection

This repository contains the code for the paper "Concept Drift Monitoring and Continual Adaptation in Streaming Network Intrusion Detection". The project evaluates continual network intrusion detection models against non-stationary data streams using chronological, prequential constraints on the CIC-IDS2018 and UNSW-NB15 datasets.

The pipeline implements raw data fetching, feature scaling, minority class balancing (via Random Oversampling and WGAN-GP), and label-free drift telemetry using Wasserstein Distance (WD) and Population Stability Index (PSI). It evaluates both static control models and adaptive continual-learning models using a sliding window and historical reservoir anchor.

## Project Structure

The repository is divided into two independent pipelines for the respective datasets.

concept-drift-nids-wd-psi-main/
├── cicids2018/
│   ├── data/                 # Generated datasets, scalers, and results (ignored in git)
│   └── src/                  # Core pipeline scripts for CIC-IDS2018
├── unswnb15/
│   ├── data/                 # Generated datasets, scalers, and results (ignored in git)
│   ├── raw_csvs/             # Placement folder for raw UNSW-NB15 block files
│   └── src/                  # Core pipeline scripts for UNSW-NB15
├── requirements.txt
└── test_xpu.py               # Utility to check Intel Arc/CUDA hardware acceleration


## Setup Instructions

1. Clone this repository to your local machine.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Acquisition & Preparation

**For CIC-IDS2018:**
The data loader automatically fetches the raw timeline data from the AWS S3 bucket.

```bash
python cicids2018/src/data_loader.py

```

**For UNSW-NB15:**
Download the 4 raw UNSW-NB15 CSV files, place them in `unswnb15/raw_csvs/`, and run the chunking processor:

```bash
python unswnb15/src/data_loader_unsw.py

```

## Execution Pipeline

Both dataset pipelines follow the same sequential execution order. Run these scripts from the repository root. *(The example below uses the CIC-IDS2018 pipeline)*:

**1. Baseline Preprocessing**
Extracts non-zero variance features and fits the global `MinMaxScaler` on the control baseline.

```bash
python cicids2018/src/preprocessing.py

```

**2. Class Balancing**
Balances the dataset to handle minority attack classes. You can choose either Random Oversampling (ROS) or Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP).

```bash
python cicids2018/src/ros_balancer.py
# OR
python cicids2018/src/wgan_balancer.py

```

**3. Concept Drift Metrics**
Calculates the Wasserstein Distance and Population Stability Index across the chronological timeline.

```bash
python cicids2018/src/drift_metrics.py

```

**4. Static Control Experiment**
Trains Random Forest, MLP, 1D-CNN, and LSTM on the static baseline and evaluates degradation against future timeline days.

```bash
python cicids2018/src/static_experiment.py

```

**5. Adaptive Sliding Window Experiment**
Executes the adaptive retraining pipeline utilizing a three-block sliding window and a 5% historical reservoir anchor.

```bash
python cicids2018/src/adaptive_experiment.py

```

*(Ablation configurations for ROS adaptation and No-Reservoir adaptation are also provided in the `src/` directory).*

## Summary of Results

The table below reports the Best-per-Block Mean F1 scores from the ablation study, evaluating the impact of the historical reservoir and balancing mechanisms across the chronological streams.

| Configuration | CIC-IDS2018 F1 | UNSW-NB15 F1 |
| --- | --- | --- |
| **WGAN-GP + Reservoir** | 38.7% | **96.3%** |
| **ROS + Reservoir** | **59.0%** | 95.5% |
| **WGAN-GP (No Reservoir)** | 38.6% | 77.3% |

*Note: Balancing effectiveness depends heavily on dataset attack profiles. WGAN-GP performs best on the diverse attack types in UNSW-NB15, while ROS better preserves the scripted, repetitive attack signatures in CIC-IDS2018.*

