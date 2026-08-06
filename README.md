# Concept Drift Adaptation on NIDS using WD and PSI

This repository contains the code for the paper "Concept Drift Adaptation on NIDS using WD and PSI". The project tests how well static and adaptive machine learning models handle network intrusions over time using the CICIDS2018 dataset.

The pipeline handles fetching data, balancing minority classes with WGAN-GP, and tracking concept drift using Wasserstein Distance (WD) and Population Stability Index (PSI).

## Project Structure

- `src/` - Core Python scripts for data loading, balancing, drift metrics, and model training.
- `notebooks/main_experiment.ipynb` - The main Google Colab notebook used to run the whole pipeline sequentially.
- `requirements.txt` - Python dependencies needed to run the code.

## Setup Instructions

1. Clone this repository to your local machine or Google Drive.
2. If running locally, install the required dependencies:
   `pip install -r requirements.txt`
3. Open `notebooks/main_experiment.ipynb`.
4. Update the `PROJECT_FOLDER` variable in the first cell to point to your working directory.

## How to Run

The entire workflow is orchestrated step-by-step through the `main_experiment.ipynb` notebook:

1. **Data Loading:** Downloads the raw CICIDS2018 dataset (Feb-Mar) from AWS S3, cleans it, and samples it.
2. **Preprocessing:** Sets up the global scaling bounds using normal traffic from the Feb 14-16 baseline.
3. **Balancing:** Trains a WGAN-GP to generate synthetic data for starved minority attack classes.
4. **Drift Metrics:** Calculates WD and PSI to measure how much the dataset drifts across the timeline.
5. **Static Experiment:** Trains a static control group (RF, MLP, 1D-CNN, LSTM) on the baseline and evaluates it against future days.
6. **Adaptive Experiment:** Uses a sliding window training approach with a 5% historical anchor to adapt models to new drift patterns.

## Output

All generated files, including cleaned CSVs, WGAN-balanced datasets, global scalers (`.pkl`), and final performance metrics (`.csv`), will be saved directly into your designated `PROJECT_FOLDER`.
