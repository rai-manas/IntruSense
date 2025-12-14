# IntruSense: Multimodal IoT Intrusion Detection System

This repository contains the implementation of **IntruSense**, a lightweight
machine-learning–based intrusion detection system (IDS) for IoT environments.


## Problem Statement
IoT and IIoT networks are vulnerable to attacks such as DoS, DDoS, and ransomware.
Traditional IDS solutions are unsuitable due to resource constraints of IoT devices.
This project explores **machine-learning–based intrusion detection** using
**multimodal data fusion**.

## Dataset
- **TON-IoT Dataset** (UNSW Canberra IoT Lab)
- Modalities used:
  - Network traffic
  - Linux / Windows system logs
  - Telemetry data

Dataset link:  
https://research.unsw.edu.au/projects/toniot-datasets

> Note: Due to size constraints, datasets are not included in this repository.

## Methodology
1. Data preprocessing (cleaning, encoding, scaling)
2. Training per-modality ML models
3. Late-fusion using probability averaging
4. Performance evaluation using accuracy, precision, recall, F1-score

## Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest (baseline)
- Fusion-based classifier

## Project Structure
- `src/` – Source code
- `results/` – Metrics and confusion matrices
- `plots/` – Generated visualizations

## How to Run
```bash
pip install -r requirements.txt
python src/fus.py
# IntruSense

```

## Results
All experimental results, confusion matrices, and evaluation metrics
are available in the `results/` and `plots/` directories.
