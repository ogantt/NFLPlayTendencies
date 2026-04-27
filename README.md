# NFL Play Call Prediction

Predicting NFL run vs pass plays using machine learning on pre-snap game situation data.

## Overview

This project uses 8 seasons (2009-2016) of NFL play-by-play data to predict whether an offense will run or pass. It also includes an interactive Play Call Advisor that recommends run or pass for any given game scenario.

## Project Structure

```
nfl-play-prediction/
├── preprocess_kaggle.py       # Downloads and cleans raw NFL play-by-play CSVs
├── ml_features.py             # Feature engineering and model training
├── run_or_pass.py             # Interactive Play Call Advisor tool
├── visualize_run_pass.py      # Generates analysis plots
├── data/                      # Raw CSVs (2009-2016) and preprocessed arrays
│   ├── reg_pbp_20XX.csv
│   └── preprocessed/
├── plots/                     # Generated visualizations
└── FinalReport.pdf            # Full project report (PDF)
```

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Preprocess data

```bash
python preprocess_kaggle.py
```

### 3. Train models and see results

```bash
python ml_features.py
```

### 4. Run the Play Call Advisor

```bash
python run_or_pass.py
```

### 5. Generate visualizations

```bash
python visualize_run_pass.py
```

## Models

- **Logistic Regression**: 56.8% accuracy
- **Random Forest**: 61.4% accuracy
- **Naive Baseline** (always predict pass): 54.6%

## Data

NFL play-by-play data from Kaggle (nflscrapR), seasons 2009-2016 (~370K plays).
