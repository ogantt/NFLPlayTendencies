# Loads raw NFL CSVs, cleans them, and saves features + labels as .npy files
# Run this first: python preprocess_kaggle.py

import os
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing


KAGGLE_CSV = "data/nfl_play_by_play_2009-2016(v3).csv"
NFLSCRAPR_PATTERN = "data/reg_pbp_*.csv"
X_NV_OUTPUT = "data/preprocessed/X_nv_pp"
Y_OUTPUT = "data/preprocessed/y_pp"

if os.path.exists(KAGGLE_CSV):
    print(f"Loading Kaggle CSV: {KAGGLE_CSV}")
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)
else:
    csvs = sorted(glob.glob(NFLSCRAPR_PATTERN))
    if not csvs:
        raise FileNotFoundError(
            f"No data found. Place either '{KAGGLE_CSV}' or nflscrapR "
            f"per-season CSVs (reg_pbp_YYYY.csv) in data/")
    print(f"Loading {len(csvs)} nflscrapR season files...")
    frames = []
    for f in csvs:
        print(f"  {f}")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    col_map = {
        "drive": "Drive",
        "defteam": "DefensiveTeam",
        "posteam_score": "PosTeamScore",
        "defteam_score": "DefTeamScore",
        "yards_gained": "Yards.Gained",
        "play_type": "PlayType",
        "yardline_100": "yrdline100",
    }
    df = df.rename(columns=col_map)

    if "FirstDown" not in df.columns:
        fd_cols = ["first_down_rush", "first_down_pass", "first_down_penalty"]
        existing = [c for c in fd_cols if c in df.columns]
        if existing:
            df["FirstDown"] = df[existing].fillna(0).max(axis=1)
        else:
            df["FirstDown"] = 0

    play_map = {"run": "Run", "pass": "Pass", "kickoff": "Kickoff",
                "no_play": "No Play", "punt": "Punt", "field_goal": "Field Goal",
                "extra_point": "Extra Point", "qb_kneel": "QB Kneel",
                "qb_spike": "QB Spike"}
    df["PlayType"] = df["PlayType"].map(play_map).fillna(df["PlayType"])

print(f"  Raw rows: {len(df)}")

df = df[~df["PlayType"].isin(["Kickoff", "No Play", "Punt", "Field Goal",
                               "Extra Point", "QB Kneel", "QB Spike"])]
df = df[df["PlayType"].isin(["Run", "Pass"])]  # only keep run/pass plays
required_cols = ["Drive", "qtr", "down", "time", "yrdln", "yrdline100",
                 "ydstogo", "posteam", "DefensiveTeam", "PosTeamScore",
                 "DefTeamScore", "Yards.Gained", "PlayType", "FirstDown"]
df = df.dropna(subset=required_cols)
print(f"  Rows after filtering: {len(df)}")

df["time_minute"] = df["time"].astype(str).str.split(":").str[0]

one_hot_cols = ["Drive", "qtr", "down", "time_minute", "posteam", "DefensiveTeam"]
enc = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")

print("One-hot encoding...")
one_hot_arrays = []
for col in one_hot_cols:
    vals = df[col].values.reshape(-1, 1)
    enc_single = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = enc_single.fit_transform(vals)
    one_hot_arrays.append(encoded)
    print(f"  {col}: {encoded.shape[1]} categories")

df = df.copy()
try:  # some CSVs have text like "CAR 35" instead of just a number
    df["yrdln"] = pd.to_numeric(df["yrdln"])
except (ValueError, TypeError):
    df["yrdln"] = df["yrdln"].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
numeric_cols = ["yrdln", "yrdline100", "ydstogo", "PosTeamScore", "DefTeamScore"]
X_numeric = df[numeric_cols].values.astype("float64")

X_features = np.concatenate(one_hot_arrays + [X_numeric], axis=1).astype("float64")
print(f"  X_features shape: {X_features.shape}")

y_real = df["Yards.Gained"].values.astype("float64").reshape(-1, 1)
y_play = np.array([0.0 if x == "Run" else 1.0 for x in df["PlayType"]]).reshape(-1, 1)  # 0=run, 1=pass
y_posplay = (1.0 * (df["Yards.Gained"].values > 0)).reshape(-1, 1)
y_ontrack = (1.0 * (df["Yards.Gained"].values > 3)).reshape(-1, 1)
y_firstdown = df["FirstDown"].values.astype("float64").reshape(-1, 1)

Y = np.concatenate([y_real, y_play, y_posplay, y_ontrack, y_firstdown], axis=1)
print(f"  Y shape: {Y.shape}")
print(f"  Y columns: [Yards.Gained, PlayType(0=Run/1=Pass), PosYards, OnTrack(>3), FirstDown]")

os.makedirs(os.path.dirname(X_NV_OUTPUT), exist_ok=True)
np.save(X_NV_OUTPUT, X_features)
np.save(Y_OUTPUT, Y)
print(f"\nSaved: {X_NV_OUTPUT}.npy")
print(f"Saved: {Y_OUTPUT}.npy")
print("Done!")
