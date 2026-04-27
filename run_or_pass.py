# Interactive tool that recommends run or pass for a given game situation
# Usage: python run_or_pass.py --demo   or   python run_or_pass.py

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_data():
    kaggle_csv = os.path.join(DATA_DIR, "nfl_play_by_play_2009-2016(v3).csv")
    pattern = os.path.join(DATA_DIR, "reg_pbp_*.csv")

    if os.path.exists(kaggle_csv):
        df = pd.read_csv(kaggle_csv, low_memory=False)
    else:
        csvs = sorted(glob.glob(pattern))
        if not csvs:
            print("ERROR: No data files found in data/.")
            print("Run preprocess_kaggle.py first or place CSV files in data/.")
            sys.exit(1)
        frames = [pd.read_csv(f, low_memory=False) for f in csvs]
        df = pd.concat(frames, ignore_index=True)

        col_map = {
            "drive": "Drive", "defteam": "DefensiveTeam",
            "posteam_score": "PosTeamScore", "defteam_score": "DefTeamScore",
            "yards_gained": "Yards.Gained", "play_type": "PlayType",
            "yardline_100": "yrdline100",
        }
        df = df.rename(columns=col_map)

        if "FirstDown" not in df.columns:
            fd_cols = ["first_down_rush", "first_down_pass", "first_down_penalty"]
            existing = [c for c in fd_cols if c in df.columns]
            df["FirstDown"] = df[existing].fillna(0).max(axis=1) if existing else 0

        play_map = {"run": "Run", "pass": "Pass"}
        df["PlayType"] = df["PlayType"].map(play_map).fillna(df["PlayType"])

    df = df[df["PlayType"].isin(["Run", "Pass"])]
    required = ["Drive", "qtr", "down", "yrdline100", "ydstogo",
                 "posteam", "DefensiveTeam", "PosTeamScore", "DefTeamScore",
                 "Yards.Gained", "PlayType", "FirstDown"]
    df = df.dropna(subset=required)
    return df



def build_features(df):
    all_teams = sorted(set(df["posteam"].unique()) | set(df["DefensiveTeam"].unique()))
    team_map = {t: i for i, t in enumerate(all_teams)}

    X = pd.DataFrame({
        "qtr": df["qtr"].astype(float),
        "down": df["down"].astype(float),
        "ydstogo": df["ydstogo"].astype(float),
        "yrdline100": df["yrdline100"].astype(float),
        "PosTeamScore": df["PosTeamScore"].astype(float),
        "DefTeamScore": df["DefTeamScore"].astype(float),
        "ScoreDiff": df["PosTeamScore"].astype(float) - df["DefTeamScore"].astype(float),
        "Drive": df["Drive"].astype(float),
        "posteam_id": df["posteam"].map(team_map).astype(float),
        "defteam_id": df["DefensiveTeam"].map(team_map).astype(float),
    })
    return X.values, team_map


def build_single_input(situation, team_map):
    pos_id = team_map.get(situation["posteam"], 0)
    def_id = team_map.get(situation["defteam"], 0)
    score_diff = situation["pos_score"] - situation["def_score"]

    return np.array([[
        situation["qtr"],
        situation["down"],
        situation["ydstogo"],
        situation["yrdline100"],
        situation["pos_score"],
        situation["def_score"],
        score_diff,
        situation["drive"],
        pos_id,
        def_id,
    ]])



def train_models(df, X, verbose=True):
    y_yards = df["Yards.Gained"].values.astype(float)
    y_fd = df["FirstDown"].values.astype(float)
    is_run = (df["PlayType"] == "Run").values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}  # trains separate models for run plays and pass plays
    for play_type, mask in [("Run", is_run), ("Pass", ~is_run)]:
        Xp, yy, yf = X_scaled[mask], y_yards[mask], y_fd[mask]
        if verbose:
            print(f"  Training {play_type} yards model ({mask.sum():,} plays)...", end="", flush=True)
        m_yards = RandomForestRegressor(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        m_yards.fit(Xp, yy)
        if verbose:
            print(" done.")

        if verbose:
            print(f"  Training {play_type} first-down model...", end="", flush=True)
        m_fd = RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        m_fd.fit(Xp, yf)
        if verbose:
            print(" done.")

        models[play_type] = {"yards": m_yards, "firstdown": m_fd}

    return models, scaler



def predict_play(models, scaler, situation, team_map):
    x_raw = build_single_input(situation, team_map)
    x_scaled = scaler.transform(x_raw)

    results = {}
    for play_type in ["Run", "Pass"]:
        exp_yards = models[play_type]["yards"].predict(x_scaled)[0]
        fd_prob = models[play_type]["firstdown"].predict_proba(x_scaled)[0]
        p_fd = fd_prob[1] if len(fd_prob) > 1 else fd_prob[0]  # index 1 = probability of first down
        results[play_type] = {"expected_yards": exp_yards, "first_down_prob": p_fd}

    return results



def display_results(situation, results):
    run = results["Run"]
    pas = results["Pass"]

    print("\n" + "=" * 55)
    print("  NFL PLAY-CALL ADVISOR")
    print("=" * 55)
    print(f"  Situation: {situation['posteam']} vs {situation['defteam']}")
    print(f"  Q{situation['qtr']}  {situation['down']}&{situation['ydstogo']}"
          f"  at opp {situation['yrdline100']}-yard line")
    print(f"  Score: {situation['posteam']} {situation['pos_score']}"
          f" - {situation['defteam']} {situation['def_score']}")
    print(f"  Drive #{situation['drive']}")
    print("-" * 55)
    print(f"  {'':15s} {'Expected Yards':>15s} {'1st Down Prob':>15s}")
    print(f"  {'RUN':15s} {run['expected_yards']:>+15.2f} {run['first_down_prob']:>14.1%}")
    print(f"  {'PASS':15s} {pas['expected_yards']:>+15.2f} {pas['first_down_prob']:>14.1%}")
    print("-" * 55)

    run_score = run["expected_yards"] + 3.0 * run["first_down_prob"]  # composite: yards + weighted 1st down prob
    pass_score = pas["expected_yards"] + 3.0 * pas["first_down_prob"]

    if situation["ydstogo"] <= 3:
        run_score = run["expected_yards"] + 5.0 * run["first_down_prob"]
        pass_score = pas["expected_yards"] + 5.0 * pas["first_down_prob"]

    if run_score > pass_score:
        rec = "RUN"
        margin = run_score - pass_score
    else:
        rec = "PASS"
        margin = pass_score - run_score

    confidence = "Strong" if margin > 1.5 else "Slight"
    print(f"\n  >>> RECOMMENDATION: {rec} ({confidence} advantage)")
    print(f"      Composite score — Run: {run_score:.2f}  Pass: {pass_score:.2f}")
    print("=" * 55)



def get_valid_input(prompt, valid_values=None, dtype=float, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return dtype(default)
        try:
            val = dtype(raw)
        except ValueError:
            print(f"    Invalid input. Please enter a valid {dtype.__name__}.")
            continue
        if valid_values and val not in valid_values:
            print(f"    Must be one of: {valid_values}")
            continue
        return val


def get_situation_interactive(team_list):
    print("\nEnter the game situation:")
    print(f"  Available teams: {', '.join(sorted(team_list))}\n")

    posteam = input("  Offensive team (abbreviation): ").strip().upper()
    while posteam not in team_list:
        print(f"    Unknown team '{posteam}'. Pick from: {', '.join(sorted(team_list))}")
        posteam = input("  Offensive team: ").strip().upper()

    defteam = input("  Defensive team (abbreviation): ").strip().upper()
    while defteam not in team_list:
        print(f"    Unknown team '{defteam}'.")
        defteam = input("  Defensive team: ").strip().upper()

    qtr = get_valid_input("Quarter (1-5, 5=OT)", valid_values={1,2,3,4,5}, dtype=int)
    down = get_valid_input("Down (1-4)", valid_values={1,2,3,4}, dtype=int)
    ydstogo = get_valid_input("Yards to go", dtype=float, default=10)
    yrdline100 = get_valid_input("Yard line (yards from opponent end zone, 1-99)", dtype=float)
    pos_score = get_valid_input("Offensive team score", dtype=float, default=0)
    def_score = get_valid_input("Defensive team score", dtype=float, default=0)
    drive = get_valid_input("Drive number", dtype=float, default=5)

    return {
        "posteam": posteam,
        "defteam": defteam,
        "qtr": qtr,
        "down": down,
        "ydstogo": ydstogo,
        "yrdline100": yrdline100,
        "pos_score": pos_score,
        "def_score": def_score,
        "drive": drive,
    }



def main():
    parser = argparse.ArgumentParser(description="NFL Play-Call Advisor: Run or Pass?")
    parser.add_argument("--demo", action="store_true", help="Run a demo scenario")
    parser.add_argument("--fast", action="store_true",
                        help="Use only 2016 season for faster loading (~30s vs ~2min)")
    args = parser.parse_args()

    print("=" * 55)
    print("  NFL PLAY-CALL ADVISOR — Loading data...")
    print("=" * 55)

    if args.fast:
        fast_csv = os.path.join(DATA_DIR, "reg_pbp_2016.csv")
        if os.path.exists(fast_csv):
            df = pd.read_csv(fast_csv, low_memory=False)
            col_map = {
                "drive": "Drive", "defteam": "DefensiveTeam",
                "posteam_score": "PosTeamScore", "defteam_score": "DefTeamScore",
                "yards_gained": "Yards.Gained", "play_type": "PlayType",
                "yardline_100": "yrdline100",
            }
            df = df.rename(columns=col_map)
            if "FirstDown" not in df.columns:
                fd_cols = ["first_down_rush", "first_down_pass", "first_down_penalty"]
                existing = [c for c in fd_cols if c in df.columns]
                df = df.copy()
                df["FirstDown"] = df[existing].fillna(0).max(axis=1) if existing else 0
            play_map = {"run": "Run", "pass": "Pass"}
            df["PlayType"] = df["PlayType"].map(play_map).fillna(df["PlayType"])
            df = df[df["PlayType"].isin(["Run", "Pass"])]
            required = ["Drive", "qtr", "down", "yrdline100", "ydstogo",
                         "posteam", "DefensiveTeam", "PosTeamScore", "DefTeamScore",
                         "Yards.Gained", "PlayType", "FirstDown"]
            df = df.dropna(subset=required)
        else:
            print("  2016 data not found, loading all seasons...")
            df = load_data()
    else:
        df = load_data()
    print(f"  Loaded {len(df):,} plays (Run: {(df['PlayType']=='Run').sum():,},"
          f" Pass: {(df['PlayType']=='Pass').sum():,})")

    X, team_map = build_features(df)
    team_list = set(team_map.keys())

    print("\nTraining models (this takes ~30-60 seconds)...")
    models, scaler = train_models(df, X)
    print("Models ready!\n")

    if args.demo:
        situation = {
            "posteam": "CAR", "defteam": "NE",
            "qtr": 3, "down": 2, "ydstogo": 7,
            "yrdline100": 65, "pos_score": 14, "def_score": 14,
            "drive": 5,
        }
        results = predict_play(models, scaler, situation, team_map)
        display_results(situation, results)
        return

    while True:
        situation = get_situation_interactive(team_list)
        results = predict_play(models, scaler, situation, team_map)
        display_results(situation, results)

        again = input("\nAnalyze another situation? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for using the NFL Play-Call Advisor!")
            break


if __name__ == "__main__":
    main()
