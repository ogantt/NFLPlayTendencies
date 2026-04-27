# Generates all the analysis plots (saves as PNG files)
# Usage: python visualize_run_pass.py

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # saves to file instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from run_or_pass import load_data, build_features, train_models, predict_play, build_single_input

def get_prediction(models, scaler, team_map, situation):
    return predict_play(models, scaler, situation, team_map)


def composite_score(result, ydstogo):
    weight = 5.0 if ydstogo <= 3 else 3.0  # weight first down prob more on short yardage
    return result["expected_yards"] + weight * result["first_down_prob"]

BASE_SITUATION = {
    "posteam": "CAR", "defteam": "NE",
    "qtr": 3, "down": 2, "ydstogo": 7,
    "yrdline100": 65, "pos_score": 14, "def_score": 14,
    "drive": 5,
}


def plot_heatmap(models, scaler, team_map, save=True):
    downs = [1, 2, 3, 4]
    ytg_range = range(1, 21)

    advantage = np.zeros((len(downs), len(ytg_range)))

    for i, d in enumerate(downs):
        for j, ytg in enumerate(ytg_range):
            sit = {**BASE_SITUATION, "down": d, "ydstogo": ytg}
            res = get_prediction(models, scaler, team_map, sit)
            run_s = composite_score(res["Run"], ytg)
            pass_s = composite_score(res["Pass"], ytg)
            advantage[i, j] = run_s - pass_s  # positive = run better

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "run_pass", ["#2166ac", "#f7f7f7", "#b2182b"])  # blue=pass, red=run
    vmax = max(abs(advantage.min()), abs(advantage.max()))
    im = ax.imshow(advantage, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ytg_range)))
    ax.set_xticklabels(ytg_range)
    ax.set_yticks(range(len(downs)))
    ax.set_yticklabels([f"{d}{'st' if d==1 else 'nd' if d==2 else 'rd' if d==3 else 'th'} Down" for d in downs])
    ax.set_xlabel("Yards to Go", fontsize=12)
    ax.set_title("Run vs Pass Advantage by Down & Distance\n(Red = Run Better, Blue = Pass Better)",
                 fontsize=14, fontweight="bold")

    for i in range(len(downs)):
        for j in range(len(ytg_range)):
            val = advantage[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            label = "RUN" if val > 0.15 else "PASS" if val < -0.15 else "~"
            ax.text(j, i, f"{label}\n{val:+.1f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Run Advantage (composite score difference)", fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig("plot_heatmap_down_distance.png", dpi=150, bbox_inches="tight")
        print("  Saved: plot_heatmap_down_distance.png")
    plt.show()


def plot_bar_comparison(models, scaler, team_map, situation=None, save=True):
    sit = situation or BASE_SITUATION
    res = get_prediction(models, scaler, team_map, sit)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    ax = axes[0]
    vals = [res["Run"]["expected_yards"], res["Pass"]["expected_yards"]]
    colors = ["#b2182b", "#2166ac"]
    bars = ax.bar(["Run", "Pass"], vals, color=colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Expected Yards", fontsize=11)
    ax.set_title("Expected Yards Gained", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{v:+.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)

    ax = axes[1]
    vals = [res["Run"]["first_down_prob"], res["Pass"]["first_down_prob"]]
    bars = ax.bar(["Run", "Pass"], vals, color=colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("First Down Probability", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax = axes[2]
    run_comp = composite_score(res["Run"], sit["ydstogo"])
    pass_comp = composite_score(res["Pass"], sit["ydstogo"])
    vals = [run_comp, pass_comp]
    bars = ax.bar(["Run", "Pass"], vals, color=colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Overall Score\n(yards + weighted 1st-down prob)", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

    winner = "RUN" if run_comp > pass_comp else "PASS"
    fig.suptitle(f"{sit['posteam']} vs {sit['defteam']} — Q{sit['qtr']} "
                 f"{sit['down']}&{sit['ydstogo']} at opp {sit['yrdline100']}  →  {winner}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig("plot_bar_comparison.png", dpi=150, bbox_inches="tight")
        print("  Saved: plot_bar_comparison.png")
    plt.show()


def plot_field_position(models, scaler, team_map, save=True):
    yard_lines = range(5, 96)
    run_scores = []
    pass_scores = []

    for yl in yard_lines:
        sit = {**BASE_SITUATION, "yrdline100": yl}
        res = get_prediction(models, scaler, team_map, sit)
        run_scores.append(composite_score(res["Run"], sit["ydstogo"]))
        pass_scores.append(composite_score(res["Pass"], sit["ydstogo"]))

    run_scores = np.array(run_scores)
    pass_scores = np.array(pass_scores)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(list(yard_lines), run_scores, color="#b2182b", linewidth=2.5, label="Run")
    ax.plot(list(yard_lines), pass_scores, color="#2166ac", linewidth=2.5, label="Pass")

    ax.fill_between(list(yard_lines), run_scores, pass_scores,
                    where=run_scores > pass_scores, alpha=0.2, color="#b2182b",
                    label="Run advantage")
    ax.fill_between(list(yard_lines), run_scores, pass_scores,
                    where=pass_scores > run_scores, alpha=0.2, color="#2166ac",
                    label="Pass advantage")

    ax.set_xlabel("Yards from Opponent End Zone (yrdline100)", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("Run vs Pass Advantage by Field Position\n"
                 f"(Q{BASE_SITUATION['qtr']}, {BASE_SITUATION['down']}&{BASE_SITUATION['ydstogo']}, "
                 f"score {BASE_SITUATION['pos_score']}-{BASE_SITUATION['def_score']})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5)

    ax.axvspan(0, 10, alpha=0.05, color="green", label="Red zone")
    ax.text(5, ax.get_ylim()[1] * 0.95, "Red\nZone", ha="center", fontsize=9, color="green")

    ax.invert_xaxis()  # closer to end zone on the right
    plt.tight_layout()
    if save:
        plt.savefig("plot_field_position.png", dpi=150, bbox_inches="tight")
        print("  Saved: plot_field_position.png")
    plt.show()


def _batch_predict(models, scaler, team_map, situations_matrix, ytg_array):
    X_scaled = scaler.transform(situations_matrix)
    results = {}
    for play_type in ["Run", "Pass"]:
        exp_yards = models[play_type]["yards"].predict(X_scaled)
        fd_proba = models[play_type]["firstdown"].predict_proba(X_scaled)
        p_fd = fd_proba[:, 1] if fd_proba.shape[1] > 1 else fd_proba[:, 0]
        results[play_type] = {"expected_yards": exp_yards, "first_down_prob": p_fd}

    weights = np.where(ytg_array <= 3, 5.0, 3.0)
    run_comp = results["Run"]["expected_yards"] + weights * results["Run"]["first_down_prob"]
    pass_comp = results["Pass"]["expected_yards"] + weights * results["Pass"]["first_down_prob"]
    return run_comp - pass_comp


def _build_batch_matrix(team_map, overrides_list):
    base = BASE_SITUATION
    pos_id = team_map.get(base["posteam"], 0)
    def_id = team_map.get(base["defteam"], 0)
    rows = []
    for ov in overrides_list:
        s = {**base, **ov}
        rows.append([
            s["qtr"], s["down"], s["ydstogo"], s["yrdline100"],
            s["pos_score"], s["def_score"],
            s["pos_score"] - s["def_score"],
            s["drive"], pos_id, def_id,
        ])
    return np.array(rows, dtype=float)


def plot_decision_contour(models, scaler, team_map, save=True):
    yl_range = np.arange(5, 96, 2)
    ytg_range = np.arange(1, 21, 1)
    YL, YTG = np.meshgrid(yl_range, ytg_range)

    flat_yl = YL.ravel()
    flat_ytg = YTG.ravel()
    overrides = [{"yrdline100": float(yl), "ydstogo": float(ytg)}
                 for yl, ytg in zip(flat_yl, flat_ytg)]
    X_batch = _build_batch_matrix(team_map, overrides)
    Z_flat = _batch_predict(models, scaler, team_map, X_batch, flat_ytg)
    Z = Z_flat.reshape(YL.shape)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "run_pass", ["#2166ac", "#f7f7f7", "#b2182b"])
    vmax = max(abs(Z.min()), abs(Z.max()))
    cf = ax.contourf(YL, YTG, Z, levels=30, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.contour(YL, YTG, Z, levels=[0], colors="black", linewidths=2)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Run Advantage (positive = Run better)", fontsize=11)

    ax.set_xlabel("Yards from Opponent End Zone", fontsize=12)
    ax.set_ylabel("Yards to Go", fontsize=12)
    ax.set_title("Run vs Pass Decision Boundary\n"
                 f"(Q{BASE_SITUATION['qtr']}, {BASE_SITUATION['down']}nd down, "
                 f"score {BASE_SITUATION['pos_score']}-{BASE_SITUATION['def_score']})",
                 fontsize=14, fontweight="bold")
    ax.invert_xaxis()

    ax.text(0.05, 0.95, "RUN\nzone", transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="#b2182b", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.text(0.95, 0.05, "PASS\nzone", transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="#2166ac", va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if save:
        plt.savefig("plot_decision_contour.png", dpi=150, bbox_inches="tight")
        print("  Saved: plot_decision_contour.png")
    plt.show()


def plot_radar(models, scaler, team_map, situation=None, save=True):
    sit = situation or BASE_SITUATION
    res = get_prediction(models, scaler, team_map, sit)

    categories = ["Expected Yards", "1st Down Prob", "Short Yardage\nBenefit",
                   "Composite Score", "Yards Over\nNeeded"]

    run_r = res["Run"]
    pass_r = res["Pass"]

    ytg = sit["ydstogo"]
    run_vals = [
        run_r["expected_yards"],
        run_r["first_down_prob"],
        max(0, run_r["expected_yards"]) / max(ytg, 1),  # how close to 1st down
        composite_score(run_r, ytg),
        max(0, run_r["expected_yards"] - ytg),
    ]
    pass_vals = [
        pass_r["expected_yards"],
        pass_r["first_down_prob"],
        max(0, pass_r["expected_yards"]) / max(ytg, 1),
        composite_score(pass_r, ytg),
        max(0, pass_r["expected_yards"] - ytg),
    ]

    all_vals = run_vals + pass_vals
    v_min = min(all_vals)
    v_max = max(all_vals) if max(all_vals) != min(all_vals) else min(all_vals) + 1
    run_norm = [(v - v_min) / (v_max - v_min) for v in run_vals]
    pass_norm = [(v - v_min) / (v_max - v_min) for v in pass_vals]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    run_norm += run_norm[:1]
    pass_norm += pass_norm[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, run_norm, alpha=0.25, color="#b2182b")
    ax.plot(angles, run_norm, color="#b2182b", linewidth=2, label="Run")
    ax.fill(angles, pass_norm, alpha=0.25, color="#2166ac")
    ax.plot(angles, pass_norm, color="#2166ac", linewidth=2, label="Pass")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f"Run vs Pass Profile\n{sit['posteam']} vs {sit['defteam']} — "
                 f"Q{sit['qtr']} {sit['down']}&{sit['ydstogo']}",
                 fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    if save:
        plt.savefig("plot_radar_comparison.png", dpi=150, bbox_inches="tight")
        print("  Saved: plot_radar_comparison.png")
    plt.show()


def plot_score_diff(models, scaler, team_map, save=True):
    diffs = range(-28, 29, 2)
    run_scores = []
    pass_scores = []

    for sd in diffs:
        sit = {**BASE_SITUATION, "pos_score": 14 + sd, "def_score": 14}
        res = get_prediction(models, scaler, team_map, sit)
        run_scores.append(composite_score(res["Run"], sit["ydstogo"]))
        pass_scores.append(composite_score(res["Pass"], sit["ydstogo"]))

    run_scores = np.array(run_scores)
    pass_scores = np.array(pass_scores)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(list(diffs), run_scores, color="#b2182b", linewidth=2.5, label="Run", marker="o", markersize=4)
    ax.plot(list(diffs), pass_scores, color="#2166ac", linewidth=2.5, label="Pass", marker="s", markersize=4)
    ax.fill_between(list(diffs), run_scores, pass_scores,
                    where=run_scores > pass_scores, alpha=0.2, color="#b2182b")
    ax.fill_between(list(diffs), run_scores, pass_scores,
                    where=pass_scores > run_scores, alpha=0.2, color="#2166ac")

    ax.axvline(0, color="gray", linewidth=1, linestyle="--", label="Tied")
    ax.set_xlabel("Score Differential (Offense - Defense)", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("How Score Differential Affects Run vs Pass Decision\n"
                 f"(Q{BASE_SITUATION['qtr']}, {BASE_SITUATION['down']}&{BASE_SITUATION['ydstogo']})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    ax.text(-20, ax.get_ylim()[1] * 0.9, "← Losing", fontsize=10, color="gray")
    ax.text(15, ax.get_ylim()[1] * 0.9, "Winning →", fontsize=10, color="gray")

    plt.tight_layout()
    if save:
        plt.savefig("plot_score_differential.png", dpi=150, bbox_inches="tight")
        print("  Saved: plot_score_differential.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Run vs Pass Advisor")
    parser.add_argument("--scenario", action="store_true",
                        help="Generate plots for the demo scenario only (bar + radar)")
    parser.add_argument("--fast", action="store_true",
                        help="Use only 2016 data for faster loading")
    args = parser.parse_args()

    print("Loading data and training models...")
    if args.fast:
        import glob
        DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        fast_csv = os.path.join(DATA_DIR, "reg_pbp_2016.csv")
        kaggle_csv = os.path.join(DATA_DIR, "nfl_play_by_play_2009-2016(v3).csv")
        if os.path.exists(fast_csv):
            print("  (fast mode: 2016 season only)")
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
            print("  2016 CSV not found, falling back to full dataset...")
            df = load_data()
    else:
        df = load_data()
    print(f"  Loaded {len(df):,} plays")
    X, team_map = build_features(df)
    models, scaler = train_models(df, X, verbose=True)
    print("Models ready. Generating plots...\n")

    if args.scenario:
        print("[1/2] Bar chart — single scenario comparison")
        plot_bar_comparison(models, scaler, team_map)
        print("[2/2] Radar chart — multi-factor profile")
        plot_radar(models, scaler, team_map)
    else:
        print("[1/6] Heatmap — Down × Yards-to-Go")
        plot_heatmap(models, scaler, team_map)
        print("[2/6] Bar chart — single scenario comparison")
        plot_bar_comparison(models, scaler, team_map)
        print("[3/6] Line plot — field position sweep")
        plot_field_position(models, scaler, team_map)
        print("[4/6] Contour — decision boundary (yard-line × yards-to-go)")
        plot_decision_contour(models, scaler, team_map)
        print("[5/6] Radar chart — multi-factor profile")
        plot_radar(models, scaler, team_map)
        print("[6/6] Score differential impact")
        plot_score_diff(models, scaler, team_map)

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
