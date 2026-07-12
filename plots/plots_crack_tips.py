import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tape import h_a, h_b, choose_traction, mu, L, L1, L2
from fit_models import fit_polynomial


def _find_latest_csv(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No CSV files found matching: %s" % pattern)
    return max(files, key=os.path.getmtime)


def _contour_columns(columns, prefix, start, end):
    cols = []
    for name in columns:
        if not name.startswith(prefix):
            continue
        match = re.search(r"Contour_(\d+)$", name)
        if not match:
            continue
        idx = int(match.group(1))
        if start <= idx <= end:
            cols.append(name)
    return cols


def _load_avg_j(csv_path, mode):
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise KeyError("Missing Time column in %s" % csv_path)

    time = df["Time"].astype(float)
    force_scale = choose_traction(mode)
    if mode == "release":
        force = force_scale * time * h_a / (mu * h_a)
    else:
        force = force_scale * time * h_b / (mu * ((L - max(L1, L2))))

    bot_cols = _contour_columns(
        df.columns,
        "J at HOUT_J-BOT",
        20,
        30,
    )
    top_cols = _contour_columns(
        df.columns,
        "J at HOUT_J-TOP",
        20,
        30,
    )
    if len(bot_cols) != 11 or len(top_cols) != 11:
        raise ValueError(
            "Expected 11 contours (20-30) for both BOT and TOP; found BOT=%d, TOP=%d"
            % (len(bot_cols), len(top_cols))
        )

    j_bot_avg = df[bot_cols].mean(axis=1) / (mu * h_a)
    j_top_avg = df[top_cols].mean(axis=1) / (mu * h_a)

    return force, j_top_avg, j_bot_avg


def plot_crack_tip_j(release_csv, peel_csv, output_path, show_fit=False):
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    color_top = "#377eb8"
    color_bot = "#e41a1c"

    for ax, mode, csv_path in [
        (axes[0], "peel", peel_csv),
        (axes[1], "release", release_csv),
    ]:
        force, j_top_avg, j_bot_avg = _load_avg_j(csv_path, mode)
        if show_fit:
            # scatter original data points
            ax.plot(
                force,
                j_top_avg,
                color=color_top,
                marker="o",
                linestyle="None",
                label=r"Top crack ($\overline{J}^+$)",
            )
            ax.plot(
                force,
                j_bot_avg,
                color=color_bot,
                marker="o",
                linestyle="None",
                label=r"Bottom crack ($\overline{J}^-$)",
            )
            # fit cubic polynomials (through origin) and plot smooth lines
            (pA, pB, pC, pD), model_top = fit_polynomial(
                force, j_top_avg, degree=3, force_through_origin=True
            )
            (mA, mB, mC, mD), model_bot = fit_polynomial(
                force, j_bot_avg, degree=3, force_through_origin=True
            )
            Fq = np.linspace(float(np.nanmin(force)), float(np.nanmax(force)), 250)
            ax.plot(Fq, model_top(Fq), color=color_top, linestyle="-", label=r"Top fit")
            ax.plot(
                Fq, model_bot(Fq), color=color_bot, linestyle="-", label=r"Bottom fit"
            )
        else:
            ax.plot(
                force,
                j_top_avg,
                color=color_top,
                # marker="o",
                label=r"Top crack ($\overline{J}^+$)",
            )
            ax.plot(
                force,
                j_bot_avg,
                color=color_bot,
                # marker="o",
                label=r"Bottom crack ($\overline{J}^-$)",
            )
        ax.set_xlabel(
            r"Normalized force $\overline{F}$"
            if mode == "release"
            else r"Normalized weight $\overline{W}$"
        )
        ax.set_ylabel("Normalized FEM $J$-integral")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot average J-integral vs force for both crack tips."
    )
    parser.add_argument(
        "--release-csv",
        default=None,
        help="Path to Tape_Fracture_release*.csv (defaults to latest).",
    )
    parser.add_argument(
        "--peel-csv",
        default=None,
        help="Path to Tape_Fracture_peel*.csv (defaults to latest).",
    )
    parser.add_argument(
        "--out",
        default="fracture_plot_j_tips.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--show-fit",
        action="store_true",
        default=False,
        help="When set, plot original data as circles and fitted exponential as lines",
    )
    args = parser.parse_args()

    release_csv = args.release_csv or _find_latest_csv("Tape_Fracture_release*.csv")
    peel_csv = args.peel_csv or _find_latest_csv("Tape_Fracture_peel*.csv")

    plot_crack_tip_j(release_csv, peel_csv, args.out, show_fit=args.show_fit)


if __name__ == "__main__":
    main()
