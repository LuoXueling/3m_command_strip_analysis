import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tape import h_a, h_b, E_ad, L, L1, L2, C10, C20, C30, H, choose_traction, mu


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


def _find_single_column(columns, token):
    matches = [c for c in columns if token in c]
    if not matches:
        raise KeyError("Missing column containing: %s" % token)
    return matches[0]


def _newton_solve(func, x0, tol=1e-10, max_iter=50):
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        if abs(fx) < tol:
            return x
        step = 1e-6 * max(1.0, abs(x))
        dfx = (func(x + step) - func(x - step)) / (2.0 * step)
        if dfx == 0.0:
            break
        x_next = x - fx / dfx
        if not np.isfinite(x_next):
            break
        x = x_next
    return x


def _yeoh_sener(i1, c10, c20, c30):
    i1m3 = i1 - 3.0
    return c10 * i1m3 + c20 * i1m3**2 + c30 * i1m3**3


def analytical_energy_release_rate(time, mode="release"):
    if mode not in ["release", "peel"]:
        raise ValueError("Invalid mode. Choose 'release' or 'peel'.")

    force_scale = choose_traction(mode)
    force = force_scale * float(time)
    if mode == "release":
        force *= h_a
        guess = 1.0

        def func(lmbd):
            i1 = lmbd**-2 + lmbd**2 + 1.0
            term = C10 + 2.0 * C20 * (i1 - 3.0) + 3.0 * C30 * (i1 - 3.0) ** 2
            return 2.0 * h_a * (lmbd - lmbd**-3) * term - force

        guess = _newton_solve(func, guess)
        if guess <= 0.0:
            guess = 1e-6
        i1 = guess**-2 + guess**2 + 1.0
        sener = _yeoh_sener(i1, C10, C20, C30)
        return force * (guess - 1.0) - sener

    span = L - max(L1, L2)
    guess = 0.0
    target = force / span

    def func(gamma):
        i1 = gamma**2 + 3.0
        term = C10 + 2.0 * C20 * (i1 - 3.0) + 3.0 * C30 * (i1 - 3.0) ** 2
        return 2.0 * gamma * term - target

    guess = _newton_solve(func, guess)
    i1 = guess**2 + 3.0
    return _yeoh_sener(i1, C10, C20, C30)


def plot_fracture(csv_path, output_path, mode):
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise KeyError("Missing Time column in %s" % csv_path)

    time = df["Time"].astype(float)
    force_scale = choose_traction(mode)
    force = force_scale * time * h_a if mode == "release" else force_scale * time * h_b

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

    j_bot_avg = df[bot_cols].mean(axis=1)
    j_top_avg = df[top_cols].mean(axis=1)
    j_sum = j_bot_avg + j_top_avg

    le11_col = _find_single_column(df.columns, "LE11")
    le12_col = _find_single_column(df.columns, "LE12")
    sener_col = _find_single_column(df.columns, "SENER")
    le11_num = df[le11_col].astype(float).to_numpy()
    le12_num = df[le12_col].astype(float).to_numpy()
    sener_num = df[sener_col].astype(float).to_numpy()
    force_energy = np.zeros_like(force, dtype=float)
    sener = np.zeros_like(force, dtype=float)
    analytical_strain = np.zeros_like(force, dtype=float)

    if mode == "release":
        guess = 1.0
        for i, fval in enumerate(force):

            def func(lmbd):
                i1 = lmbd**-2 + lmbd**2 + 1.0
                term = C10 + 2.0 * C20 * (i1 - 3.0) + 3.0 * C30 * (i1 - 3.0) ** 2
                return 2.0 * h_a * (lmbd - lmbd**-3) * term - fval

            guess = _newton_solve(func, guess)
            if guess <= 0.0:
                guess = 1e-6
            i1 = guess**-2 + guess**2 + 1.0
            analytical_strain[i] = guess
            sener[i] = _yeoh_sener(i1, C10, C20, C30)
            force_energy[i] = fval * (guess - 1.0) - sener[i]

        small_deformation = 3.0 * force**2 / (8.0 * E_ad)
        numerical_strain = np.exp(le11_num)
    else:
        gamma_col = _find_single_column(df.columns, "Gamma")
        gamma_num = df[gamma_col].astype(float).to_numpy()
        span = L - max(L1, L2)
        guess = 0.0
        for i, fval in enumerate(force):
            target = fval / span

            def func(gamma):
                i1 = gamma**2 + 3.0
                term = C10 + 2.0 * C20 * (i1 - 3.0) + 3.0 * C30 * (i1 - 3.0) ** 2
                return 2.0 * gamma * term - target

            guess = _newton_solve(func, guess)
            i1 = guess**2 + 3.0
            analytical_strain[i] = guess
            sener[i] = _yeoh_sener(i1, C10, C20, C30)
            force_energy[i] = sener[i]

        small_deformation = 3.0 * force**2 / (2.0 * E_ad * (L - max(L1, L2)) ** 2)
        numerical_strain = -gamma_num

    if mode == "release":
        force_energy_num = force * (np.exp(le11_num) - 1.0) - sener_num
    else:
        force_energy_num = sener_num.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        sener_rel = np.where(sener_num != 0.0, (sener - sener_num) / sener_num, np.nan)
        force_energy_rel = np.where(
            force_energy_num != 0.0,
            (force_energy - force_energy_num) / force_energy_num,
            np.nan,
        )

    print("Numerical sener:", sener_num)
    print("Analytical sener:", sener)
    print("Relative diff (sener):", sener_rel)
    print("Numerical force_energy:", np.array(force_energy_num))
    print("Analytical force_energy:", force_energy)
    print("Relative diff (force_energy):", force_energy_rel)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_num_vs_j = np.where(
            j_sum != 0.0, (force_energy_num - j_sum) / j_sum, np.nan
        )
        rel_ana_vs_j = np.where(j_sum != 0.0, (force_energy - j_sum) / j_sum, np.nan)

    print("Relative diff (numerical force_energy vs J):", rel_num_vs_j)
    print("Relative diff (analytical force_energy vs J):", rel_ana_vs_j)

    fig_size = (7.5, 3.2)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=fig_size, gridspec_kw={"width_ratios": [1, 2]}
    )
    color_left = "#1b9e77"
    color_right = "#d95f02"
    color_mid = "#7570b3"

    if mode == "release":
        force_normalized = force / (mu * h_a)
    else:
        force_normalized = force / (mu * ((L - max(L1, L2))))

    def plot_panel(ax1, mask, show_legend):
        print(1 / (mu * h_a))
        ax1.plot(
            force_normalized[mask],
            j_sum[mask] / (mu * h_a),
            color=color_left,
            linestyle="None",
            marker="o",
            markerfacecolor="none",
            label=r"$\overline{J}^+ + \overline{J}^-$ (FEM)",
        )
        if mode == "peel":
            ax1.plot(
                force_normalized[mask],
                j_top_avg[mask] / (mu * h_a),
                color=color_left,
                linestyle="None",
                marker="^",
                markerfacecolor="none",
                label=r"$\overline{J}^+$ (FEM)",
            )
        ax1.set_xlabel(
            r"Normalized force $\overline{F}$"
            if mode == "release"
            else r"Normalized weight $\overline{W}$"
        )
        ax1.set_ylabel("Normalized FEM J-integral or\nanalytical energy release rate")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.plot(
            force_normalized[mask],
            force_energy[mask] / (mu * h_a),
            color=color_right,
            label=(
                r"$\overline{G}_w^+$ (Finite deformation)"
                if mode == "peel"
                else r"$2\overline{G}_r^+$ (Finite deformation)"
            ),
        )
        ax1.plot(
            force_normalized[mask],
            small_deformation[mask] / (mu * h_a),
            color=color_mid,
            linestyle="--",
            label=(
                r"$\overline{G}_w^+$ (Small deformation)"
                if mode == "peel"
                else r"$2\overline{G}_r^+$ (Small deformation)"
            ),
        )

        if show_legend:
            ax1.legend(
                loc="upper left",
                frameon=False,
                fontsize=9,
            )

    ax_left.plot(force_normalized, analytical_strain, color="k", label="Analytical")
    ax_left.plot(
        force_normalized,
        numerical_strain,
        color="#4d4d4d",
        linestyle="None",
        marker="o",
        markerfacecolor="none",
        label="FEM",
    )
    ax_left.set_xlabel(
        r"Normalized force $\overline{F}$"
        if mode == "release"
        else r"Normalized weight $\overline{W}$"
    )
    ax_left.set_ylabel(
        "Longitudinal stretch $\\lambda$"
        if mode == "release"
        else r"Shear strain $\gamma_\infty$"
    )
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.legend(frameon=False, fontsize=9, loc="upper left")

    plot_panel(ax_right, np.ones_like(force_normalized, dtype=bool), show_legend=True)

    j_min = float(np.nanmin(j_sum))
    j_max = float(np.nanmax(j_sum))
    f_min = float(np.nanmin([force_energy.min(), small_deformation.min()]))
    f_max = float(np.nanmax([force_energy.max(), small_deformation.max()]))
    if mode == "peel":
        y_min = min(j_min, float(np.nanmin(force_energy))) / (mu * h_a)
        y_max = max(j_max, float(np.nanmax(force_energy))) / (mu * h_a)
    else:
        y_min = min(j_min, f_min) / (mu * h_a)
        y_max = max(j_max, f_max) / (mu * h_a)
    y_pad = 0.03 * (y_max - y_min) if y_max > y_min else 0.0
    y_lower = y_min - y_pad
    y_upper = y_max + y_pad

    ax_right.set_ylim(y_lower, y_upper)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot fracture metrics from Abaqus CSV output."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Path to Tape_Fracture_*.csv (defaults to latest for mode).",
    )
    parser.add_argument("--mode", choices=["release", "peel"], default="peel")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        if args.mode == "release":
            csv_path = _find_latest_csv("Tape_Fracture_release*.csv")
        else:
            csv_path = _find_latest_csv("Tape_Fracture_peel*.csv")

    plot_fracture(csv_path, f"fracture_plot_{args.mode}.pdf", args.mode)


if __name__ == "__main__":
    main()
