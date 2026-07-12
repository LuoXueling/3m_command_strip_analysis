import argparse
import matplotlib.pyplot as plt
import numpy as np

from plots import analytical_energy_release_rate
from release_50_utils import load_release_data
from tape import mu, h_a, choose_traction
from plots_crack_tips import _find_latest_csv, _load_avg_j
from fit_models import fit_polynomial, fit_tanh


def main():
    parser = argparse.ArgumentParser(
        description="Plot J-integral vs crack distance at a specific time."
    )
    parser.add_argument("--time", type=float, required=True, help="Target time value.")
    parser.add_argument(
        "--show-fit",
        action="store_true",
        default=False,
        help="When set, plot original data (circles) and fitted functions (lines)",
    )
    parser.add_argument("--ylim-top", type=float, default=None)
    parser.add_argument("--ylim-bottom", type=float, default=None)
    parser.add_argument("--show-legend", action="store_true", default=False)
    parser.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: release_50_summary_t<time>.pdf).",
    )
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    dist, j_top, j_bot = load_release_data(args.time)

    # Normalize distance and J for fitting/plotting
    dist_norm = dist / h_a
    j_top_norm = j_top / (mu * h_a)
    j_bot_norm = j_bot / (mu * h_a)

    # If requested, build fits using (1) release crack-tip vs F data and (2) d/h_a at time=1.0
    if args.show_fit:
        # 1) Exponential fits J(F) using the release panel data from plots_crack_tips
        try:
            release_csv = _find_latest_csv("Tape_Fracture_release*.csv")
            F_data, j_top_F, j_bot_F = _load_avg_j(release_csv, "release")
        except Exception as e:
            raise RuntimeError(
                "Failed to load crack-tip force data for fitting: %s" % e
            )

        (A_plus_F, B_plus_F, *C_plus_F), Jplus_of_F = fit_polynomial(F_data, j_top_F, degree=3, force_through_origin=True)
        (A_minus_F, B_minus_F, *C_minus_F), Jminus_of_F = fit_polynomial(
            F_data, j_bot_F, degree=3, force_through_origin=True
        )

        coeffs_plus_F, Jplus_of_F = fit_polynomial(F_data, j_top_F, degree=3, force_through_origin=True)
        coeffs_minus_F, Jminus_of_F = fit_polynomial(
            F_data, j_bot_F, degree=3, force_through_origin=True
        )
        # Unpack coefficients: tuple of (coeff_F3, coeff_F2, coeff_F1, 0.0)
        coeff_plus_F3, coeff_plus_F2, coeff_plus_F1, _ = coeffs_plus_F
        coeff_minus_F3, coeff_minus_F2, coeff_minus_F1, _ = coeffs_minus_F

        print("=" * 70)
        print("POLYNOMIAL FIT FOR J^+(F) and J^-(F)")
        print("=" * 70)
        print(f"J^+(F) = {coeff_plus_F3:.10g}*F^3 + {coeff_plus_F2:.10g}*F^2 + {coeff_plus_F1:.10g}*F")
        print(f"J^-(F) = {coeff_minus_F3:.10g}*F^3 + {coeff_minus_F2:.10g}*F^2 + {coeff_minus_F1:.10g}*F")
        print()

        # 2) Tanh fits J(d/h_a) using release_50 data at time=1.0
        dist_ref, j_top_ref, j_bot_ref = load_release_data(1.0)
        dist_ref_norm = dist_ref / h_a
        j_top_ref_norm = j_top_ref / (mu * h_a)
        j_bot_ref_norm = j_bot_ref / (mu * h_a)

        (A_plus_d, B_plus_d, C_plus_d), Jplus_of_d = fit_tanh(dist_ref_norm, j_top_ref_norm)
        (A_minus_d, B_minus_d, C_minus_d), Jminus_of_d = fit_tanh(
            dist_ref_norm, j_bot_ref_norm
        )

        print("=" * 70)
        print("TANH FIT FOR J^+(d) and J^-(d) at time=1.0")
        print("=" * 70)
        print(f"J^+(d) = {A_plus_d:.10g}*tanh({B_plus_d:.10g}*d) + {C_plus_d:.10g}")
        print(f"J^-(d) = {A_minus_d:.10g}*tanh({B_minus_d:.10g}*d) + {C_minus_d:.10g}")
        print()

        # Combine: if J depends on both F and d, scale J(d) by J(F) (direct product)
        # Compute normalized force for the current time and for the reference time t=1.0
        F_norm_current = choose_traction("release") * args.time * h_a / (mu * h_a)
        F_norm_ref = choose_traction("release") * 1.0 * h_a / (mu * h_a)
        Jplus_F_val = Jplus_of_F(F_norm_current)
        Jminus_F_val = Jminus_of_F(F_norm_current)
        Jplus_F_ref = Jplus_of_F(F_norm_ref)
        Jminus_F_ref = Jminus_of_F(F_norm_ref)

        # Normalize the force-response scalars relative to the reference (time=1.0)
        if Jplus_F_ref == 0 or not np.isfinite(Jplus_F_ref):
            factor_plus = 1.0
        else:
            factor_plus = Jplus_F_val / Jplus_F_ref

        if Jminus_F_ref == 0 or not np.isfinite(Jminus_F_ref):
            factor_minus = 1.0
        else:
            factor_minus = Jminus_F_val / Jminus_F_ref

        print("=" * 70)
        print(f"SCALING FACTORS at time={args.time}")
        print("=" * 70)
        print(f"F_norm (current) = {F_norm_current:.10g}")
        print(f"F_norm (ref @t=1.0) = {F_norm_ref:.10g}")
        print(f"J^+(F_current) = {Jplus_F_val:.10g}")
        print(f"J^-(F_current) = {Jminus_F_val:.10g}")
        print(f"J^+(F_ref) = {Jplus_F_ref:.10g}")
        print(f"J^-(F_ref) = {Jminus_F_ref:.10g}")
        print(f"Scaling factor for J^+: {factor_plus:.10g}")
        print(f"Scaling factor for J^-: {factor_minus:.10g}")
        print()

        # Create dense grid for plotting fitted curves (using normalized distance)
        x_dense = np.linspace(dist_norm.min(), dist_norm.max(), 300)
        Jplus_d_vals = Jplus_of_d(x_dense)
        Jminus_d_vals = Jminus_of_d(x_dense)

        # Final combined functions — scale the time=1.0 d-curves by the normalized factors
        Jplus_combined = factor_plus * Jplus_d_vals
        Jminus_combined = factor_minus * Jminus_d_vals

        print("=" * 70)
        print("COMBINED EXPRESSIONS FOR PLOTTING")
        print("=" * 70)
        print(f"J^+(F,d) = factor_plus * J^+(d)")
        print(f"         = {factor_plus:.10g} * [{A_plus_d:.10g}*tanh({B_plus_d:.10g}*d) + {C_plus_d:.10g}]")
        print(f"J^-(F,d) = factor_minus * J^-(d)")
        print(f"         = {factor_minus:.10g} * [{A_minus_d:.10g}*tanh({B_minus_d:.10g}*d) + {C_minus_d:.10g}]")
        print()

    fig_size = (4.0, 3.2)
    fig, ax = plt.subplots(figsize=fig_size)
    color_left = "#1b9e77"
    color_right = "#d95f02"

    ax.plot(
        dist_norm,
        j_top / (mu * h_a),
        color=color_left,
        marker="o" if not args.show_fit else "o",
        markersize=5,
        linestyle="None" if args.show_fit else None,
        label=r"Top crack ($\overline{J}^+$)",
    )
    ax.plot(
        dist_norm,
        j_bot / (mu * h_a),
        color=color_right,
        marker="o" if not args.show_fit else "o",
        markersize=5,
        linestyle="None" if args.show_fit else None,
        label=r"Bottom crack ($\overline{J}^-$)",
    )
    ax.plot(
        dist_norm,
        (j_top + j_bot) / (mu * h_a),
        color="#7570b3",
        marker="o",
        markersize=5,
        label="Sum",
    )
    analytical_rate = analytical_energy_release_rate(args.time, mode="release")
    ax.axhline(
        analytical_rate / (mu * h_a),
        color="#4d4d4d",
        linestyle="--",
        label="Analytical",
    )
    # Plot fitted curves if requested
    if args.show_fit:
        ax.plot(
            x_dense, Jplus_combined, color=color_left, linestyle="-", label=r"J^+ fit"
        )
        ax.plot(
            x_dense, Jminus_combined, color=color_right, linestyle="-", label=r"J^- fit"
        )
        # also plot the reference J(d) at time=1.0 (d-only fits) if desired (dashed)
        # ax.plot(
        #     x_dense,
        #     Jplus_d_vals,
        #     color=color_left,
        #     linestyle="--",
        #     label=r"J^+ (d only, t=1)",
        # )
        # ax.plot(
        #     x_dense,
        #     Jminus_d_vals,
        #     color=color_right,
        #     linestyle="--",
        #     label=r"J^- (d only, t=1)",
        # )
    ax.set_xlabel("Distance between crack tips $d$ / $h_a$")
    ax.set_ylabel("Normalized $J$-integral")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if args.ylim_top is not None:
        ax.set_ylim(top=args.ylim_top)
    if args.ylim_bottom is not None:
        ax.set_ylim(bottom=args.ylim_bottom)
    if args.show_legend:
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_path = args.out
    if output_path is None:
        output_path = "release_50_summary_t%.6f.pdf" % args.time
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
