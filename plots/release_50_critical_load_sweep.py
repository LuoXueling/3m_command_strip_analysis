import argparse

import matplotlib.pyplot as plt
import numpy as np

from fit_models import fit_polynomial, fit_tanh
from plots_crack_tips import _find_latest_csv, _load_avg_j
from release_50_utils import load_release_data
from tape import h_a, mu, choose_traction


def _bisection_root(func, left, right, tol=1e-8, max_iter=100):
    f_left = func(left)
    f_right = func(right)
    if not np.isfinite(f_left) or not np.isfinite(f_right):
        return np.nan
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left * f_right > 0:
        return np.nan

    a = float(left)
    b = float(right)
    fa = float(f_left)
    fb = float(f_right)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = func(mid)
        if not np.isfinite(fm):
            return np.nan
        if abs(fm) < tol or abs(b - a) < tol:
            return float(mid)
        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return float(0.5 * (a + b))


def _find_minimum_f(func, f_min, f_max, n_scan=300):
    """Find the smallest F where func(F) >= 0 by scanning and then bisecting."""
    f_min = float(f_min)
    f_max = float(f_max)
    if f_max <= f_min:
        return np.nan

    y0 = func(f_min)
    if np.isfinite(y0) and y0 >= 0.0:
        return f_min

    grid = np.linspace(f_min, f_max, n_scan)
    prev_x = None
    prev_y = None
    for x in grid:
        y = func(float(x))
        if not np.isfinite(y):
            prev_x = None
            prev_y = None
            continue
        if y >= 0.0:
            if prev_x is None:
                return float(x)
            if prev_y is None or prev_y < 0.0:
                return _bisection_root(func, float(prev_x), float(x))
        prev_x = float(x)
        prev_y = float(y)
    return np.nan


def _invert_tanh(target_j, scale, coeffs):
    """Invert J = scale * (A*tanh(B*d) + C) over the full real domain d in (-inf, inf).

    Returns:
      - d when a real crossing exists,
      - NaN when target is unreachable (outside asymptotic tanh range).
    """
    A, B, C = coeffs
    if (
        not np.isfinite(scale)
        or not np.isfinite(A)
        or not np.isfinite(B)
        or not np.isfinite(C)
        or scale <= 0.0
        or abs(A) < 1e-14
        or abs(B) < 1e-14
    ):
        return np.nan

    target = target_j / scale
    u = (target - C) / A
    if not np.isfinite(u):
        return np.nan

    # Reachability for real d requires |u| < 1; allow tiny numeric tolerance.
    tol = 1e-12
    if u <= -1.0 - tol or u >= 1.0 + tol:
        return np.nan

    # Clamp only for numerical safety near asymptotes.
    u = min(max(u, -1.0 + tol), 1.0 - tol)
    d = np.arctanh(u) / B
    if not np.isfinite(d):
        return np.nan
    return float(d)


def _build_models():
    """Fit the J(F) and J(d/h_a) expressions used by the critical load solver."""
    release_csv = _find_latest_csv("Tape_Fracture_release*.csv")
    force, j_top_f, j_bot_f = _load_avg_j(release_csv, "release")

    coeffs_plus_f, jplus_of_f = fit_polynomial(
        force, j_top_f, degree=3, force_through_origin=True
    )
    coeffs_minus_f, jminus_of_f = fit_polynomial(
        force, j_bot_f, degree=3, force_through_origin=True
    )

    dist_ref, j_top_ref, j_bot_ref = load_release_data(1.0)
    dist_ref_norm = dist_ref / h_a
    j_top_ref_norm = j_top_ref / (mu * h_a)
    j_bot_ref_norm = j_bot_ref / (mu * h_a)

    coeffs_plus_d, jplus_of_d = fit_tanh(dist_ref_norm, j_top_ref_norm)
    coeffs_minus_d, jminus_of_d = fit_tanh(dist_ref_norm, j_bot_ref_norm)

    f_ref = choose_traction("release") * 1.0 * h_a / (mu * h_a)
    jplus_f_ref = float(jplus_of_f(f_ref))
    jminus_f_ref = float(jminus_of_f(f_ref))

    return (
        coeffs_plus_f,
        coeffs_minus_f,
        coeffs_plus_d,
        coeffs_minus_d,
        jplus_of_f,
        jminus_of_f,
        jplus_of_d,
        jminus_of_d,
        jplus_f_ref,
        jminus_f_ref,
        f_ref,
    )


def _solve_critical_load(gamma_ba, gamma_as, models, f_max):
    (
        coeffs_plus_f,
        coeffs_minus_f,
        coeffs_plus_d,
        coeffs_minus_d,
        jplus_of_f,
        jminus_of_f,
        jplus_of_d,
        jminus_of_d,
        jplus_f_ref,
        jminus_f_ref,
        f_ref,
    ) = models

    A_p_d, B_p_d, C_p_d = coeffs_plus_d
    A_m_d, B_m_d, C_m_d = coeffs_minus_d

    def scale_plus(F):
        return float(jplus_of_f(F)) / jplus_f_ref if jplus_f_ref != 0 else np.nan

    def scale_minus(F):
        return float(jminus_of_f(F)) / jminus_f_ref if jminus_f_ref != 0 else np.nan

    def jplus_d(d):
        return A_p_d * np.tanh(B_p_d * d) + C_p_d

    def jminus_d(d):
        return A_m_d * np.tanh(B_m_d * d) + C_m_d

    # First threshold: weaker mode at d = 0
    if gamma_as < gamma_ba:

        def first_eq(F):
            return scale_minus(F) * jminus_d(0.0) - gamma_as

        first_mode = "minus"
        first_gamma = gamma_as
    else:

        def first_eq(F):
            return scale_plus(F) * jplus_d(0.0) - gamma_ba

        first_mode = "plus"
        first_gamma = gamma_ba

    f1 = _find_minimum_f(first_eq, 0.0, f_max)

    # Second threshold: Jplus(dminus(F),F) > gamma_ba where Jminus(dminus,F)=gamma_as
    def second_eq(F):
        s_minus = scale_minus(F)
        d_minus = _invert_tanh(gamma_as, s_minus, coeffs_minus_d)
        if not np.isfinite(d_minus):
            return np.nan
        return scale_plus(F) * jplus_d(d_minus) - gamma_ba

    f2 = _find_minimum_f(second_eq, 0.0, f_max)

    # Third threshold: Jminus(dplus(F),F) > gamma_as where Jplus(dplus,F)=gamma_ba
    def third_eq(F):
        s_plus = scale_plus(F)
        d_plus = _invert_tanh(gamma_ba, s_plus, coeffs_plus_d)
        if not np.isfinite(d_plus):
            return np.nan
        return scale_minus(F) * jminus_d(d_plus) - gamma_as

    f3 = _find_minimum_f(third_eq, 0.0, f_max)

    values = np.array([f1, f2, f3], dtype=float)
    # If any of the three forces are NaN, the final critical value is NaN
    if np.all(np.isfinite(values)):
        final = float(np.max(values))
    else:
        final = np.nan
    if np.isfinite(final) and final > 9.0:
        final = np.nan
    return final, f1, f2, f3, first_mode, first_gamma


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Gamma_ba/Gamma_as and plot the normalized critical load heatmap."
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Time used to set the Gamma sweep range (default: 1.0).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=25,
        help="Number of points in each Gamma direction.",
    )
    parser.add_argument(
        "--plot-target",
        choices=["critical", "first", "second", "third"],
        default="critical",
        help=(
            "Plot target: 'critical' = max(F1,F2,F3), "
            "'first' = F1, 'second' = F2, 'third' = F3."
        ),
    )
    parser.add_argument("--out", default=None, help="Output PDF path.")
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    # Use the same sweep range logic as release_50_sweep.py, but in normalized units.
    _, j_top, j_bot = load_release_data(args.time)
    j_top_norm = j_top / (mu * h_a)
    j_bot_norm = j_bot / (mu * h_a)
    all_j = np.concatenate([j_top_norm, j_bot_norm])
    j_min = float(np.min(all_j))
    j_max = float(np.max(all_j))
    j_range = j_max - j_min
    gamma_min = j_min + 0.01 * j_range
    gamma_max = j_max - 0.01 * j_range

    gamma_ba_vals = np.linspace(gamma_min, gamma_max, args.grid_size)
    gamma_as_vals = np.linspace(gamma_min, gamma_max, args.grid_size)

    models = _build_models()

    # Set an upper bound for the search using the force range from the fitted data.
    force, _, _ = _load_avg_j(_find_latest_csv("Tape_Fracture_release*.csv"), "release")
    f_max = float(np.max(force)) * 1.5
    if not np.isfinite(f_max) or f_max <= 0.0:
        f_max = 10.0

    heatmap_values = np.full((args.grid_size, args.grid_size), np.nan, dtype=float)
    for i, gamma_as in enumerate(gamma_as_vals):
        for j, gamma_ba in enumerate(gamma_ba_vals):
            final, f1, f2, f3, first_mode, first_gamma = _solve_critical_load(
                gamma_ba, gamma_as, models, f_max
            )
            if args.plot_target == "critical":
                value = final
            elif args.plot_target == "first":
                value = f1
            elif args.plot_target == "second":
                value = f2
            else:
                value = f3
            if np.isfinite(value) and value > 9.0:
                value = np.nan
            heatmap_values[i, j] = value

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_bad(color="#e7e7e7")
    mesh = ax.pcolormesh(
        gamma_ba_vals,
        gamma_as_vals,
        heatmap_values,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=9.0,
    )
    ax.set_xlabel(r"$\overline{\Gamma}_{ba}$")
    ax.set_ylabel(r"$\overline{\Gamma}_{as}$")
    ax.set_aspect("equal", adjustable="box")
    if args.plot_target == "first":
        cbar_label = r"First-equation load $F_1$"
    elif args.plot_target == "second":
        cbar_label = r"Second-equation load $F_2$"
    elif args.plot_target == "third":
        cbar_label = r"Third-equation load $F_3$"
    else:
        cbar_label = r"Normalized critical load $\overline{F}_r$"
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.tight_layout()

    output_path = args.out
    if output_path is None:
        suffix = args.plot_target
        output_path = "release_50_critical_load_sweep_%s_t%.6f.pdf" % (
            suffix,
            args.time,
        )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
