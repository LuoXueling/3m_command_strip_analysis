import argparse
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from release_50_utils import (
    build_natural_cubic_spline,
    find_crossing,
    load_release_data,
)
from tape import h_a, mu

_DIST = None
_SPLINE_TOP = None
_SPLINE_BOT = None
_MAX_ITER = None
_TARGET_DELTA = None
_MIN_J_TOP = None
_MIN_J_BOT = None


def _init_worker(
    dist, spline_top, spline_bot, max_iter, target_delta, min_j_top, min_j_bot
):
    global _DIST, _SPLINE_TOP, _SPLINE_BOT, _MAX_ITER, _TARGET_DELTA, _MIN_J_TOP, _MIN_J_BOT
    _DIST = dist
    _SPLINE_TOP = spline_top
    _SPLINE_BOT = spline_bot
    _MAX_ITER = max_iter
    _TARGET_DELTA = target_delta
    _MIN_J_TOP = min_j_top
    _MIN_J_BOT = min_j_bot


def _simulate_iterations(
    dist,
    spline_top,
    spline_bot,
    gc_top,
    gc_bot,
    max_iter,
    target_delta,
    min_j_top,
    min_j_bot,
):
    weak_is_top = gc_top <= gc_bot
    dir_top = -1
    dir_bot = 1
    dist_current = 0.0

    cum_top = 0.0
    cum_bot = 0.0
    dist_after_weak_iter2 = np.nan

    for iteration in range(1, max_iter + 1):
        j_top_now = float(spline_top(dist_current))
        j_bot_now = float(spline_bot(dist_current))
        if j_top_now <= gc_top and j_bot_now <= gc_bot:
            return np.nan, dist_after_weak_iter2

        moved = False

        try:
            if weak_is_top:
                new_dist, delta, did = find_crossing(
                    spline_top, dist_current, gc_top, dir_top, dist[0], dist[-1]
                )
                if did and delta > 0.0:
                    cum_top += delta
                    dist_current = new_dist
                    moved = True
            else:
                new_dist, delta, did = find_crossing(
                    spline_bot, dist_current, gc_bot, dir_bot, dist[0], dist[-1]
                )
                if did and delta > 0.0:
                    cum_bot += delta
                    dist_current = new_dist
                    moved = True
        except RuntimeError:
            if (weak_is_top and gc_top < min_j_top) or (
                not weak_is_top and gc_bot < min_j_bot
            ):
                return 0.0, np.nan
            return np.nan, np.nan

        if iteration == 2:
            dist_after_weak_iter2 = dist_current

        if cum_top >= target_delta or cum_bot >= target_delta:
            return float(iteration), dist_after_weak_iter2

        try:
            if not weak_is_top:
                new_dist, delta, did = find_crossing(
                    spline_top, dist_current, gc_top, dir_top, dist[0], dist[-1]
                )
                if did and delta > 0.0:
                    cum_top += delta
                    dist_current = new_dist
                    moved = True
            else:
                new_dist, delta, did = find_crossing(
                    spline_bot, dist_current, gc_bot, dir_bot, dist[0], dist[-1]
                )
                if did and delta > 0.0:
                    cum_bot += delta
                    dist_current = new_dist
                    moved = True
        except RuntimeError:
            if (not weak_is_top and gc_top < min_j_top) or (
                weak_is_top and gc_bot < min_j_bot
            ):
                return 0.0, np.nan
            return np.nan, np.nan

        if cum_top >= target_delta or cum_bot >= target_delta:
            return float(iteration), dist_after_weak_iter2

        if not moved:
            return np.nan, dist_after_weak_iter2

    return np.nan, dist_after_weak_iter2


def _simulate_pair(gc_pair):
    gc_top, gc_bot = gc_pair
    return _simulate_iterations(
        _DIST,
        _SPLINE_TOP,
        _SPLINE_BOT,
        gc_top,
        gc_bot,
        _MAX_ITER,
        _TARGET_DELTA,
        _MIN_J_TOP,
        _MIN_J_BOT,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Grid sweep of Gc1/Gc2 for crack propagation iterations."
    )
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    dist, j_top, j_bot = load_release_data(args.time)
    spline_top = build_natural_cubic_spline(dist, j_top)
    spline_bot = build_natural_cubic_spline(dist, j_bot)
    min_j_top = float(np.min(j_top))
    min_j_bot = float(np.min(j_bot))

    all_j = np.concatenate([j_top, j_bot])
    j_min = float(np.min(all_j))
    j_max = float(np.max(all_j))
    j_range = j_max - j_min
    gc_min = j_min + 0.01 * j_range
    gc_max = j_max - 0.01 * j_range
    gc_top_vals = np.linspace(gc_min, gc_max, args.grid_size)
    gc_bot_vals = np.linspace(gc_min, gc_max, args.grid_size)
    print("Gc1 (top) values:", gc_top_vals)
    print("Gc2 (bottom) values:", gc_bot_vals)

    iterations = np.full((args.grid_size, args.grid_size), np.nan, dtype=float)
    dist_mid = np.full((args.grid_size, args.grid_size), np.nan, dtype=float)
    target_delta = 100.0

    gc_pairs = [(gc_top, gc_bot) for gc_bot in gc_bot_vals for gc_top in gc_top_vals]
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            dist,
            spline_top,
            spline_bot,
            args.max_iter,
            target_delta,
            min_j_top,
            min_j_bot,
        ),
    ) as pool:
        results = pool.map(_simulate_pair, gc_pairs)
    results_arr = np.array(results, dtype=float).reshape(
        args.grid_size, args.grid_size, 2
    )
    iterations = results_arr[:, :, 0]
    dist_mid = results_arr[:, :, 1]
    dist_mid = np.where(np.isnan(iterations), np.nan, dist_mid)

    fig_size = (8.3, 3.2)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=fig_size)
    mesh = ax_left.pcolormesh(
        gc_top_vals / (h_a * mu),
        gc_bot_vals / (h_a * mu),
        iterations,
        shading="auto",
        cmap="summer",
    )
    ax_left.set_xlabel(r"$\overline{\Gamma}_{ba}$")
    ax_left.set_ylabel(r"$\overline{\Gamma}_{as}$")
    ax_left.set_aspect("equal", adjustable="box")
    fig.colorbar(
        mesh,
        ax=ax_left,
        label="Iterations to reach $100 h_a$ (total failure)",
    )

    mesh2 = ax_right.pcolormesh(
        gc_top_vals / (h_a * mu),
        gc_bot_vals / (h_a * mu),
        dist_mid,
        shading="auto",
        cmap="RdBu",
    )
    ax_right.set_xlabel(r"$\overline{\Gamma}_{ba}$")
    ax_right.set_ylabel(r"$\overline{\Gamma}_{as}$")
    ax_right.set_aspect("equal", adjustable="box")
    fig.colorbar(
        mesh2,
        ax=ax_right,
        label="Distance between crack tips $d$ / $h_a$",
    )

    fig.tight_layout()
    output_path = args.out
    if output_path is None:
        output_path = "release_50_sweep_t%.6f.pdf" % args.time
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
