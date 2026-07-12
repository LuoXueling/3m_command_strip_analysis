import argparse
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from release_50_utils import (
    build_natural_cubic_spline,
    find_crossing,
    load_release_data,
)
from tape import choose_traction, h_a, mu


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
            return np.nan, np.nan

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


_TIME_VALS = None
_GC2_VALS = None
_SPLINE_TOP_LIST = None
_SPLINE_BOT_LIST = None
_MIN_J_TOP_LIST = None
_MIN_J_BOT_LIST = None
_DIST_LIST = None
_GC1 = None
_MAX_ITER = None
_TARGET_DELTA = None


def _init_worker(
    time_vals,
    gc2_vals,
    spline_top_list,
    spline_bot_list,
    min_j_top_list,
    min_j_bot_list,
    dist_list,
    gc1,
    max_iter,
    target_delta,
):
    global _TIME_VALS, _GC2_VALS, _SPLINE_TOP_LIST, _SPLINE_BOT_LIST
    global _MIN_J_TOP_LIST, _MIN_J_BOT_LIST, _DIST_LIST, _GC1, _MAX_ITER, _TARGET_DELTA
    _TIME_VALS = time_vals
    _GC2_VALS = gc2_vals
    _SPLINE_TOP_LIST = spline_top_list
    _SPLINE_BOT_LIST = spline_bot_list
    _MIN_J_TOP_LIST = min_j_top_list
    _MIN_J_BOT_LIST = min_j_bot_list
    _DIST_LIST = dist_list
    _GC1 = gc1
    _MAX_ITER = max_iter
    _TARGET_DELTA = target_delta


def _simulate_pair(idx_pair):
    t_idx, g_idx = idx_pair
    return _simulate_iterations(
        _DIST_LIST[t_idx],
        _SPLINE_TOP_LIST[t_idx],
        _SPLINE_BOT_LIST[t_idx],
        _GC1,
        _GC2_VALS[g_idx],
        _MAX_ITER,
        _TARGET_DELTA,
        _MIN_J_TOP_LIST[t_idx],
        _MIN_J_BOT_LIST[t_idx],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Gc2 and time for crack propagation iterations."
    )
    parser.add_argument(
        "--gc1", type=float, required=True, help="Top crack critical value."
    )
    parser.add_argument("--grid-size-gc2", type=int, default=25)
    parser.add_argument("--grid-size-time", type=int, default=25)
    parser.add_argument("--time-min", type=float, default=0.6)
    parser.add_argument("--time-max", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    time_vals = np.linspace(args.time_min, args.time_max, args.grid_size_time)
    traction = choose_traction("release")
    force_vals = time_vals * traction * h_a

    dist, j_top, j_bot = load_release_data(1.0)
    all_j = np.concatenate([j_top, j_bot])
    j_min = float(np.min(all_j))
    j_max = float(np.max(all_j))
    j_range = j_max - j_min
    gc_min = j_min + 0.01 * j_range
    gc_max = j_max - 0.01 * j_range
    gc2_max = min(gc_max, 0.35)
    gc2_vals = np.linspace(gc_min, gc2_max, args.grid_size_gc2)

    iterations = np.full((args.grid_size_gc2, args.grid_size_time), np.nan, dtype=float)
    dist_mid = np.full((args.grid_size_gc2, args.grid_size_time), np.nan, dtype=float)
    target_delta = 100.0

    spline_top_list = []
    spline_bot_list = []
    min_j_top_list = []
    min_j_bot_list = []
    dist_list = []
    for time_val in time_vals:
        dist, j_top, j_bot = load_release_data(time_val)
        dist_list.append(dist)
        spline_top_list.append(build_natural_cubic_spline(dist, j_top))
        spline_bot_list.append(build_natural_cubic_spline(dist, j_bot))
        min_j_top_list.append(float(np.min(j_top)))
        min_j_bot_list.append(float(np.min(j_bot)))

    idx_pairs = [
        (t_idx, g_idx)
        for t_idx in range(args.grid_size_time)
        for g_idx in range(args.grid_size_gc2)
    ]
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            time_vals,
            gc2_vals,
            spline_top_list,
            spline_bot_list,
            min_j_top_list,
            min_j_bot_list,
            dist_list,
            args.gc1,
            args.max_iter,
            target_delta,
        ),
    ) as pool:
        results = pool.map(_simulate_pair, idx_pairs)

    results_arr = np.array(results, dtype=float).reshape(
        args.grid_size_time, args.grid_size_gc2, 2
    )
    iterations = results_arr[:, :, 0].T
    dist_mid = results_arr[:, :, 1].T

    dist_mid = np.where(np.isnan(iterations), np.nan, dist_mid)

    fig_size = (8.3, 3.2)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=fig_size)

    mesh = ax_left.pcolormesh(
        force_vals / (h_a * mu),
        gc2_vals / (h_a * mu),
        iterations,
        shading="auto",
        cmap="summer",
    )
    ax_left.set_xlabel(r"Normalized force $\overline{F}$")
    ax_left.set_ylabel(r"$\overline{\Gamma}_{as}$")
    ax_left.set_ylim(gc2_vals[0] / (h_a * mu), gc2_vals[-1] / (h_a * mu))
    fig.colorbar(mesh, ax=ax_left, label="Iterations to reach $100 h_a$")

    mesh2 = ax_right.pcolormesh(
        force_vals / (h_a * mu),
        gc2_vals / (h_a * mu),
        dist_mid,
        shading="auto",
        cmap="Blues",
    )
    ax_right.set_xlabel(r"Normalized force $\overline{F}$")
    ax_right.set_ylabel(r"$\overline{\Gamma}_{as}$")
    ax_right.set_ylim(gc2_vals[0] / (h_a * mu), gc2_vals[-1] / (h_a * mu))
    fig.colorbar(mesh2, ax=ax_right, label="Distance between crack tips $d$ / $h_a$")

    fig.tight_layout()
    output_path = args.out
    if output_path is None:
        output_path = "release_50_sweep_time_gc1_%.6f.pdf" % args.gc1
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
