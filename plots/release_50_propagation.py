import argparse
import matplotlib.pyplot as plt
import numpy as np

from release_50_utils import (
    build_natural_cubic_spline,
    find_crossing,
    load_release_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate crack propagation from release_50_*.csv data."
    )
    parser.add_argument("--jc-top", type=float, required=True)
    parser.add_argument("--jc-bot", type=float, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--start-distance", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--ylim-top", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    dist, j_top, j_bot = load_release_data(args.time)
    if args.start_distance < dist[0] or args.start_distance > dist[-1]:
        raise ValueError(
            "start-distance must be within [%.3f, %.3f]" % (dist[0], dist[-1])
        )

    spline_top = build_natural_cubic_spline(dist, j_top)
    spline_bot = build_natural_cubic_spline(dist, j_bot)

    weak_is_top = args.jc_top <= args.jc_bot
    dir_top = -1
    dir_bot = 1
    dist_current = float(args.start_distance)

    iterations = []
    cum_top = 0.0
    cum_bot = 0.0
    cum_top_list = []
    cum_bot_list = []
    per_iter_total = []
    per_iter_top = []
    per_iter_bot = []

    tol_delta = 1e-6
    for iteration in range(1, args.max_iter + 1):
        prev_dist = dist_current
        j_top_now = float(spline_top(dist_current))
        j_bot_now = float(spline_bot(dist_current))
        if j_top_now <= args.jc_top and j_bot_now <= args.jc_bot:
            print(
                "No propagation: J_top=%.6f, J_bot=%.6f at iter %d"
                % (j_top_now, j_bot_now, iteration)
            )
            break

        delta_top = 0.0
        delta_bot = 0.0
        moved = False

        if weak_is_top:
            new_dist, delta, did = find_crossing(
                spline_top, dist_current, args.jc_top, dir_top, dist[0], dist[-1]
            )
            if did and delta > 0.0:
                delta_top += delta
                cum_top += delta
                dist_current = new_dist
                moved = True
        else:
            new_dist, delta, did = find_crossing(
                spline_bot, dist_current, args.jc_bot, dir_bot, dist[0], dist[-1]
            )
            if did and delta > 0.0:
                delta_bot += delta
                cum_bot += delta
                dist_current = new_dist
                moved = True

        j_top_mid = float(spline_top(dist_current))
        j_bot_mid = float(spline_bot(dist_current))
        print(
            "Iter %d (after weaker): dist=%.4f, J_top=%.6f, J_bot=%.6f"
            % (iteration, dist_current, j_top_mid, j_bot_mid)
        )

        if not weak_is_top:
            new_dist, delta, did = find_crossing(
                spline_top, dist_current, args.jc_top, dir_top, dist[0], dist[-1]
            )
            if did and delta > 0.0:
                delta_top += delta
                cum_top += delta
                dist_current = new_dist
                moved = True
        else:
            new_dist, delta, did = find_crossing(
                spline_bot, dist_current, args.jc_bot, dir_bot, dist[0], dist[-1]
            )
            if did and delta > 0.0:
                delta_bot += delta
                cum_bot += delta
                dist_current = new_dist
                moved = True

        j_top_now = float(spline_top(dist_current))
        j_bot_now = float(spline_bot(dist_current))
        if not moved and j_top_now <= args.jc_top and j_bot_now <= args.jc_bot:
            print(
                "No propagation: J_top=%.6f, J_bot=%.6f at iter %d"
                % (j_top_now, j_bot_now, iteration)
            )
            break

        iterations.append(iteration)
        cum_top_list.append(cum_top)
        cum_bot_list.append(cum_bot)
        per_iter_total.append(delta_top + delta_bot)
        per_iter_top.append(delta_top)
        per_iter_bot.append(delta_bot)

        print(
            "Iter %d (after stronger): dist=%.4f, J_top=%.6f, J_bot=%.6f, "
            "delta_top=%.6f, delta_bot=%.6f"
            % (iteration, dist_current, j_top_now, j_bot_now, delta_top, delta_bot)
        )

        if not moved:
            break

        if (
            abs(delta_top) < tol_delta
            and abs(delta_bot) < tol_delta
            and j_top_now <= args.jc_top
            and j_bot_now <= args.jc_bot
        ):
            break

    if not iterations:
        print("No propagation iterations executed.")
        return

    x = np.arange(1, args.max_iter + 1)
    width = 0.35

    fig_size = (4.0, 3.2)
    fig, ax = plt.subplots(figsize=fig_size)
    color_top = "#1b9e77"
    color_bot = "#d95f02"

    top_pad = cum_top_list + [cum_top_list[-1]] * (args.max_iter - len(cum_top_list))
    bot_pad = cum_bot_list + [cum_bot_list[-1]] * (args.max_iter - len(cum_bot_list))
    ax.bar(
        x - width / 2.0,
        top_pad,
        width,
        color=color_top,
        label=r"Top crack ($\mathcal{C}^+$)",
    )
    ax.bar(
        x + width / 2.0,
        bot_pad,
        width,
        color=color_bot,
        label=r"Bottom crack ($\mathcal{C}^-$)",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Crack length / $h_a$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    if args.ylim_top is not None:
        ax.set_ylim(top=args.ylim_top)

    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    output_path = args.out
    if output_path is None:
        output_path = "release_50_propagation_t%.6f_gc1%.6f_gc2%.6f.pdf" % (
            args.time,
            args.jc_top,
            args.jc_bot,
        )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
