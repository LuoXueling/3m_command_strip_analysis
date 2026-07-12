import glob
import os
import re

import numpy as np
import pandas as pd

from tape import h_a


def contour_columns(columns, prefix, start, end):
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


def parse_bottom_length(filename):
    match = re.search(r"release_50_(\d{2,3})\.csv$", filename)
    if not match:
        return None
    raw = match.group(1)
    if len(raw) == 3:
        return int(raw) / 10.0
    return float(raw)


def interp_at_time(time_vals, series_vals, target_time):
    time_vals = np.asarray(time_vals, dtype=float)
    series_vals = np.asarray(series_vals, dtype=float)
    if target_time < time_vals.min() or target_time > time_vals.max():
        raise ValueError("Requested time %.6f is outside data range." % target_time)
    return np.interp(target_time, time_vals, series_vals)


def prepare_xy(dist, values):
    order = np.argsort(dist)
    dist_sorted = np.asarray(dist)[order]
    values_sorted = np.asarray(values)[order]

    unique_dist = []
    unique_vals = []
    start = 0
    while start < len(dist_sorted):
        end = start + 1
        while end < len(dist_sorted) and dist_sorted[end] == dist_sorted[start]:
            end += 1
        unique_dist.append(dist_sorted[start])
        unique_vals.append(values_sorted[start:end].mean())
        start = end

    return np.array(unique_dist, dtype=float), np.array(unique_vals, dtype=float)


def build_natural_cubic_spline(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError("Need at least 3 points for cubic spline")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing for spline")

    h = np.diff(x)
    a = h[:-1]
    b = 2.0 * (h[:-1] + h[1:])
    c = h[1:]
    d = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    cp = np.zeros_like(c)
    dp = np.zeros_like(d)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, len(d)):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom if i < len(c) else 0.0
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    m = np.zeros_like(x)
    m[-2] = dp[-1]
    for i in range(len(d) - 2, -1, -1):
        m[i + 1] = dp[i] - cp[i] * m[i + 2]

    def eval_spline(xq):
        xq = np.asarray(xq, dtype=float)
        idx = np.searchsorted(x, xq, side="right") - 1
        idx = np.clip(idx, 0, len(x) - 2)
        x0 = x[idx]
        x1 = x[idx + 1]
        hq = x1 - x0
        a0 = (x1 - xq) / hq
        b0 = (xq - x0) / hq
        return (
            a0 * y[idx]
            + b0 * y[idx + 1]
            + ((a0**3 - a0) * m[idx] + (b0**3 - b0) * m[idx + 1]) * (hq**2) / 6.0
        )

    return eval_spline


def bisect_root(func, x0, x1, tol=1e-8, max_iter=60):
    f0 = func(x0)
    f1 = func(x1)
    if f0 == 0.0:
        return x0
    if f1 == 0.0:
        return x1
    if f0 * f1 > 0.0:
        raise ValueError("Root is not bracketed")

    lo, hi = x0, x1
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if abs(fm) < tol:
            return mid
        if f0 * fm <= 0.0:
            hi = mid
            f1 = fm
        else:
            lo = mid
            f0 = fm
    return 0.5 * (lo + hi)


def find_crossing(spline, x0, gc, direction, x_min, x_max):
    j0 = float(spline(x0))
    if j0 <= gc:
        return x0, 0.0, False

    end = x_max if direction > 0 else x_min
    grid = np.linspace(x0, end, 200)
    fvals = spline(grid) - gc

    if direction < 0:
        grid = grid[::-1]
        fvals = fvals[::-1]

    if np.min(fvals) > 0.0:
        raise RuntimeError("Gc too low; unstable propagation.")

    for i in range(len(grid) - 1):
        f0 = fvals[i]
        f1 = fvals[i + 1]
        if f0 == 0.0:
            return grid[i], 0.0, True
        if f0 * f1 < 0.0:
            root = bisect_root(lambda x: float(spline(x)) - gc, grid[i], grid[i + 1])
            delta = root - x0 if direction > 0 else x0 - root
            return root, delta, True

    return x0, 0.0, False


def load_release_data(target_time):
    files = glob.glob("release_50_*.csv")
    if not files:
        raise FileNotFoundError("No files found matching release_50_*.csv")

    records = []
    for path in files:
        bottom_len = parse_bottom_length(os.path.basename(path))
        if bottom_len is None:
            continue

        df = pd.read_csv(path)
        bot_cols = contour_columns(df.columns, "J at HOUT_J-BOT", 21, 30)
        top_cols = contour_columns(df.columns, "J at HOUT_J-TOP", 21, 30)
        if len(bot_cols) != 10 or len(top_cols) != 10:
            raise ValueError(
                "Expected 10 contours (21-30) for both BOT and TOP in %s" % path
            )

        if "Time" not in df.columns:
            raise KeyError("Missing Time column in %s" % path)
        time_vals = df["Time"].astype(float).to_numpy()
        j_bot_series = df[bot_cols].mean(axis=1).to_numpy()
        j_top_series = df[top_cols].mean(axis=1).to_numpy()
        j_bot_avg = interp_at_time(time_vals, j_bot_series, target_time)
        j_top_avg = interp_at_time(time_vals, j_top_series, target_time)
        distance_norm = bottom_len - 50

        records.append((distance_norm, j_top_avg, j_bot_avg))

    if not records:
        raise ValueError("No valid release_50_*.csv records found")

    records.sort(key=lambda r: r[0])
    dist = np.array([r[0] for r in records], dtype=float)
    j_top = np.array([r[1] for r in records], dtype=float)
    j_bot = np.array([r[2] for r in records], dtype=float)

    dist, j_top = prepare_xy(dist, j_top)
    _, j_bot = prepare_xy(dist, j_bot)

    return dist, j_top, j_bot
