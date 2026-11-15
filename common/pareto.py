"""Pareto-front utilities (any number of objectives, each 'min' or 'max').

    mask = pareto_mask(df, {"mAP50_95": "max", "lat_p50_ms": "min"})
    plot_pareto(df, x="lat_p50_ms", y="mAP50_95", x_dir="min", y_dir="max", label="model")
"""
from __future__ import annotations

import numpy as np


def _to_minimisation(values: np.ndarray, directions: list[str]) -> np.ndarray:
    v = values.astype(float).copy()
    for j, d in enumerate(directions):
        if d not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {d}")
        if d == "max":
            v[:, j] = -v[:, j]
    return v


def pareto_mask_array(values: np.ndarray, directions: list[str]) -> np.ndarray:
    """True for non-dominated rows. O(n^2), fine for the few hundred configs of a course."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array")
    if values.shape[1] != len(directions):
        raise ValueError("directions must contain one entry per objective")

    # Convert all objectives to minimisation so that one comparison handles
    # both 'min' and 'max' directions.
    values = _to_minimisation(values, directions)
    keep = np.ones(len(values), dtype=bool)

    for i, candidate in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            # Identical rows are not strictly better than one another, so
            # exact duplicates both remain on the Pareto front.
            if np.all(other <= candidate) and np.any(other < candidate):
                keep[i] = False
                break

    return keep


def pareto_mask(df, objectives: dict[str, str]) -> np.ndarray:
    cols = list(objectives)
    sub = df[cols].to_numpy(dtype=float)
    ok = ~np.isnan(sub).any(axis=1)
    mask = np.zeros(len(df), dtype=bool)
    mask[ok] = pareto_mask_array(sub[ok], [objectives[c] for c in cols])
    return mask


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float], directions=("min", "max")) -> float:
    """Area dominated by a 2-D front w.r.t. a reference point (a single scalar to compare fronts,
    e.g. 'image-level detector + smoothing' vs 'video model')."""
    p = _to_minimisation(np.asarray(points, float), list(directions))
    r = _to_minimisation(np.asarray([ref], float), list(directions))[0]
    p = p[np.all(p <= r, axis=1)]
    if len(p) == 0:
        return 0.0
    p = p[pareto_mask_array(p, ["min", "min"])]
    p = p[np.argsort(p[:, 0])]
    hv, prev_y = 0.0, r[1]
    for x, y in p:
        hv += (r[0] - x) * (prev_y - y)
        prev_y = y
    return float(hv)


def plot_pareto(df, x: str, y: str, x_dir: str = "min", y_dir: str = "max", label: str | None = None,
                hue: str | None = None, ax=None, logx: bool = False, title: str | None = None,
                annotate: bool = True, budget_x: float | None = None):
    """Scatter all configurations, highlight + connect the Pareto front.
    budget_x draws a vertical constraint line (e.g. 33 ms = 30 fps, or RTF = 1)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    mask = pareto_mask(df, {x: x_dir, y: y_dir})
    groups = [(None, df)] if hue is None else list(df.groupby(hue))
    for name, g in groups:
        ax.scatter(g[x], g[y], s=40, alpha=0.6, label=name)
    front = df[mask].sort_values(x)
    ax.plot(front[x], front[y], "k--", lw=1.2, drawstyle="steps-post" if x_dir == "min" else "default")
    ax.scatter(front[x], front[y], s=110, facecolors="none", edgecolors="k", lw=1.5, label="Pareto front")
    if annotate and label is not None:
        for _, r in df.iterrows():
            ax.annotate(str(r[label]), (r[x], r[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if budget_x is not None:
        ax.axvline(budget_x, color="r", ls=":", lw=1.2, label=f"budget {x}={budget_x}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(f"{x} ({'lower' if x_dir == 'min' else 'higher'} is better)")
    ax.set_ylabel(f"{y} ({'lower' if y_dir == 'min' else 'higher'} is better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax, mask
