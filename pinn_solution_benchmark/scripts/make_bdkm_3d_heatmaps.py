#!/usr/bin/env python
"""Reproduce 3D intensity surfaces and 2D heat maps for verified BDKm waves.

The numerical values are read from the archived branch catalogue; no physical
coefficient or travelling-wave parameter is retyped in this program.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from matplotlib import cm  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data" / "verified_branches" / "branch_catalog.json"
OUTPUT = ROOT / "reports" / "figures" / "summary" / "bdkm_3d_heatmaps.png"
BRANCHES = (
    ("n1_m0_f0", r"Bright $J_{1,0}=\mathrm{sech}(\xi)$"),
    ("n1_m1_f0", r"Dark $J_{1,1}=\tanh(\xi)$"),
)


def as_float(value: str | int | float) -> float:
    """Evaluate the catalogue's exact rational/square-root strings."""
    return float(sp.N(sp.sympify(value), 17))


def load_branch(branch_id: str) -> dict:
    with CATALOGUE.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return next(record for record in records if record["branch_id"] == branch_id)


def intensity(record: dict, z_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Return |A|^2 for A=a*sinh(xi)^m/cosh(xi)^n."""
    params = record["representative_sets"][0]
    n = as_float(record["n"])
    m = as_float(record["m"])
    amplitude = as_float(params["a"])
    lambda_1 = as_float(params["lambda1"])
    lambda_2 = as_float(params["lambda2"])
    xi = lambda_1 * z_grid + lambda_2 * t_grid
    profile = amplitude * np.sinh(xi) ** m / np.cosh(xi) ** n
    return np.abs(profile) ** 2


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "savefig.dpi": 300,
    })
    fig = plt.figure(figsize=(12.4, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1.0))

    for column, (branch_id, title) in enumerate(BRANCHES):
        record = load_branch(branch_id)
        params = record["representative_sets"][0]
        lambda_1 = as_float(params["lambda1"])
        lambda_2 = as_float(params["lambda2"])
        xi_min, xi_max = (as_float(x) for x in record["xi_domain"])
        z = np.linspace(0.0, 1.0, 161)
        # Convert the archived xi window to a t window at z=0.
        t = np.linspace(xi_min / lambda_2, xi_max / lambda_2, 321)
        T, Z = np.meshgrid(t, z)
        field = intensity(record, Z, T)

        ax_surface = fig.add_subplot(grid[0, column], projection="3d")
        surface = ax_surface.plot_surface(
            T, Z, field, cmap=cm.viridis, linewidth=0, antialiased=True,
            rcount=121, ccount=241,
        )
        ax_surface.set_xlabel(r"retarded time $t$")
        ax_surface.set_ylabel(r"distance $z$")
        ax_surface.set_zlabel(r"intensity $|A|^2$")
        ax_surface.set_title(f"{title}: 3D surface")
        ax_surface.view_init(elev=28, azim=-132)
        fig.colorbar(surface, ax=ax_surface, shrink=0.62, pad=0.09)

        ax_heat = fig.add_subplot(grid[1, column])
        heat = ax_heat.pcolormesh(T, Z, field, shading="auto", cmap="viridis")
        ax_heat.set_xlabel(r"retarded time $t$")
        ax_heat.set_ylabel(r"distance $z$")
        ax_heat.set_title(f"{title}: heat map")
        fig.colorbar(heat, ax=ax_heat, label=r"$|A|^2$")

        ax_heat.text(
            0.02, 0.96,
            rf"$\lambda_1={lambda_1:g},\ \lambda_2={lambda_2:g}$",
            transform=ax_heat.transAxes, va="top", ha="left",
            color="white", fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "Verified BDKm travelling-wave examples from the archived exact catalogue",
        fontsize=12,
    )
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
