"""Rebuild comparable Gaussian KDE surfaces and ranked conflict hotspots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "0.6": ROOT / "data" / "ds_vt_ct_csv.CSV",
    "0.8": ROOT / "data" / "ds_vt_ct_0.8_csv.CSV",
    "1.0": ROOT / "data" / "ds_vt_ct_1.0_csv.CSV",
}


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    samples: np.ndarray


def load_events(ttc_ceiling: float) -> pd.DataFrame:
    """Load all headways and retain valid metric event coordinates."""
    frames: list[pd.DataFrame] = []
    required = ["scenario", "minTTC", "ego_pos_x", "ego_pos_y"]
    for tau, path in DATASETS.items():
        frame = pd.read_csv(path, sep=";")
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing {sorted(missing)}")
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required)
        frame = frame.loc[frame["minTTC"] <= ttc_ceiling].copy()
        frame["scenario"] = frame["scenario"].astype(int)
        frame["tau"] = tau
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_grid(events: pd.DataFrame, step_m: float, padding_m: float) -> Grid:
    """Create shared bounds so every configuration remains comparable."""
    x0 = np.floor((events["ego_pos_x"].min() - padding_m) / step_m) * step_m
    x1 = np.ceil((events["ego_pos_x"].max() + padding_m) / step_m) * step_m
    y0 = np.floor((events["ego_pos_y"].min() - padding_m) / step_m) * step_m
    y1 = np.ceil((events["ego_pos_y"].max() + padding_m) / step_m) * step_m
    x = np.arange(x0, x1 + step_m, step_m)
    y = np.arange(y0, y1 + step_m, step_m)
    xx, yy = np.meshgrid(x, y)
    return Grid(x, y, xx, yy, np.column_stack([xx.ravel(), yy.ravel()]))


def estimate_surface(
    events: pd.DataFrame,
    grid: Grid,
    bandwidth_m: float,
    neighbourhood_radius_m: float,
) -> np.ndarray:
    """Return smoothed event counts per standard circular neighbourhood."""
    points = events[["ego_pos_x", "ego_pos_y"]].to_numpy(float)
    if len(points) < 2:
        return np.zeros_like(grid.xx, dtype=float)
    estimator = KernelDensity(kernel="gaussian", bandwidth=bandwidth_m).fit(points)
    density = np.exp(estimator.score_samples(grid.samples)) * len(points)
    neighbourhood_area = np.pi * neighbourhood_radius_m**2
    return (density * neighbourhood_area).reshape(grid.xx.shape)


def rank_peaks(
    surface: np.ndarray,
    grid: Grid,
    count: int,
    separation_m: float,
) -> pd.DataFrame:
    """Select separated local maxima from the mean configuration surface."""
    working = surface.copy()
    peaks: list[dict[str, float | int]] = []
    for rank in range(1, count + 1):
        row, column = np.unravel_index(np.nanargmax(working), working.shape)
        x_value = float(grid.xx[row, column])
        y_value = float(grid.yy[row, column])
        peaks.append(
            {
                "rank": rank,
                "sumo_x_m": x_value,
                "sumo_y_m": y_value,
                "mean_conflicts_per_neighbourhood": float(working[row, column]),
            }
        )
        distance_squared = (grid.xx - x_value) ** 2 + (grid.yy - y_value) ** 2
        working[distance_squared < separation_m**2] = -np.inf
    return pd.DataFrame(peaks)


def save_plot(
    surface: np.ndarray,
    grid: Grid,
    destination: Path,
    title: str,
    colour_max: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    image = axis.imshow(
        surface,
        origin="lower",
        extent=[grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()],
        cmap="inferno",
        vmin=0,
        vmax=colour_max,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set(title=title, xlabel="SUMO x coordinate (m)", ylabel="SUMO y coordinate (m)")
    figure.colorbar(image, ax=axis).set_label("Smoothed conflicts per neighbourhood")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "kde")
    parser.add_argument("--ttc-ceiling", type=float, default=1.0)
    parser.add_argument("--grid-step-m", type=float, default=25.0)
    parser.add_argument("--bandwidth-m", type=float, default=150.0)
    parser.add_argument("--neighbourhood-radius-m", type=float, default=150.0)
    parser.add_argument("--padding-m", type=float, default=300.0)
    parser.add_argument("--peak-count", type=int, default=3)
    parser.add_argument("--minimum-peak-separation-m", type=float, default=750.0)
    parser.add_argument("--panel-scenarios", type=int, nargs="+", default=[2, 6, 10])
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.ttc_ceiling)
    grid = make_grid(events, args.grid_step_m, args.padding_m)

    all_surfaces: list[np.ndarray] = []
    panels: list[tuple[int, str, int, np.ndarray]] = []
    for (scenario, tau), group in events.groupby(["scenario", "tau"], sort=True):
        surface = estimate_surface(
            group, grid, args.bandwidth_m, args.neighbourhood_radius_m
        )
        all_surfaces.append(surface)
        if int(scenario) in args.panel_scenarios:
            panels.append((int(scenario), str(tau), len(group), surface))

    shared_max = max(float(surface.max()) for _, _, _, surface in panels)
    for scenario, tau, count, surface in panels:
        save_plot(
            surface,
            grid,
            args.output_dir / f"kde_s{scenario}_tau_{tau.replace('.', 'p')}.png",
            f"S{scenario} | headway {tau} s | n={count:,}",
            shared_max,
        )

    mean_surface = np.mean(all_surfaces, axis=0)
    save_plot(
        mean_surface,
        grid,
        args.output_dir / "kde_mean_all_configurations.png",
        "Mean KDE across all scenario-headway configurations",
        float(mean_surface.max()),
    )
    rank_peaks(
        mean_surface, grid, args.peak_count, args.minimum_peak_separation_m
    ).to_csv(args.output_dir / "ranked_hotspots_metric.csv", index=False)

    pd.DataFrame(
        [{
            "ttc_ceiling_s": args.ttc_ceiling,
            "grid_step_m": args.grid_step_m,
            "bandwidth_m": args.bandwidth_m,
            "neighbourhood_radius_m": args.neighbourhood_radius_m,
            "minimum_peak_separation_m": args.minimum_peak_separation_m,
            "configuration_count": len(all_surfaces),
            "event_count": len(events),
            "shared_panel_colour_max": shared_max,
        }]
    ).to_csv(args.output_dir / "kde_metadata.csv", index=False)
    print(f"Wrote KDE outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

