"""Generate figures from frozen or density-runner benchmark JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_INPUT = Path("outputs/shield_density_benchmark_seed42.json")
DEFAULT_OUTPUT_DIR = Path("docs/assets")
REQUIRED_DENSITIES = (3, 5, 10)
METHODS = ("robust", "shielded")
RUNNER_METHODS = {
    "robust": "social_spacetime_robust",
    "shielded": "social_spacetime_shielded",
}
METRICS = (
    "success_rate",
    "collision_rate",
    "mean_spl",
    "mean_minimum_human_distance",
    "mean_social_violation_rate",
)

METHOD_LABELS = {
    "robust": "Robust",
    "shielded": "Shielded",
}
METHOD_STYLES = {
    "robust": {"color": "#3B6EA8", "linestyle": "--", "marker": "o"},
    "shielded": {"color": "#D97706", "linestyle": "-", "marker": "s"},
}


class BenchmarkFormatError(ValueError):
    """Raised when a benchmark artifact lacks required plotting data."""


@dataclass(frozen=True)
class BenchmarkData:
    """The density series needed by the public benchmark figures."""

    densities: tuple[int, ...]
    series: dict[str, dict[str, tuple[float, ...]]]


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BenchmarkFormatError(f"{context} must be a JSON object")
    return value


def _require_metric(
    summary: dict[str, object],
    metric: str,
    *,
    method: str,
    density: int,
) -> float:
    if metric not in summary:
        raise BenchmarkFormatError(
            f"{method} density {density} summary is missing required field "
            f"{metric!r}"
        )
    value = summary[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkFormatError(
            f"{method} density {density} field {metric!r} must be numeric"
        )
    resolved = float(value)
    if not isfinite(resolved):
        raise BenchmarkFormatError(
            f"{method} density {density} field {metric!r} must be finite"
        )
    if metric != "mean_minimum_human_distance" and not 0.0 <= resolved <= 1.0:
        raise BenchmarkFormatError(
            f"{method} density {density} field {metric!r} must be in [0, 1]"
        )
    if metric == "mean_minimum_human_distance" and resolved < 0.0:
        raise BenchmarkFormatError(
            f"{method} density {density} field {metric!r} must be non-negative"
        )
    return resolved


def _resolve_method_rows(
    root: dict[str, object],
    method: str,
) -> dict[str, object]:
    """Resolve one plot series from frozen or density-runner JSON."""
    if method in root:
        return _require_mapping(root[method], f"{method} results")

    if "per_density" not in root:
        raise BenchmarkFormatError(
            f"benchmark root is missing required method {method!r}"
        )

    per_density = _require_mapping(
        root["per_density"],
        "benchmark per_density",
    )
    runner_method = RUNNER_METHODS[method]
    method_rows: dict[str, object] = {}
    for density_key, density_value in per_density.items():
        density_row = _require_mapping(
            density_value,
            f"density {density_key}",
        )
        if "per_method" not in density_row:
            continue
        per_method = _require_mapping(
            density_row["per_method"],
            f"density {density_key} per_method",
        )
        if runner_method in per_method:
            method_rows[density_key] = per_method[runner_method]

    if not method_rows:
        raise BenchmarkFormatError(
            "density-runner results are missing required method "
            f"{runner_method!r}"
        )
    return method_rows


def load_benchmark_data(input_path: Path) -> BenchmarkData:
    """Load and validate Robust/Shielded series from a benchmark artifact."""
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkFormatError(
            f"could not read benchmark JSON {input_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkFormatError(
            f"invalid benchmark JSON {input_path}: {exc}"
        ) from exc

    root = _require_mapping(document, "benchmark root")
    series: dict[str, dict[str, tuple[float, ...]]] = {}
    for method in METHODS:
        method_rows = _resolve_method_rows(root, method)
        values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for density in REQUIRED_DENSITIES:
            density_key = str(density)
            if density_key not in method_rows:
                raise BenchmarkFormatError(
                    f"{method} results are missing required density {density}"
                )
            row = _require_mapping(
                method_rows[density_key],
                f"{method} density {density}",
            )
            if "summary" not in row:
                raise BenchmarkFormatError(
                    f"{method} density {density} is missing required field "
                    "'summary'"
                )
            summary = _require_mapping(
                row["summary"],
                f"{method} density {density} summary",
            )
            for metric in METRICS:
                values[metric].append(
                    _require_metric(
                        summary,
                        metric,
                        method=method,
                        density=density,
                    )
                )
        series[method] = {
            metric: tuple(metric_values)
            for metric, metric_values in values.items()
        }

    return BenchmarkData(densities=REQUIRED_DENSITIES, series=series)


def _plot_method_series(
    axis: plt.Axes,
    data: BenchmarkData,
    metric: str,
) -> None:
    for method in METHODS:
        axis.plot(
            data.densities,
            data.series[method][metric],
            label=METHOD_LABELS[method],
            linewidth=2.2,
            markersize=6,
            **METHOD_STYLES[method],
        )
    axis.set_xticks(data.densities)
    axis.set_xlabel("Pedestrian count")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    figure.tight_layout()
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "SocialNav-Bench"},
    )
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "SocialNav-Bench", "Date": None},
    )
    plt.close(figure)
    return png_path, svg_path


def generate_figures(data: BenchmarkData, output_dir: Path) -> tuple[Path, ...]:
    """Generate the six PNG/SVG benchmark assets and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1F2937",
            "axes.edgecolor": "#6B7280",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "socialnav-bench-density-figures",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
        _plot_method_series(axes[0], data, "success_rate")
        axes[0].set_title("Navigation success")
        axes[0].set_ylabel("Success rate")
        axes[0].set_ylim(0.0, 1.0)
        _plot_method_series(axes[1], data, "collision_rate")
        axes[1].set_title("Human collisions")
        axes[1].set_ylabel("Collision rate")
        axes[1].set_ylim(0.0, 1.0)
        created.extend(
            _save_figure(
                figure,
                output_dir,
                "success_collision_vs_density",
            )
        )

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
        _plot_method_series(axes[0], data, "mean_minimum_human_distance")
        axes[0].set_title("Human clearance")
        axes[0].set_ylabel("Mean minimum human distance (m)")
        maximum_distance = max(
            value
            for method in METHODS
            for value in data.series[method]["mean_minimum_human_distance"]
        )
        axes[0].set_ylim(0.0, max(1.0, maximum_distance * 1.1))
        _plot_method_series(axes[1], data, "mean_social_violation_rate")
        axes[1].set_title("Social-distance violations")
        axes[1].set_ylabel("Mean social violation rate")
        axes[1].set_ylim(0.0, 1.0)
        created.extend(
            _save_figure(figure, output_dir, "social_metrics_vs_density")
        )

        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        _plot_method_series(axis, data, "mean_spl")
        axis.set_title("Navigation efficiency by pedestrian density")
        axis.set_ylabel("Mean SPL")
        axis.set_ylim(0.0, 1.0)
        created.extend(_save_figure(figure, output_dir, "spl_vs_density"))

    return tuple(created)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Robust vs Shielded density-benchmark results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"benchmark JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"figure output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data = load_benchmark_data(args.input)
        created = generate_figures(data, args.output_dir)
    except BenchmarkFormatError as exc:
        parser.error(str(exc))
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
