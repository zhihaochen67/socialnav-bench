"""Render the frozen representative Robust-versus-Shielded demo episode."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socialnav.benchmark.diagnostics import EpisodeTrace  # noqa: E402
from socialnav.benchmark.failure_analysis import EpisodeEvidence  # noqa: E402
from socialnav.benchmark.robust_failure_probe import (  # noqa: E402
    run_robust_episode_with_evidence,
)
from socialnav.benchmark.robust_space_time_runner import (  # noqa: E402
    ROBUST_SPACE_TIME_METHOD,
    SHIELDED_SPACE_TIME_METHOD,
)
from socialnav.benchmark.scenario import (  # noqa: E402
    Scenario,
    generate_diverse_scenarios,
)
from socialnav.env.demo_map import SOCIAL_DISTANCE, grid_to_world  # noqa: E402
from socialnav.env.world import (  # noqa: E402
    HUMAN_COLLISION_DISTANCE,
    OBSTACLE_HALF_EXTENT,
    ROBOT_RADIUS,
    SIMULATION_STEP,
)
from socialnav.evaluation import EpisodeResult  # noqa: E402

DEFAULT_SCENARIO_ID = "diverse-seed-42-episode-0024-pedestrians-3"
DEFAULT_OUTPUT_DIR = Path("docs/assets")
DEFAULT_VIDEO_FPS = 20
DEFAULT_PREVIEW_FPS = 6
DEMO_METHODS = (ROBUST_SPACE_TIME_METHOD, SHIELDED_SPACE_TIME_METHOD)
METHOD_LABELS = {
    ROBUST_SPACE_TIME_METHOD: "Robust Space-Time Social",
    SHIELDED_SPACE_TIME_METHOD: "Shielded Space-Time Social",
}
METHOD_COLORS = {
    ROBUST_SPACE_TIME_METHOD: "#2563A6",
    SHIELDED_SPACE_TIME_METHOD: "#D97706",
}
HUMAN_COLORS = ("#8B5CF6", "#DB2777", "#0F9D8A", "#7C3AED", "#059669")
_SCENARIO_PATTERN = re.compile(
    r"^diverse-seed-(?P<seed>\d+)-episode-(?P<episode>\d+)"
    r"-pedestrians-(?P<pedestrians>\d+)$"
)


@dataclass(frozen=True)
class DemoEpisode:
    """One official benchmark replay and its read-only trajectory evidence."""

    method: str
    result: EpisodeResult
    trace: EpisodeTrace
    evidence: EpisodeEvidence


def resolve_scenario(scenario_id: str = DEFAULT_SCENARIO_ID) -> Scenario:
    """Regenerate exactly one frozen diverse scenario from its public ID."""
    match = _SCENARIO_PATTERN.fullmatch(scenario_id)
    if match is None:
        raise ValueError(
            "scenario id must match diverse-seed-<seed>-episode-<index>-"
            "pedestrians-<count>"
        )
    episode_index = int(match.group("episode"))
    scenario = generate_diverse_scenarios(
        episode_index + 1,
        int(match.group("seed")),
        pedestrian_count=int(match.group("pedestrians")),
    )[episode_index]
    if scenario.scenario_id != scenario_id:
        raise RuntimeError(
            f"resolved {scenario.scenario_id!r}, expected {scenario_id!r}"
        )
    return scenario


def capture_comparison(scenario: Scenario) -> tuple[DemoEpisode, ...]:
    """Replay both frozen methods through the existing instrumented runner."""
    episodes = []
    for method in DEMO_METHODS:
        result, trace, evidence = run_robust_episode_with_evidence(
            scenario,
            method=method,
        )
        episodes.append(DemoEpisode(method, result, trace, evidence))
    return tuple(episodes)


def first_collision(
    evidence: EpisodeEvidence,
) -> tuple[int, tuple[int, ...]] | None:
    """Return the first collision sample and involved pedestrian indices."""
    for sample in range(1, len(evidence.robot_trajectory)):
        robot = evidence.robot_trajectory[sample]
        indices = tuple(
            index
            for index, trajectory in enumerate(evidence.human_trajectories)
            if hypot(
                robot[0] - trajectory[sample][0],
                robot[1] - trajectory[sample][1],
            )
            <= HUMAN_COLLISION_DISTANCE
        )
        if indices:
            return sample, indices
    return None


def frame_times(duration: float, fps: int) -> tuple[float, ...]:
    """Return deterministic, endpoint-inclusive animation sample times."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_count = max(2, round(duration * fps))
    return tuple(
        index * duration / (frame_count - 1)
        for index in range(frame_count)
    )


def _validate_default_outcomes(episodes: Sequence[DemoEpisode]) -> None:
    by_method = {episode.method: episode for episode in episodes}
    robust = by_method[ROBUST_SPACE_TIME_METHOD]
    shielded = by_method[SHIELDED_SPACE_TIME_METHOD]
    if not (
        not robust.result.success
        and robust.trace.timed_out
        and robust.result.human_collision
    ):
        raise RuntimeError("the frozen Robust demo outcome no longer matches")
    if not (
        shielded.result.success
        and not shielded.trace.timed_out
        and not shielded.result.human_collision
    ):
        raise RuntimeError("the frozen Shielded demo outcome no longer matches")


def _outcome(episode: DemoEpisode) -> str:
    if episode.result.success:
        outcome = "SUCCESS"
    elif episode.trace.timed_out:
        outcome = "TIMEOUT"
    else:
        outcome = "STOPPED"
    if episode.result.human_collision:
        outcome += "  |  COLLISION"
    return outcome


def _draw_map(ax: Axes, scenario: Scenario) -> None:
    for obstacle in scenario.obstacle_cells:
        x, y = grid_to_world(obstacle, scenario.grid_scale)
        ax.add_patch(
            Rectangle(
                (x - OBSTACLE_HALF_EXTENT, y - OBSTACLE_HALF_EXTENT),
                2 * OBSTACLE_HALF_EXTENT,
                2 * OBSTACLE_HALF_EXTENT,
                facecolor="#3F4752",
                edgecolor="#252B33",
                linewidth=0.8,
                zorder=2,
            )
        )
    start = grid_to_world(scenario.start, scenario.grid_scale)
    goal = grid_to_world(scenario.goal, scenario.grid_scale)
    ax.scatter(
        *start,
        marker="D",
        s=50,
        color="#CBD5E1",
        edgecolor="#334155",
        linewidth=0.8,
        zorder=5,
    )
    ax.scatter(
        *goal,
        marker="*",
        s=190,
        color="#22C55E",
        edgecolor="#14532D",
        linewidth=0.8,
        zorder=6,
    )
    margin = scenario.grid_scale * 0.65
    ax.set_xlim(-margin, (scenario.grid_width - 1) * scenario.grid_scale + margin)
    ax.set_ylim(-margin, (scenario.grid_height - 1) * scenario.grid_scale + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#F8FAFC")
    ax.grid(color="#CBD5E1", linewidth=0.55, alpha=0.55)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _human_color(index: int) -> str:
    return HUMAN_COLORS[index % len(HUMAN_COLORS)]


def _trajectory_xy(
    trajectory: Sequence[tuple[float, float]],
    stop: int | None = None,
) -> tuple[list[float], list[float]]:
    selected = trajectory if stop is None else trajectory[:stop]
    return [point[0] for point in selected], [point[1] for point in selected]


def _snapshot_sample(episodes: Sequence[DemoEpisode]) -> int:
    robust = next(
        episode
        for episode in episodes
        if episode.method == ROBUST_SPACE_TIME_METHOD
    )
    collision = first_collision(robust.evidence)
    if collision is None:
        raise RuntimeError("representative Robust episode has no collision")
    return collision[0]


def save_trajectory_figure(
    scenario: Scenario,
    episodes: Sequence[DemoEpisode],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save README-ready PNG and SVG panels from the exact trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot_sample(episodes)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, episode in zip(axes, episodes):
        _draw_map(ax, scenario)
        for index, trajectory in enumerate(episode.evidence.human_trajectories):
            x, y = _trajectory_xy(trajectory)
            ax.plot(
                x,
                y,
                color=_human_color(index),
                linestyle=(0, (2, 2)),
                linewidth=1.4,
                alpha=0.72,
                zorder=3,
            )
            position = trajectory[min(snapshot, len(trajectory) - 1)]
            ax.add_patch(
                Circle(
                    position,
                    SOCIAL_DISTANCE,
                    facecolor=_human_color(index),
                    edgecolor=_human_color(index),
                    linewidth=0.8,
                    alpha=0.08,
                    zorder=1,
                )
            )
            ax.scatter(
                *position,
                s=46,
                color=_human_color(index),
                edgecolor="white",
                linewidth=0.7,
                zorder=6,
            )
        robot_x, robot_y = _trajectory_xy(episode.evidence.robot_trajectory)
        ax.plot(
            robot_x,
            robot_y,
            color=METHOD_COLORS[episode.method],
            linewidth=2.8,
            zorder=5,
        )
        ax.scatter(
            robot_x[0],
            robot_y[0],
            s=65,
            color=METHOD_COLORS[episode.method],
            edgecolor="white",
            linewidth=0.9,
            zorder=7,
        )
        collision = first_collision(episode.evidence)
        if collision is not None:
            point = episode.evidence.robot_trajectory[collision[0]]
            ax.scatter(
                *point,
                marker="X",
                s=150,
                color="#DC2626",
                edgecolor="white",
                linewidth=1.0,
                zorder=9,
            )
            ax.annotate(
                f"collision at {collision[0] * SIMULATION_STEP:.2f} s",
                point,
                xytext=(8, 10),
                textcoords="offset points",
                color="#991B1B",
                fontsize=9,
                fontweight="bold",
                zorder=10,
            )
        ax.set_title(
            f"{METHOD_LABELS[episode.method]}\n{_outcome(episode)}",
            color=METHOD_COLORS[episode.method],
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.02,
            f"minimum human distance: "
            f"{episode.result.minimum_human_distance:.3f} m",
            transform=ax.transAxes,
            fontsize=9,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9},
            zorder=12,
        )
    figure.suptitle(
        f"Same frozen seeded episode | N={scenario.pedestrian_count} | "
        f"social-space snapshot at t={snapshot * SIMULATION_STEP:.2f} s",
        fontsize=13,
        fontweight="bold",
    )
    legend = (
        Line2D([], [], color="#2563A6", linewidth=3, label="robot trajectory"),
        Line2D(
            [],
            [],
            color=HUMAN_COLORS[0],
            linestyle=(0, (2, 2)),
            linewidth=2,
            label="pedestrian trajectory",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="#CBD5E1",
            markeredgecolor="#334155",
            linestyle="None",
            label="robot start",
        ),
        Line2D(
            [],
            [],
            marker="*",
            markersize=12,
            color="#22C55E",
            markeredgecolor="#14532D",
            linestyle="None",
            label="goal",
        ),
    )
    figure.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
    png_path = output_dir / "trajectory_comparison.png"
    svg_path = output_dir / "trajectory_comparison.svg"
    figure.savefig(png_path, dpi=160, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
    return png_path, svg_path

def _build_animation(
    scenario: Scenario,
    episodes: Sequence[DemoEpisode],
    times: Sequence[float],
) -> tuple[object, animation.FuncAnimation]:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = []
    collisions = [first_collision(episode.evidence) for episode in episodes]
    for ax, episode in zip(axes, episodes):
        _draw_map(ax, scenario)
        robot_trail, = ax.plot(
            [],
            [],
            color=METHOD_COLORS[episode.method],
            linewidth=2.8,
            zorder=5,
        )
        robot = Circle(
            episode.evidence.robot_trajectory[0],
            ROBOT_RADIUS,
            facecolor=METHOD_COLORS[episode.method],
            edgecolor="white",
            linewidth=1.1,
            zorder=8,
        )
        ax.add_patch(robot)
        human_trails = []
        humans = []
        social_regions = []
        for index, trajectory in enumerate(episode.evidence.human_trajectories):
            trail, = ax.plot(
                [],
                [],
                color=_human_color(index),
                linestyle=(0, (2, 2)),
                linewidth=1.3,
                alpha=0.65,
                zorder=3,
            )
            social_region = Circle(
                trajectory[0],
                SOCIAL_DISTANCE,
                facecolor=_human_color(index),
                edgecolor=_human_color(index),
                linewidth=0.7,
                alpha=0.08,
                zorder=1,
            )
            human = Circle(
                trajectory[0],
                0.12,
                facecolor=_human_color(index),
                edgecolor="white",
                linewidth=0.8,
                zorder=7,
            )
            ax.add_patch(social_region)
            ax.add_patch(human)
            human_trails.append(trail)
            humans.append(human)
            social_regions.append(social_region)
        collision_marker, = ax.plot(
            [],
            [],
            marker="X",
            markersize=12,
            markerfacecolor="#DC2626",
            markeredgecolor="white",
            linestyle="None",
            zorder=10,
        )
        ax.set_title(
            f"{METHOD_LABELS[episode.method]}\nOutcome: {_outcome(episode)}",
            color=METHOD_COLORS[episode.method],
            fontweight="bold",
        )
        state_text = ax.text(
            0.02,
            0.97,
            "RUNNING",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
            zorder=12,
        )
        panels.append(
            (
                robot_trail,
                robot,
                tuple(human_trails),
                tuple(humans),
                tuple(social_regions),
                collision_marker,
                state_text,
            )
        )
    time_text = figure.suptitle(
        f"Frozen seeded comparison | {scenario.scenario_id}",
        fontsize=12,
        fontweight="bold",
    )

    def update(frame: int) -> list[object]:
        time_seconds = times[frame]
        time_text.set_text(
            f"Frozen seeded comparison | N={scenario.pedestrian_count} | "
            f"t={time_seconds:05.2f} s"
        )
        artists: list[object] = [time_text]
        for episode, panel, collision in zip(episodes, panels, collisions):
            (
                robot_trail,
                robot,
                human_trails,
                humans,
                social_regions,
                collision_marker,
                state_text,
            ) = panel
            evidence = episode.evidence
            sample = min(
                round(time_seconds / SIMULATION_STEP),
                len(evidence.robot_trajectory) - 1,
            )
            robot_x, robot_y = _trajectory_xy(
                evidence.robot_trajectory,
                sample + 1,
            )
            robot_trail.set_data(robot_x, robot_y)
            robot.center = evidence.robot_trajectory[sample]
            for index, trajectory in enumerate(evidence.human_trajectories):
                human_sample = min(sample, len(trajectory) - 1)
                x, y = _trajectory_xy(trajectory, human_sample + 1)
                human_trails[index].set_data(x, y)
                position = trajectory[human_sample]
                humans[index].center = position
                social_regions[index].center = position
            finished = sample == len(evidence.robot_trajectory) - 1
            collision_reached = collision is not None and sample >= collision[0]
            if collision_reached and collision is not None:
                point = evidence.robot_trajectory[collision[0]]
                collision_marker.set_data([point[0]], [point[1]])
            else:
                collision_marker.set_data([], [])
            if finished:
                state = _outcome(episode)
            elif collision_reached:
                state = "COLLISION DETECTED"
            else:
                state = "RUNNING"
            state_text.set_text(state)
            state_text.set_color(
                "#B91C1C"
                if collision_reached
                else (
                    "#15803D"
                    if finished and episode.result.success
                    else "#334155"
                )
            )
            artists.extend(
                [
                    robot_trail,
                    robot,
                    collision_marker,
                    state_text,
                    *human_trails,
                    *humans,
                    *social_regions,
                ]
            )
        return artists

    demo_animation = animation.FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=1000 * times[-1] / (len(times) - 1),
        blit=False,
    )
    return figure, demo_animation


def _ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "MP4 rendering requires ffmpeg. Install system ffmpeg or run "
            "`python -m pip install imageio-ffmpeg`."
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def save_video(
    scenario: Scenario,
    episodes: Sequence[DemoEpisode],
    output_path: Path,
    *,
    fps: int = DEFAULT_VIDEO_FPS,
) -> Path:
    """Render an H.264 MP4 using system or imageio-bundled ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(
        episode.result.steps * SIMULATION_STEP for episode in episodes
    )
    times = frame_times(duration, fps)
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_executable()
    figure, demo_animation = _build_animation(scenario, episodes, times)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=(
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
        ),
        metadata={"title": "SocialNav-Bench Robust versus Shielded demo"},
    )
    try:
        demo_animation.save(output_path, writer=writer, dpi=100)
    finally:
        plt.close(figure)
    return output_path


def save_preview_gif(
    scenario: Scenario,
    episodes: Sequence[DemoEpisode],
    output_path: Path,
    *,
    fps: int = DEFAULT_PREVIEW_FPS,
) -> Path:
    """Render a compact README-compatible animated preview."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(
        episode.result.steps * SIMULATION_STEP for episode in episodes
    )
    times = frame_times(duration, fps)
    figure, demo_animation = _build_animation(scenario, episodes, times)
    try:
        demo_animation.save(
            output_path,
            writer=animation.PillowWriter(fps=fps),
            dpi=75,
        )
    finally:
        plt.close(figure)
    return output_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=_positive_int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--preview-fps",
        type=_positive_int,
        default=DEFAULT_PREVIEW_FPS,
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="render only the trajectory figure and optional GIF",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="omit the compact README GIF",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = resolve_scenario(args.scenario_id)
    episodes = capture_comparison(scenario)
    if scenario.scenario_id == DEFAULT_SCENARIO_ID:
        _validate_default_outcomes(episodes)

    created: list[Path] = list(
        save_trajectory_figure(scenario, episodes, args.output_dir)
    )
    if not args.skip_video:
        created.append(
            save_video(
                scenario,
                episodes,
                args.output_dir / "socialnav_demo.mp4",
                fps=args.fps,
            )
        )
    if not args.skip_preview:
        created.append(
            save_preview_gif(
                scenario,
                episodes,
                args.output_dir / "socialnav_demo_preview.gif",
                fps=args.preview_fps,
            )
        )

    print(f"Scenario: {scenario.scenario_id}")
    for episode in episodes:
        print(
            f"{METHOD_LABELS[episode.method]}: {_outcome(episode)}; "
            f"steps={episode.result.steps}; "
            f"minimum_human_distance="
            f"{episode.result.minimum_human_distance:.3f} m"
        )
    for path in created:
        print(f"Created {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
