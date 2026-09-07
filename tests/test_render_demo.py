"""Lightweight tests for deterministic demo selection and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import experiments.render_demo as render_demo
from socialnav.benchmark.diagnostics import EpisodeTrace
from socialnav.benchmark.failure_analysis import EpisodeEvidence
from socialnav.evaluation import EpisodeResult


def _episode(
    method: str,
    *,
    collision: bool = False,
) -> render_demo.DemoEpisode:
    if collision:
        robot = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))
        humans = (((1.0, 0.6), (0.7, 0.2), (0.7, 0.2)),)
    else:
        robot = ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0))
        humans = (((1.0, 0.6), (1.2, 0.6), (1.4, 0.6)),)
    result = EpisodeResult(
        success=not collision,
        path_length=1.0,
        time_to_goal=None if collision else 2 * render_demo.SIMULATION_STEP,
        spl=0.0 if collision else 1.0,
        minimum_human_distance=0.2 if collision else 0.6,
        social_violation_rate=0.5 if collision else 0.0,
        human_collision=collision,
        obstacle_collision=False,
        steps=2,
    )
    trace = EpisodeTrace(
        planned_path=robot,
        speed_scales=(1.0, 1.0),
        timed_out=collision,
        final_robot_position=robot[-1],
        final_pedestrian_position=humans[0][-1],
        max_steps=2,
    )
    evidence = EpisodeEvidence(
        robot_trajectory=robot,
        human_trajectories=humans,
        phases=("start", "MOVE", "MOVE"),
    )
    return render_demo.DemoEpisode(method, result, trace, evidence)


def test_representative_scenario_resolves_deterministically() -> None:
    first = render_demo.resolve_scenario()
    second = render_demo.resolve_scenario()

    assert first == second
    assert first.scenario_id == render_demo.DEFAULT_SCENARIO_ID
    assert first.pedestrian_count == 3


def test_comparison_uses_only_expected_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_capture(
        scenario: object,
        *,
        method: str,
    ) -> tuple[object, ...]:
        calls.append(method)
        episode = _episode(
            method,
            collision=method == render_demo.ROBUST_SPACE_TIME_METHOD,
        )
        return episode.result, episode.trace, episode.evidence

    monkeypatch.setattr(
        render_demo,
        "run_robust_episode_with_evidence",
        fake_capture,
    )

    episodes = render_demo.capture_comparison(render_demo.resolve_scenario())

    assert calls == list(render_demo.DEMO_METHODS)
    assert (
        tuple(episode.method for episode in episodes)
        == render_demo.DEMO_METHODS
    )


def test_frame_times_are_deterministic_and_endpoint_inclusive() -> None:
    expected = render_demo.frame_times(2.0, 2)

    assert expected == (0.0, 2 / 3, 4 / 3, 2.0)
    assert render_demo.frame_times(2.0, 2) == expected
    with pytest.raises(ValueError, match="fps must be positive"):
        render_demo.frame_times(2.0, 0)


def test_first_collision_reports_sample_and_human() -> None:
    episode = _episode(
        render_demo.ROBUST_SPACE_TIME_METHOD,
        collision=True,
    )

    assert render_demo.first_collision(episode.evidence) == (1, (0,))


def test_trajectory_figure_smoke(tmp_path: Path) -> None:
    scenario = render_demo.resolve_scenario()
    episodes = (
        _episode(render_demo.ROBUST_SPACE_TIME_METHOD, collision=True),
        _episode(render_demo.SHIELDED_SPACE_TIME_METHOD),
    )

    png, svg = render_demo.save_trajectory_figure(
        scenario,
        episodes,
        tmp_path / "nested",
    )

    assert png.stat().st_size > 1_000
    assert svg.stat().st_size > 1_000
    with Image.open(png) as image:
        assert image.size == (2048, 1024)
