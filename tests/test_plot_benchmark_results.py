"""Focused tests for the frozen benchmark plotting utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.plot_benchmark_results import (
    BenchmarkFormatError,
    generate_figures,
    load_benchmark_data,
    main,
)


def _benchmark_document() -> dict[str, object]:
    document: dict[str, object] = {}
    for method, offset in (("robust", 0.0), ("shielded", 0.05)):
        document[method] = {
            str(density): {
                "summary": {
                    "success_rate": 0.2 + offset + index * 0.1,
                    "collision_rate": 0.4 - offset - index * 0.1,
                    "mean_spl": 0.15 + offset + index * 0.1,
                    "mean_minimum_human_distance": (
                        0.45 + offset + index * 0.05
                    ),
                    "mean_social_violation_rate": (
                        0.35 - offset - index * 0.05
                    ),
                }
            }
            for index, density in enumerate((3, 5, 10))
        }
    return document


def _write_benchmark(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def test_load_benchmark_data_finds_expected_densities(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    _write_benchmark(input_path, _benchmark_document())

    data = load_benchmark_data(input_path)

    assert data.densities == (3, 5, 10)
    assert data.series["robust"]["success_rate"] == pytest.approx(
        (0.2, 0.3, 0.4)
    )
    assert data.series["shielded"]["collision_rate"] == pytest.approx(
        (0.35, 0.25, 0.15)
    )


def test_missing_required_field_fails_clearly(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    document = _benchmark_document()
    shielded = document["shielded"]
    assert isinstance(shielded, dict)
    density_five = shielded["5"]
    assert isinstance(density_five, dict)
    summary = density_five["summary"]
    assert isinstance(summary, dict)
    del summary["mean_spl"]
    _write_benchmark(input_path, document)

    with pytest.raises(
        BenchmarkFormatError,
        match="shielded density 5 summary is missing required field 'mean_spl'",
    ):
        load_benchmark_data(input_path)


def test_missing_required_density_fails_clearly(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    document = _benchmark_document()
    robust = document["robust"]
    assert isinstance(robust, dict)
    del robust["10"]
    _write_benchmark(input_path, document)

    with pytest.raises(
        BenchmarkFormatError,
        match="robust results are missing required density 10",
    ):
        load_benchmark_data(input_path)


def test_generate_figures_creates_non_empty_png_and_svg_files(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "benchmark.json"
    output_dir = tmp_path / "figures"
    _write_benchmark(input_path, _benchmark_document())

    created = generate_figures(load_benchmark_data(input_path), output_dir)

    assert {path.name for path in created} == {
        "success_collision_vs_density.png",
        "success_collision_vs_density.svg",
        "social_metrics_vs_density.png",
        "social_metrics_vs_density.svg",
        "spl_vs_density.png",
        "spl_vs_density.svg",
    }
    assert all(path.stat().st_size > 1_000 for path in created)


def test_cli_does_not_mutate_source_json(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    output_dir = tmp_path / "figures"
    _write_benchmark(input_path, _benchmark_document())
    original_bytes = input_path.read_bytes()

    assert main(
        ["--input", str(input_path), "--output-dir", str(output_dir)]
    ) == 0

    assert input_path.read_bytes() == original_bytes
