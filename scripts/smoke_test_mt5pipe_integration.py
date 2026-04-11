#!/usr/bin/env python3
r"""
Smoke test the APHELION <-> mt5pipe integration from the main repo.

What it checks:
  1. the main APHELION repo can bootstrap the external data-pipeline package
  2. a trusted dataset artifact alias resolves through mt5pipe's public API
  3. the dataset split parquet files are readable from outside the data repo
  4. experiment/model artifacts, if requested, line up with that dataset

Usage:
    python scripts/smoke_test_mt5pipe_integration.py
    python scripts/smoke_test_mt5pipe_integration.py --dataset-ref dataset://xau_m1_nonhuman@1.0.0
    python scripts/smoke_test_mt5pipe_integration.py --skip-training-checks
    python scripts/smoke_test_mt5pipe_integration.py --datapipe-root C:\path\to\Datapipe\aphelion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test mt5pipe integration from the main APHELION repo.")
    parser.add_argument(
        "--datapipe-root",
        default="",
        help="Path to the Datapipe/aphelion repo root. Falls back to APHELION_MT5PIPE_ROOT or common local paths.",
    )
    parser.add_argument(
        "--dataset-ref",
        default="dataset://xau_m1_nonhuman@1.0.0",
        help="Trusted dataset alias or artifact id to inspect.",
    )
    parser.add_argument(
        "--experiment-ref",
        default="experiment://xau_m1_nonhuman_direction_nb@1.0.0",
        help="Experiment alias to inspect when training checks are enabled.",
    )
    parser.add_argument(
        "--model-ref",
        default="model://xau_m1_nonhuman_direction_nb@1.0.0",
        help="Model alias to inspect when training checks are enabled.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="How many train rows to print as a tiny sample.",
    )
    parser.add_argument(
        "--skip-training-checks",
        action="store_true",
        help="Only validate dataset integration and skip experiment/model checks.",
    )
    return parser.parse_args()


def _resolve_datapipe_root(explicit_value: str) -> Path:
    candidates: list[Path] = []

    if explicit_value:
        candidates.append(Path(explicit_value).expanduser())

    env_value = os.environ.get("APHELION_MT5PIPE_ROOT", "")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            Path.home() / "Downloads" / "Datapipe" / "aphelion",
            ROOT.parent / "Datapipe" / "aphelion",
            ROOT / "data_pipeline" / "aphelion",
        ]
    )

    for candidate in candidates:
        repo_root = candidate.resolve()
        if (repo_root / "data" / "config" / "pipeline.yaml").exists() and (repo_root / "data" / "mt5pipe").exists():
            return repo_root

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise SystemExit(
        "Unable to locate the Datapipe repo root.\n"
        "Pass --datapipe-root or set APHELION_MT5PIPE_ROOT.\n"
        f"Searched:\n{searched}"
    )


def _bootstrap_mt5pipe(datapipe_root: Path) -> Path:
    data_root = datapipe_root / "data"
    config_path = data_root / "config" / "pipeline.yaml"
    os.environ["MT5PIPE_CONFIG"] = str(config_path)
    if str(data_root) not in sys.path:
        sys.path.insert(0, str(data_root))
    return data_root


def _resolve_data_path(data_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = [
        (data_root / path).resolve(),
        (data_root.parent / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _format_mapping(mapping: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in mapping.items())


def main() -> None:
    args = _parse_args()
    datapipe_root = _resolve_datapipe_root(args.datapipe_root)
    data_root = _bootstrap_mt5pipe(datapipe_root)

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - hard failure is the point
        raise SystemExit("polars is required for the mt5pipe integration smoke test.") from exc

    from mt5pipe.compiler.public import inspect_artifact, inspect_experiment, inspect_model

    dataset = inspect_artifact(args.dataset_ref)
    if dataset.artifact.status not in {"accepted", "published"}:
        raise SystemExit(
            f"Dataset artifact {dataset.artifact.artifact_id} is not usable: status={dataset.artifact.status}"
        )
    if dataset.trust_report is None or not dataset.trust_report.accepted_for_publication:
        raise SystemExit(
            f"Dataset artifact {dataset.artifact.artifact_id} is not truth-accepted for publication."
        )

    artifact_root = _resolve_data_path(data_root, dataset.manifest.artifact_uri)
    if not artifact_root.exists():
        raise SystemExit(f"Dataset artifact parquet root does not exist: {artifact_root}")

    split_counts_verified: dict[str, int] = {}
    sample_frame = None
    for split_name, expected_rows in dataset.split_row_counts.items():
        split_dir = artifact_root / f"split={split_name}"
        split_files = sorted(split_dir.glob("*.parquet"))
        if not split_files:
            raise SystemExit(f"Missing parquet files for split '{split_name}' in {split_dir}")
        frame = pl.concat([pl.read_parquet(path) for path in split_files], how="diagonal_relaxed")
        split_counts_verified[split_name] = frame.height
        if frame.height != int(expected_rows):
            raise SystemExit(
                f"Split '{split_name}' row mismatch: manifest={expected_rows}, read={frame.height}"
            )
        if split_name == "train":
            sample_frame = frame.head(max(1, args.sample_rows))

    label_columns = [
        column
        for column in dataset.schema_columns
        if column.startswith("direction_") or column.startswith("triple_barrier_")
    ]

    print("mt5pipe integration smoke test")
    print(f"aphelion_repo: {ROOT}")
    print(f"datapipe_repo: {datapipe_root}")
    print(f"dataset_ref: {args.dataset_ref}")
    print(f"dataset_artifact_id: {dataset.artifact.artifact_id}")
    print(f"dataset_status: {dataset.artifact.status}")
    print(f"trust_status: {dataset.trust_report.status}")
    print(f"trust_decision: {dataset.trust_decision_summary}")
    print(f"time_range: {dataset.time_range['start']} -> {dataset.time_range['end']}")
    print(f"feature_families: {', '.join(dataset.feature_families)}")
    print(f"split_rows_manifest: {_format_mapping(dataset.split_row_counts)}")
    print(f"split_rows_verified: {_format_mapping(split_counts_verified)}")
    print(f"schema_columns: {len(dataset.schema_columns)}")
    print(f"label_columns: {', '.join(label_columns[:8]) if label_columns else '-'}")
    print(f"artifact_root: {artifact_root}")
    if dataset.source_modes:
        print(f"source_modes: {_format_mapping(dataset.source_modes)}")
    if dataset.trust_score_breakdown:
        print(f"trust_breakdown: {_format_mapping(dataset.trust_score_breakdown)}")

    if sample_frame is not None and sample_frame.height > 0:
        print("\ntrain_sample:")
        for row in sample_frame.to_dicts():
            print(json.dumps(row, default=str, sort_keys=True))

    if args.skip_training_checks:
        print("\ntraining_checks: skipped")
        return

    experiment = inspect_experiment(args.experiment_ref)
    model = inspect_model(args.model_ref)

    if experiment.dataset_artifact is None or experiment.dataset_artifact.artifact_id != dataset.artifact.artifact_id:
        raise SystemExit(
            "Experiment artifact is not linked to the inspected dataset artifact."
        )
    if model.dataset_artifact is None or model.dataset_artifact.artifact_id != dataset.artifact.artifact_id:
        raise SystemExit(
            "Model artifact is not linked to the inspected dataset artifact."
        )

    holdout_balanced_accuracy = model.summary.get("holdout_metrics", {}).get("balanced_accuracy")
    print("\ntraining_artifacts:")
    print(f"experiment_ref: {args.experiment_ref}")
    print(f"experiment_artifact_id: {experiment.artifact.artifact_id}")
    print(f"experiment_status: {experiment.artifact.status}")
    print(f"model_ref: {args.model_ref}")
    print(f"model_artifact_id: {model.artifact.artifact_id}")
    print(f"model_status: {model.artifact.status}")
    print(
        "walk_forward_balanced_accuracy_mean: "
        f"{experiment.summary.get('walk_forward_summary', {}).get('balanced_accuracy_mean')}"
    )
    print(f"holdout_balanced_accuracy: {holdout_balanced_accuracy}")
    if experiment.predictions_path is not None:
        print(f"predictions_path: {experiment.predictions_path}")


if __name__ == "__main__":
    main()
