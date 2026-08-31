#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "runs" / "paper_readiness_evaluation_framework_v1"
HIERARCHY_ROOT = REPO_ROOT / "runs" / "recovery_hierarchy" / "recovery_hierarchy_n500"
CCTP_ROOT = REPO_ROOT / "runs" / "clean_counterfactual_tube_projection_paper_v1"
CCTP_SOURCE_ROOT = REPO_ROOT / "runs" / "cctp_required_logit_artifacts_v1"
PRESSURE_SUITE_ROOT = REPO_ROOT / "runs" / "hysteresis_model_suite_v1" / "pressure_ramp"
RELABEL_AGG_ROOT = REPO_ROOT / "runs" / "hysteresis_model_suite_v1" / "aggregate"
TRUNCATION_ROOT = REPO_ROOT / "runs" / "expanded_truncation_boundary_v1"
LEGACY_TRUNCATION_ROOT = REPO_ROOT / "runs" / "truncation_boundary_v1"

MAIN_HIERARCHY_CONDITIONS = [
    "ordinary_strong_reset",
    "explicit_user_retraction",
    "system_level_reset",
    "fresh_context_deletion",
    "context_truncation",
    "neutral_summary_replacement",
    "factual_state_reconstruction",
    "self_verification_final_label",
]

HIERARCHY_LABELS = {
    "ordinary_strong_reset": "R0 ordinary reset",
    "explicit_user_retraction": "R1 user retraction",
    "system_level_reset": "R2 system reset",
    "fresh_context_deletion": "R3a fresh-context deletion",
    "context_truncation": "R3b context truncation",
    "neutral_summary_replacement": "R4 neutral summary",
    "factual_state_reconstruction": "R5 factual reconstruction",
    "self_verification_final_label": "R6 self-verification",
}

HIERARCHY_FAMILIES = {
    "strict": {
        "wrong_gap": 0.025,
        "accuracy_slack": 0.010,
        "p_correct_slack": 0.025,
        "entropy_slack": 0.100,
        "max_prob_slack": 0.100,
    },
    "main": {
        "wrong_gap": 0.050,
        "accuracy_slack": 0.030,
        "p_correct_slack": 0.050,
        "entropy_slack": 0.150,
        "max_prob_slack": 0.150,
    },
    "loose": {
        "wrong_gap": 0.075,
        "accuracy_slack": 0.050,
        "p_correct_slack": 0.075,
        "entropy_slack": 0.200,
        "max_prob_slack": 0.200,
    },
}

CCTP_TUBE_FAMILIES = {
    "strict": {
        "top_clean_probability_slack": 0.030,
        "entropy_slack": 0.100,
        "max_prob_slack": 0.100,
        "kl_threshold": 0.050,
    },
    "main": {
        "top_clean_probability_slack": 0.050,
        "entropy_slack": 0.150,
        "max_prob_slack": 0.150,
        "kl_threshold": 0.100,
    },
    "loose": {
        "top_clean_probability_slack": 0.100,
        "entropy_slack": 0.200,
        "max_prob_slack": 0.200,
        "kl_threshold": 0.200,
    },
}

CCTP_ALPHA_STEP = 0.01
NUM_TOL = 1e-9


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout


def git_diff_stat() -> str:
    result = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout


def source_artifacts() -> List[Path]:
    candidates = [
        HIERARCHY_ROOT / "recovery_hierarchy_raw.jsonl",
        HIERARCHY_ROOT / "recovery_hierarchy_summary.csv",
        HIERARCHY_ROOT / "minimal_recovery_operation.csv",
        HIERARCHY_ROOT / "recovery_threshold_sensitivity.csv",
        HIERARCHY_ROOT / "pressure_ramp_with_ci.csv",
        HIERARCHY_ROOT / "decomposition_rates.csv",
        HIERARCHY_ROOT / "relabel_with_ci.csv",
        TRUNCATION_ROOT / "expanded_truncation_summary.csv",
        TRUNCATION_ROOT / "expanded_truncation_by_model_dataset.csv",
        TRUNCATION_ROOT / "expanded_truncation_item_level.jsonl",
        LEGACY_TRUNCATION_ROOT / "truncation_boundary_summary.csv",
        CCTP_ROOT / "combined_item_level.csv",
        CCTP_ROOT / "combined_summary.csv",
        CCTP_ROOT / "combined_binding_constraints.csv",
        CCTP_ROOT / "combined_bootstrap_cis.csv",
        CCTP_ROOT / "decision.json",
        CCTP_SOURCE_ROOT / "combined_item_level.csv",
        RELABEL_AGG_ROOT / "semantic_letter_summary.csv",
        RELABEL_AGG_ROOT / "relabel_model_summary.csv",
    ]
    candidates.extend(sorted(PRESSURE_SUITE_ROOT.glob("*/*/pressure_ramp_item_summary.csv")))
    return [path for path in candidates if path.is_file()]


def source_checksum_rows() -> List[Dict[str, Any]]:
    rows = []
    for path in source_artifacts():
        rows.append({"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return rows


def write_checksum_file(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [f"{row['sha256']}  {row['path']}" for row in sorted(rows, key=lambda item: item["path"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def load_hierarchy_summary() -> pd.DataFrame:
    return read_csv_checked(HIERARCHY_ROOT / "recovery_hierarchy_summary.csv")


def load_hierarchy_raw() -> pd.DataFrame:
    return load_jsonl(HIERARCHY_ROOT / "recovery_hierarchy_raw.jsonl")


def load_cctp_items() -> pd.DataFrame:
    return read_csv_checked(CCTP_ROOT / "combined_item_level.csv")


def load_cctp_summary() -> pd.DataFrame:
    return read_csv_checked(CCTP_ROOT / "combined_summary.csv")


def load_truncation_summary() -> pd.DataFrame:
    path = TRUNCATION_ROOT / "expanded_truncation_summary.csv"
    if not path.is_file():
        path = LEGACY_TRUNCATION_ROOT / "truncation_boundary_summary.csv"
    return read_csv_checked(path)


def load_truncation_by_pair() -> pd.DataFrame:
    path = TRUNCATION_ROOT / "expanded_truncation_by_model_dataset.csv"
    if not path.is_file():
        path = LEGACY_TRUNCATION_ROOT / "truncation_boundary_by_model_dataset.csv"
    return read_csv_checked(path)


def load_pressure_items() -> pd.DataFrame:
    frames = []
    for path in sorted(PRESSURE_SUITE_ROOT.glob("*/*/pressure_ramp_item_summary.csv")):
        frame = pd.read_csv(path)
        parts = path.relative_to(PRESSURE_SUITE_ROOT).parts
        frame["model_key"] = parts[0]
        frame["dataset"] = parts[1]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def hierarchy_pass(row: pd.Series, thresholds: Dict[str, float]) -> bool:
    return bool(
        abs(float(row["intervention_hysteresis_gap"])) <= thresholds["wrong_gap"]
        and float(row["accuracy"]) >= float(row["clean_accuracy"]) - thresholds["accuracy_slack"]
        and float(row["intervention_p_correct"]) >= float(row["clean_p_correct"]) - thresholds["p_correct_slack"]
        and float(row["entropy"]) <= float(row["clean_entropy"]) + thresholds["entropy_slack"]
        and float(row["max_probability"]) >= float(row["clean_max_probability"]) - thresholds["max_prob_slack"]
    )


def hierarchy_counts(summary: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, int]:
    counts = {}
    main = summary[summary["condition"].isin(MAIN_HIERARCHY_CONDITIONS)].copy()
    for condition in MAIN_HIERARCHY_CONDITIONS:
        rows = main[main["condition"].eq(condition)]
        counts[condition] = int(rows.apply(lambda row: hierarchy_pass(row, thresholds), axis=1).sum())
    return counts


def hierarchy_minimal_counts(summary: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, int]:
    order = MAIN_HIERARCHY_CONDITIONS
    counts = {condition: 0 for condition in order}
    counts["not_restored"] = 0
    for _, group in summary[summary["condition"].isin(order)].groupby(["model_key", "dataset"], dropna=False):
        selected: Optional[str] = None
        for condition in order:
            rows = group[group["condition"].eq(condition)]
            if not rows.empty and hierarchy_pass(rows.iloc[0], thresholds):
                selected = condition
                break
        counts[selected or "not_restored"] += 1
    return counts


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_latex_table(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str], caption: str, label: str) -> None:
    ensure_dir(path.parent)
    row_end = r" \\" 
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\hline",
        " & ".join(columns) + row_end,
        r"\hline",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("_", r"\_") for column in columns]
        lines.append(" & ".join(values) + row_end)
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def ci_from_bootstrap(values: np.ndarray) -> Tuple[float, float]:
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def bootstrap_mean(values: np.ndarray, samples: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, arr.size, size=(samples, arr.size))
    boot = np.nanmean(arr[idx], axis=1)
    low, high = ci_from_bootstrap(boot)
    return float(np.nanmean(arr)), low, high


def add_ci_row(rows: List[Dict[str, Any]], table: str, group: Dict[str, Any], metric: str, estimate: float, low: float, high: float, n: int, samples: int, seed: int, source: Path, notes: str = "") -> None:
    row = {
        "table": table,
        "metric": metric,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "n": n,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "source_artifact_path": rel(source),
        "notes": notes,
    }
    row.update(group)
    rows.append(row)


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def parser_with_output(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    return parser


def output_root_from_args(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    return root if root.is_absolute() else REPO_ROOT / root
