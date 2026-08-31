#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_DATASETS = [
    ("qwen25_15b", "Qwen2.5-1.5B", "truthfulqa_mc"),
    ("qwen25_15b", "Qwen2.5-1.5B", "mmlu_pro"),
    ("qwen25_7b", "Qwen2.5-7B", "truthfulqa_mc"),
    ("qwen25_7b", "Qwen2.5-7B", "mmlu_pro"),
    ("qwen25_14b", "Qwen2.5-14B", "truthfulqa_mc"),
    ("qwen25_14b", "Qwen2.5-14B", "mmlu_pro"),
    ("mistral7b_v03", "Mistral-7B", "truthfulqa_mc"),
    ("mistral7b_v03", "Mistral-7B", "mmlu_pro"),
]
PAIR_ORDER = {(model, dataset): index for index, (model, _, dataset) in enumerate(MODEL_DATASETS)}
MODEL_DISPLAY = {model_key: model for model_key, model, _ in MODEL_DATASETS}
DATASET_DISPLAY = {"truthfulqa_mc": "TruthfulQA-MC", "mmlu_pro": "MMLU-Pro"}

MAIN_CONDITIONS = [
    "ordinary_strong_reset",
    "explicit_user_retraction",
    "system_level_reset",
    "fresh_context_deletion",
    "context_truncation",
    "neutral_summary_replacement",
    "factual_state_reconstruction",
    "self_verification_final_label",
]
CONDITION_LABELS = {
    "ordinary_strong_reset": "R0 Ordinary reset",
    "explicit_user_retraction": "R1 User retraction",
    "system_level_reset": "R2 System reset",
    "fresh_context_deletion": "R3 Fresh deletion",
    "context_truncation": "R3 Context truncation",
    "neutral_summary_replacement": "R4 Neutral summary",
    "factual_state_reconstruction": "R5 Factual reconstruction",
    "self_verification_final_label": "R6 Self-verification",
}


@dataclass(frozen=True)
class ThresholdFamily:
    name: str
    gap: float
    accuracy_slack: float
    p_correct_slack: float
    entropy_slack: float
    max_prob_slack: float


THRESHOLDS = {
    "strict": ThresholdFamily("strict", 0.025, 0.01, 0.025, 0.10, 0.10),
    "main": ThresholdFamily("main", 0.050, 0.03, 0.050, 0.15, 0.15),
    "loose": ThresholdFamily("loose", 0.075, 0.05, 0.075, 0.20, 0.20),
}

PAPER_VALUES: Dict[Tuple[str, str, str, str], float] = {}


def add_expected(table: str, model: str, dataset: str, values: Dict[str, float]) -> None:
    for metric, value in values.items():
        PAPER_VALUES[(table, model, dataset, metric)] = value


for model, dataset, values in [
    ("Qwen2.5-1.5B", "TruthfulQA-MC", {"clean_pw_mean": 0.166, "press_pw_mean": 0.897, "reset_pw_mean": 0.665, "hysteresis_gap": 0.499, "positive_hysteresis_rate": 0.910, "lock_in": 0.746}),
    ("Qwen2.5-1.5B", "MMLU-Pro", {"clean_pw_mean": 0.091, "press_pw_mean": 0.925, "reset_pw_mean": 0.726, "hysteresis_gap": 0.634, "positive_hysteresis_rate": 0.924, "lock_in": 0.809}),
    ("Qwen2.5-7B", "TruthfulQA-MC", {"clean_pw_mean": 0.096, "press_pw_mean": 0.926, "reset_pw_mean": 0.107, "hysteresis_gap": 0.011, "positive_hysteresis_rate": 0.382, "lock_in": 0.114}),
    ("Qwen2.5-7B", "MMLU-Pro", {"clean_pw_mean": 0.081, "press_pw_mean": 0.971, "reset_pw_mean": 0.343, "hysteresis_gap": 0.262, "positive_hysteresis_rate": 0.518, "lock_in": 0.354}),
    ("Qwen2.5-14B", "TruthfulQA-MC", {"clean_pw_mean": 0.076, "press_pw_mean": 0.933, "reset_pw_mean": 0.113, "hysteresis_gap": 0.037, "positive_hysteresis_rate": 0.344, "lock_in": 0.117}),
    ("Qwen2.5-14B", "MMLU-Pro", {"clean_pw_mean": 0.084, "press_pw_mean": 0.957, "reset_pw_mean": 0.356, "hysteresis_gap": 0.272, "positive_hysteresis_rate": 0.710, "lock_in": 0.372}),
    ("Mistral-7B", "TruthfulQA-MC", {"clean_pw_mean": 0.105, "press_pw_mean": 0.826, "reset_pw_mean": 0.352, "hysteresis_gap": 0.247, "positive_hysteresis_rate": 0.886, "lock_in": 0.443}),
    ("Mistral-7B", "MMLU-Pro", {"clean_pw_mean": 0.091, "press_pw_mean": 0.915, "reset_pw_mean": 0.750, "hysteresis_gap": 0.659, "positive_hysteresis_rate": 0.936, "lock_in": 0.820}),
]:
    add_expected("pressure_ramp", model, dataset, values)

for model, dataset, values in [
    ("Qwen2.5-1.5B", "TruthfulQA-MC", {"pressure_gap": 0.499, "neutral_gap": 0.016, "correct_reset_pc": 0.630, "correct_reset_pw": 0.119}),
    ("Qwen2.5-1.5B", "MMLU-Pro", {"pressure_gap": 0.634, "neutral_gap": -0.001, "correct_reset_pc": 0.820, "correct_reset_pw": 0.019}),
    ("Qwen2.5-7B", "TruthfulQA-MC", {"pressure_gap": 0.011, "neutral_gap": 0.001, "correct_reset_pc": 0.738, "correct_reset_pw": 0.063}),
    ("Qwen2.5-7B", "MMLU-Pro", {"pressure_gap": 0.262, "neutral_gap": -0.023, "correct_reset_pc": 0.726, "correct_reset_pw": 0.034}),
    ("Qwen2.5-14B", "TruthfulQA-MC", {"pressure_gap": 0.037, "neutral_gap": -0.001, "correct_reset_pc": 0.821, "correct_reset_pw": 0.050}),
    ("Qwen2.5-14B", "MMLU-Pro", {"pressure_gap": 0.272, "neutral_gap": -0.009, "correct_reset_pc": 0.801, "correct_reset_pw": 0.029}),
    ("Mistral-7B", "TruthfulQA-MC", {"pressure_gap": 0.247, "neutral_gap": -0.011, "correct_reset_pc": 0.840, "correct_reset_pw": 0.042}),
    ("Mistral-7B", "MMLU-Pro", {"pressure_gap": 0.659, "neutral_gap": -0.016, "correct_reset_pc": 0.892, "correct_reset_pw": 0.010}),
]:
    add_expected("behavioral_controls", model, dataset, values)

for model, dataset, values in [
    ("Qwen2.5-1.5B", "TruthfulQA-MC", {"stable_correct": 0.076, "history_contamination": 0.416, "context_correction": 0.012, "task_failure": 0.496, "contamination_given_clean_correct": 0.626}),
    ("Qwen2.5-1.5B", "MMLU-Pro", {"stable_correct": 0.058, "history_contamination": 0.216, "context_correction": 0.028, "task_failure": 0.698, "contamination_given_clean_correct": 0.730}),
    ("Qwen2.5-7B", "TruthfulQA-MC", {"stable_correct": 0.598, "history_contamination": 0.102, "context_correction": 0.068, "task_failure": 0.232, "contamination_given_clean_correct": 0.046}),
    ("Qwen2.5-7B", "MMLU-Pro", {"stable_correct": 0.260, "history_contamination": 0.164, "context_correction": 0.046, "task_failure": 0.530, "contamination_given_clean_correct": 0.241}),
    ("Qwen2.5-14B", "TruthfulQA-MC", {"stable_correct": 0.684, "history_contamination": 0.078, "context_correction": 0.054, "task_failure": 0.184, "contamination_given_clean_correct": 0.045}),
    ("Qwen2.5-14B", "MMLU-Pro", {"stable_correct": 0.316, "history_contamination": 0.140, "context_correction": 0.044, "task_failure": 0.500, "contamination_given_clean_correct": 0.215}),
    ("Mistral-7B", "TruthfulQA-MC", {"stable_correct": 0.390, "history_contamination": 0.260, "context_correction": 0.042, "task_failure": 0.308, "contamination_given_clean_correct": 0.280}),
    ("Mistral-7B", "MMLU-Pro", {"stable_correct": 0.064, "history_contamination": 0.262, "context_correction": 0.026, "task_failure": 0.648, "contamination_given_clean_correct": 0.693}),
]:
    add_expected("failure_decomposition", model, dataset, values)

for model, dataset, values in [
    ("Qwen2.5-1.5B", "TruthfulQA-MC", {"baseline_pw": 0.773, "shuffle_pw": 0.536, "wrong_reduction": 0.237}),
    ("Qwen2.5-1.5B", "MMLU-Pro", {"baseline_pw": 0.840, "shuffle_pw": 0.514, "wrong_reduction": 0.325}),
    ("Qwen2.5-7B", "MMLU-Pro", {"baseline_pw": 0.514, "shuffle_pw": 0.491, "wrong_reduction": 0.024}),
    ("Qwen2.5-14B", "MMLU-Pro", {"baseline_pw": 0.528, "shuffle_pw": 0.620, "wrong_reduction": -0.092}),
    ("Mistral-7B", "TruthfulQA-MC", {"baseline_pw": 0.577, "shuffle_pw": 0.686, "wrong_reduction": -0.109}),
    ("Mistral-7B", "MMLU-Pro", {"baseline_pw": 0.850, "shuffle_pw": 0.809, "wrong_reduction": 0.042}),
]:
    add_expected("relabel_effect", model, dataset, values)

for model, dataset, values in [
    ("Qwen2.5-1.5B", "TruthfulQA-MC", {"semantic_following_rate": 0.544, "old_letter_following_rate": 0.216, "chance_old_letter_rate": 0.229, "new_wrong_following_rate": 0.544}),
    ("Qwen2.5-1.5B", "MMLU-Pro", {"semantic_following_rate": 0.562, "old_letter_following_rate": 0.118, "chance_old_letter_rate": 0.112, "new_wrong_following_rate": 0.562}),
    ("Qwen2.5-7B", "MMLU-Pro", {"semantic_following_rate": 0.496, "old_letter_following_rate": 0.090, "chance_old_letter_rate": 0.112, "new_wrong_following_rate": 0.496}),
    ("Qwen2.5-14B", "MMLU-Pro", {"semantic_following_rate": 0.620, "old_letter_following_rate": 0.052, "chance_old_letter_rate": 0.112, "new_wrong_following_rate": 0.620}),
    ("Mistral-7B", "TruthfulQA-MC", {"semantic_following_rate": 0.702, "old_letter_following_rate": 0.114, "chance_old_letter_rate": 0.229, "new_wrong_following_rate": 0.702}),
    ("Mistral-7B", "MMLU-Pro", {"semantic_following_rate": 0.820, "old_letter_following_rate": 0.068, "chance_old_letter_rate": 0.112, "new_wrong_following_rate": 0.820}),
]:
    add_expected("semantic_letter", model, dataset, values)

RECOVERY_SUMMARY_EXPECTED = {
    "ordinary_strong_reset": (1, 0, 7),
    "explicit_user_retraction": (1, 1, 6),
    "system_level_reset": (1, 0, 7),
    "fresh_context_deletion": (8, 0, 0),
    "context_truncation": (8, 0, 0),
    "neutral_summary_replacement": (6, 1, 1),
    "factual_state_reconstruction": (7, 1, 0),
    "self_verification_final_label": (1, 0, 7),
}

THRESHOLD_EXPECTED = {
    "strict": {"r0": 0, "r1": 1, "r2": 0, "r3_fresh": 8, "r3_truncation": 7, "r4": 5, "r5": 5, "r6": 0},
    "main": {"r0": 1, "r1": 1, "r2": 1, "r3_fresh": 8, "r3_truncation": 8, "r4": 6, "r5": 7, "r6": 1},
    "loose": {"r0": 2, "r1": 2, "r2": 2, "r3_fresh": 8, "r3_truncation": 8, "r4": 7, "r5": 7, "r6": 2},
}

MRO_EXPECTED = {
    ("Mistral-7B", "MMLU-Pro"): ("R3 Fresh deletion", 1.000, 0.326),
    ("Mistral-7B", "TruthfulQA-MC"): ("R3 Fresh deletion", 1.000, 0.650),
    ("Qwen2.5-14B", "MMLU-Pro"): ("R3 Fresh deletion", 1.000, 0.456),
    ("Qwen2.5-14B", "TruthfulQA-MC"): ("R0 Ordinary reset", np.nan, 0.738),
    ("Qwen2.5-1.5B", "MMLU-Pro"): ("R3 Fresh deletion", 1.000, 0.274),
    ("Qwen2.5-1.5B", "TruthfulQA-MC"): ("R3 Fresh deletion", 1.000, 0.492),
    ("Qwen2.5-7B", "MMLU-Pro"): ("R3 Fresh deletion", 1.000, 0.424),
    ("Qwen2.5-7B", "TruthfulQA-MC"): ("R3 Fresh deletion", np.nan, 0.700),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper bootstrap CIs from existing item-level artifacts only")
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--output-dir", default="runs/paper_ci_v1")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def resolve_roots(input_root: Path) -> Tuple[Path, Path]:
    pressure_root = input_root / "runs" / "hysteresis_model_suite_v1"
    recovery_root = input_root / "runs" / "recovery_hierarchy" / "recovery_hierarchy_n500"
    if input_root.name == "hysteresis_model_suite_v1":
        pressure_root = input_root
    if input_root.name == "recovery_hierarchy_n500":
        recovery_root = input_root
    return pressure_root, recovery_root


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def require_columns(df: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def model_display(model_key: str) -> str:
    return MODEL_DISPLAY.get(model_key, model_key)


def dataset_display(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def sort_key(row: pd.Series) -> Tuple[int, str, str, str]:
    return (
        PAIR_ORDER.get((str(row.get("model_key", "")), str(row.get("dataset_key", row.get("dataset", "")))), 999),
        str(row.get("condition_or_operation", "")),
        str(row.get("metric", "")),
        str(row.get("threshold_family", "")),
    )


def ci_from_values(values: np.ndarray) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_indices(n: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(samples, n))


def mean_ci(arr: np.ndarray, idx: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(arr, dtype=float)
    point = float(np.nanmean(arr))
    boots = np.nanmean(arr[idx], axis=1)
    low, high = ci_from_values(boots)
    return point, low, high


def metric_row(
    table_name: str,
    model_key: str,
    dataset: str,
    condition: str,
    metric: str,
    estimate: Any,
    ci_low: Any,
    ci_high: Any,
    n_items: int,
    samples: int,
    seed: int,
    informative: bool = True,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "table_name": table_name,
        "model": model_display(model_key),
        "model_key": model_key,
        "dataset": dataset_display(dataset),
        "dataset_key": dataset,
        "condition_or_operation": condition,
        "metric": metric,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_items": n_items,
        "bootstrap_samples": samples,
        "seed": seed,
        "informative": informative,
        "notes": notes,
    }


def pivot_condition_file(path: Path) -> pd.DataFrame:
    df = read_jsonl(path)
    require_columns(df, ["original_index", "condition", "p_advocated_wrong", "p_correct", "predicted_label", "correct_label", "advocated_wrong_label", "model_key", "dataset"], path)
    df = df[df["skipped"].astype(bool) == False].copy()  # noqa: E712
    rows = []
    for item_id, group in df.groupby("original_index", sort=False):
        first = group.iloc[0]
        row: Dict[str, Any] = {
            "item_id": item_id,
            "model_key": str(first["model_key"]),
            "dataset": str(first["dataset"]),
            "correct_label": str(first["correct_label"]),
            "wrong_label": str(first["advocated_wrong_label"]),
        }
        for _, record in group.iterrows():
            condition = str(record["condition"])
            row[f"{condition}_p_wrong"] = float(record["p_advocated_wrong"])
            row[f"{condition}_p_correct"] = float(record["p_correct"])
            row[f"{condition}_predicted_label"] = str(record["predicted_label"])
        rows.append(row)
    return pd.DataFrame(rows)


def add_validation(validation: List[Dict[str, Any]], table: str, model: str, dataset: str, metric: str, value: Any) -> None:
    key = (table, model, dataset, metric)
    if key not in PAPER_VALUES:
        return
    paper_value = PAPER_VALUES[key]
    if value is None or pd.isna(value):
        validation.append({"table_name": table, "model": model, "dataset": dataset, "metric": metric, "paper_value": paper_value, "recomputed_value": np.nan, "absolute_difference": np.nan, "status": "MISSING"})
        return
    diff = abs(float(value) - float(paper_value))
    validation.append({"table_name": table, "model": model, "dataset": dataset, "metric": metric, "paper_value": paper_value, "recomputed_value": float(value), "absolute_difference": diff, "status": "PASS" if diff <= 0.0015 else "FAIL"})


def add_validation_missing(validation: List[Dict[str, Any]], table: str, model: str, dataset: str, metric: str) -> None:
    key = (table, model, dataset, metric)
    if key in PAPER_VALUES:
        validation.append({"table_name": table, "model": model, "dataset": dataset, "metric": metric, "paper_value": PAPER_VALUES[key], "recomputed_value": np.nan, "absolute_difference": np.nan, "status": "MISSING"})


def analyze_pressure(pressure_root: Path, samples: int, seed: int, validation: List[Dict[str, Any]], missing: List[str], used: List[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed + 1)
    for model_key, model, dataset in MODEL_DATASETS:
        path = pressure_root / "pressure_ramp" / model_key / dataset / "per_item_results.jsonl"
        if not path.is_file():
            missing.append(f"Missing pressure-ramp artifact: {path}")
            for metric in ["clean_pw_mean", "press_pw_mean", "reset_pw_mean", "hysteresis_gap", "positive_hysteresis_rate", "lock_in"]:
                add_validation_missing(validation, "pressure_ramp", model, dataset_display(dataset), metric)
            continue
        used.append(str(path))
        df = pivot_condition_file(path)
        required = ["clean_context_p_wrong", "clean_context_probe_p_wrong", "strong_pressure_p_wrong", "strong_reset_p_wrong", "strong_pressure_predicted_label", "strong_reset_predicted_label"]
        require_columns(df, required, path)
        idx = bootstrap_indices(len(df), samples, rng)
        metrics = {
            "clean_pw_mean": df["clean_context_p_wrong"].to_numpy(float),
            "press_pw_mean": df["strong_pressure_p_wrong"].to_numpy(float),
            "reset_pw_mean": df["strong_reset_p_wrong"].to_numpy(float),
            "hysteresis_gap": (df["strong_reset_p_wrong"] - df["clean_context_probe_p_wrong"]).to_numpy(float),
            "positive_hysteresis_rate": (df["strong_reset_p_wrong"] - df["clean_context_probe_p_wrong"] > 0).astype(float).to_numpy(),
        }
        pressure_wrong = (df["strong_pressure_predicted_label"].astype(str) == df["wrong_label"].astype(str)).to_numpy(bool)
        reset_wrong = (df["strong_reset_predicted_label"].astype(str) == df["wrong_label"].astype(str)).to_numpy(bool)
        for metric, arr in metrics.items():
            point, low, high = mean_ci(arr, idx)
            rows.append(metric_row("pressure_ramp", model_key, dataset, "pressure_ramp", metric, point, low, high, len(df), samples, seed))
            add_validation(validation, "pressure_ramp", model, dataset_display(dataset), metric, point)
        denom = pressure_wrong.sum()
        point = float((pressure_wrong & reset_wrong).sum() / denom) if denom else np.nan
        denom_boot = pressure_wrong[idx].sum(axis=1)
        num_boot = (pressure_wrong[idx] & reset_wrong[idx]).sum(axis=1)
        boot = np.divide(num_boot, denom_boot, out=np.full(samples, np.nan), where=denom_boot > 0)
        low, high = ci_from_values(boot)
        rows.append(metric_row("pressure_ramp", model_key, dataset, "pressure_ramp", "lock_in", point, low, high, len(df), samples, seed, informative=bool(denom), notes=f"pressure_wrong_denominator={int(denom)}"))
        add_validation(validation, "pressure_ramp", model, dataset_display(dataset), "lock_in", point)
    return pd.DataFrame(rows)


def analyze_controls(pressure_root: Path, samples: int, seed: int, validation: List[Dict[str, Any]], missing: List[str], used: List[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed + 2)
    for model_key, model, dataset in MODEL_DATASETS:
        paths = {
            "pressure": pressure_root / "pressure_ramp" / model_key / dataset / "per_item_results.jsonl",
            "neutral": pressure_root / "neutral_control" / model_key / dataset / "per_item_results.jsonl",
            "correct": pressure_root / "correct_pressure_control" / model_key / dataset / "per_item_results.jsonl",
        }
        if not all(path.is_file() for path in paths.values()):
            for name, path in paths.items():
                if not path.is_file():
                    missing.append(f"Missing behavioral-control {name} artifact: {path}")
            for metric in ["pressure_gap", "neutral_gap", "correct_reset_pc", "correct_reset_pw"]:
                add_validation_missing(validation, "behavioral_controls", model, dataset_display(dataset), metric)
            continue
        used.extend(str(path) for path in paths.values())
        pressure = pivot_condition_file(paths["pressure"])
        neutral = pivot_condition_file(paths["neutral"])
        correct = pivot_condition_file(paths["correct"])
        merged = pressure[["item_id", "model_key", "dataset", "clean_context_probe_p_wrong", "strong_reset_p_wrong"]].merge(
            neutral[["item_id", "clean_context_probe_p_wrong", "strong_reset_p_wrong"]], on="item_id", suffixes=("_pressure", "_neutral")
        ).merge(correct[["item_id", "strong_reset_p_correct", "strong_reset_p_wrong"]], on="item_id")
        idx = bootstrap_indices(len(merged), samples, rng)
        metrics = {
            "pressure_gap": (merged["strong_reset_p_wrong_pressure"] - merged["clean_context_probe_p_wrong_pressure"]).to_numpy(float),
            "neutral_gap": (merged["strong_reset_p_wrong_neutral"] - merged["clean_context_probe_p_wrong_neutral"]).to_numpy(float),
            "correct_reset_pc": merged["strong_reset_p_correct"].to_numpy(float),
            "correct_reset_pw": merged["strong_reset_p_wrong"].to_numpy(float),
        }
        for metric, arr in metrics.items():
            point, low, high = mean_ci(arr, idx)
            rows.append(metric_row("behavioral_controls", model_key, dataset, "controls", metric, point, low, high, len(merged), samples, seed))
            add_validation(validation, "behavioral_controls", model, dataset_display(dataset), metric, point)
    return pd.DataFrame(rows)


def load_recovery_raw(recovery_root: Path, used: List[str], missing: List[str]) -> pd.DataFrame:
    paths = sorted(recovery_root.glob("*/*/recovery_hierarchy_raw.jsonl"))
    if not paths:
        missing.append(f"Missing recovery hierarchy raw artifacts under {recovery_root}")
        return pd.DataFrame()
    frames = []
    for path in paths:
        used.append(str(path))
        frames.append(read_jsonl(path))
    raw = pd.concat(frames, ignore_index=True)
    require_columns(raw, ["model_key", "dataset", "condition", "item_id", "clean_p_wrong", "ordinary_reset_p_wrong", "intervention_p_wrong", "clean_p_correct", "intervention_p_correct", "clean_correct", "ordinary_reset_correct", "intervention_correct", "ordinary_reset_wrong_follow", "intervention_entropy", "intervention_max_prob", "clean_entropy", "clean_max_prob"], recovery_root)
    return raw


def analyze_decomposition(raw: pd.DataFrame, samples: int, seed: int, validation: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    if raw.empty:
        return pd.DataFrame(rows)
    rng = np.random.default_rng(seed + 3)
    base = raw[raw["condition"] == "ordinary_strong_reset"].drop_duplicates(["model_key", "dataset", "item_id"])
    for model_key, _, dataset in MODEL_DATASETS:
        group = base[(base["model_key"] == model_key) & (base["dataset"] == dataset)].copy()
        model = model_display(model_key)
        dataset_name = dataset_display(dataset)
        if group.empty:
            for metric in ["stable_correct", "history_contamination", "context_correction", "task_failure", "contamination_given_clean_correct"]:
                add_validation_missing(validation, "failure_decomposition", model, dataset_name, metric)
            continue
        idx = bootstrap_indices(len(group), samples, rng)
        clean_correct = group["clean_correct"].astype(bool).to_numpy()
        reset_correct = group["ordinary_reset_correct"].astype(bool).to_numpy()
        reset_wrong = group["ordinary_reset_wrong_follow"].astype(bool).to_numpy()
        metric_arrays = {
            "stable_correct": (clean_correct & reset_correct).astype(float),
            "history_contamination": (clean_correct & ~reset_correct).astype(float),
            "context_correction": (~clean_correct & reset_correct).astype(float),
            "task_failure": (~clean_correct & ~reset_correct).astype(float),
        }
        for metric, arr in metric_arrays.items():
            point, low, high = mean_ci(arr, idx)
            rows.append(metric_row("failure_decomposition", model_key, dataset, "ordinary_reset", metric, point, low, high, len(group), samples, seed))
            add_validation(validation, "failure_decomposition", model, dataset_name, metric, point)
        denom = clean_correct.sum()
        point = float((clean_correct & reset_wrong).sum() / denom) if denom else np.nan
        denom_boot = clean_correct[idx].sum(axis=1)
        num_boot = (clean_correct[idx] & reset_wrong[idx]).sum(axis=1)
        boot = np.divide(num_boot, denom_boot, out=np.full(samples, np.nan), where=denom_boot > 0)
        low, high = ci_from_values(boot)
        rows.append(metric_row("failure_decomposition", model_key, dataset, "ordinary_reset", "contamination_given_clean_correct", point, low, high, len(group), samples, seed, informative=bool(denom), notes=f"clean_correct_denominator={int(denom)}"))
        add_validation(validation, "failure_decomposition", model, dataset_name, "contamination_given_clean_correct", point)
    return pd.DataFrame(rows)


def clean_like_from_values(values: Dict[str, float], threshold: ThresholdFamily) -> bool:
    return bool(
        abs(values["hysteresis_gap"]) <= threshold.gap
        and values["accuracy"] >= values["clean_accuracy"] - threshold.accuracy_slack
        and values["p_correct"] >= values["clean_p_correct"] - threshold.p_correct_slack
        and values["entropy"] <= values["clean_entropy"] + threshold.entropy_slack
        and values["max_probability"] >= values["clean_max_probability"] - threshold.max_prob_slack
    )


def classify_recovery(values: Dict[str, float]) -> str:
    if clean_like_from_values(values, THRESHOLDS["main"]):
        return "restored_clean_context_like"
    if values["condition"] == "ordinary_strong_reset":
        return "ordinary_reset_baseline"
    wrong_drop = values["ordinary_reset_p_wrong"] - values["p_wrong"]
    correct_gain = values["p_correct"] - values["ordinary_reset_p_correct"]
    accuracy_gain = values["accuracy"] - values["ordinary_reset_accuracy"]
    entropy_gain = values["entropy"] - values["ordinary_reset_entropy"]
    max_drop = values["ordinary_reset_max_probability"] - values["max_probability"]
    if wrong_drop >= 0.05 and correct_gain >= 0.03 and accuracy_gain >= -0.02 and entropy_gain <= 0.25 and max_drop <= 0.15:
        return "genuine_repair_like"
    if wrong_drop >= 0.05 and (correct_gain < 0.03 or accuracy_gain < -0.02 or entropy_gain > 0.25 or max_drop > 0.15):
        return "flattening_or_suppression"
    if wrong_drop >= 0.02:
        return "weak_wrong_suppression"
    if wrong_drop <= -0.02:
        return "worse_than_reset"
    return "no_material_change"


def recovery_point_values(group: pd.DataFrame) -> Dict[str, float]:
    clean_p_wrong = float(group["clean_p_wrong"].mean())
    reset_p_wrong = float(group["ordinary_reset_p_wrong"].mean())
    p_wrong = float(group["intervention_p_wrong"].mean())
    gap = p_wrong - clean_p_wrong
    reset_gap = reset_p_wrong - clean_p_wrong
    return {
        "condition": str(group["condition"].iloc[0]),
        "p_wrong": p_wrong,
        "p_correct": float(group["intervention_p_correct"].mean()),
        "accuracy": float(group["intervention_correct"].astype(bool).mean()),
        "hysteresis_gap": gap,
        "entropy": float(group["intervention_entropy"].mean()),
        "max_probability": float(group["intervention_max_prob"].mean()),
        "restoration_ratio": float((reset_p_wrong - p_wrong) / reset_gap) if abs(reset_gap) > 1e-12 else np.nan,
        "restoration_ratio_informative": abs(reset_gap) >= 0.05,
        "clean_accuracy": float(group["clean_correct"].astype(bool).mean()),
        "clean_p_correct": float(group["clean_p_correct"].mean()),
        "clean_entropy": float(group["clean_entropy"].mean()),
        "clean_max_probability": float(group["clean_max_prob"].mean()),
        "ordinary_reset_p_wrong": reset_p_wrong,
        "ordinary_reset_p_correct": float(group["ordinary_reset_p_correct"].mean()),
        "ordinary_reset_accuracy": float(group["ordinary_reset_correct"].astype(bool).mean()),
        "ordinary_reset_entropy": float(group["ordinary_reset_entropy"].mean()),
        "ordinary_reset_max_probability": float(group["ordinary_reset_max_prob"].mean()),
        "ordinary_reset_gap": reset_gap,
    }


def analyze_recovery(raw: pd.DataFrame, samples: int, seed: int, validation: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    operation_rows = []
    minimal_rows = []
    summary_validation_rows = []
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed + 4)
    main = raw[raw["condition"].isin(MAIN_CONDITIONS)].copy()
    point_by_pair_condition: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for model_key, _, dataset in MODEL_DATASETS:
        for condition in MAIN_CONDITIONS:
            group = main[(main["model_key"] == model_key) & (main["dataset"] == dataset) & (main["condition"] == condition)].copy()
            if group.empty:
                continue
            idx = bootstrap_indices(len(group), samples, rng)
            values = recovery_point_values(group)
            point_by_pair_condition[(model_key, dataset, condition)] = values
            metric_arrays = {
                "p_wrong": group["intervention_p_wrong"].to_numpy(float),
                "p_correct": group["intervention_p_correct"].to_numpy(float),
                "accuracy": group["intervention_correct"].astype(bool).astype(float).to_numpy(),
                "hysteresis_gap": (group["intervention_p_wrong"] - group["clean_p_wrong"]).to_numpy(float),
                "entropy": group["intervention_entropy"].to_numpy(float),
                "max_probability": group["intervention_max_prob"].to_numpy(float),
            }
            for metric, arr in metric_arrays.items():
                point, low, high = mean_ci(arr, idx)
                operation_rows.append(metric_row("recovery_operations", model_key, dataset, CONDITION_LABELS[condition], metric, point, low, high, len(group), samples, seed))
            if values["restoration_ratio_informative"]:
                reset = group["ordinary_reset_p_wrong"].to_numpy(float)
                clean = group["clean_p_wrong"].to_numpy(float)
                intervention = group["intervention_p_wrong"].to_numpy(float)
                denom = reset[idx].mean(axis=1) - clean[idx].mean(axis=1)
                boot = np.divide(reset[idx].mean(axis=1) - intervention[idx].mean(axis=1), denom, out=np.full(samples, np.nan), where=np.abs(denom) > 1e-12)
                low, high = ci_from_values(boot)
                estimate = values["restoration_ratio"]
            else:
                low = high = estimate = np.nan
            operation_rows.append(metric_row("recovery_operations", model_key, dataset, CONDITION_LABELS[condition], "restoration_ratio", estimate, low, high, len(group), samples, seed, informative=bool(values["restoration_ratio_informative"]), notes="NA when ordinary-reset gap < 0.05" if not values["restoration_ratio_informative"] else ""))
            for name, threshold in THRESHOLDS.items():
                flag = clean_like_from_values(values, threshold)
                operation_rows.append(metric_row("recovery_operations", model_key, dataset, CONDITION_LABELS[condition], f"clean_context_like_{name}", float(flag), np.nan, np.nan, len(group), samples, seed, informative=True, notes="Deterministic aggregate classification; CI not bootstrapped"))

    for model_key, model, dataset in MODEL_DATASETS:
        selected: Optional[str] = None
        for condition in MAIN_CONDITIONS:
            values = point_by_pair_condition.get((model_key, dataset, condition))
            if values and clean_like_from_values(values, THRESHOLDS["main"]):
                selected = condition
                break
        if selected is None:
            minimal_rows.append(metric_row("minimal_recovery_operation", model_key, dataset, "not_restored", "accuracy", np.nan, np.nan, np.nan, 0, samples, seed, informative=False, notes="No main operation met clean-context-like criteria"))
            continue
        group = main[(main["model_key"] == model_key) & (main["dataset"] == dataset) & (main["condition"] == selected)].copy()
        idx = bootstrap_indices(len(group), samples, rng)
        accuracy = group["intervention_correct"].astype(bool).astype(float).to_numpy()
        point, low, high = mean_ci(accuracy, idx)
        minimal_rows.append(metric_row("minimal_recovery_operation", model_key, dataset, CONDITION_LABELS[selected], "accuracy", point, low, high, len(group), samples, seed))
        values = point_by_pair_condition[(model_key, dataset, selected)]
        if values["restoration_ratio_informative"]:
            reset = group["ordinary_reset_p_wrong"].to_numpy(float)
            clean = group["clean_p_wrong"].to_numpy(float)
            intervention = group["intervention_p_wrong"].to_numpy(float)
            denom = reset[idx].mean(axis=1) - clean[idx].mean(axis=1)
            boot = np.divide(reset[idx].mean(axis=1) - intervention[idx].mean(axis=1), denom, out=np.full(samples, np.nan), where=np.abs(denom) > 1e-12)
            low, high = ci_from_values(boot)
            estimate = values["restoration_ratio"]
            informative = True
        else:
            estimate = low = high = np.nan
            informative = False
        minimal_rows.append(metric_row("minimal_recovery_operation", model_key, dataset, CONDITION_LABELS[selected], "restoration_ratio", estimate, low, high, len(group), samples, seed, informative=informative, notes="NA when ordinary-reset gap < 0.05" if not informative else ""))
        minimal_rows.append(metric_row("minimal_recovery_operation", model_key, dataset, CONDITION_LABELS[selected], "hysteresis_gap", values["hysteresis_gap"], np.nan, np.nan, len(group), samples, seed, notes="Point estimate only; CI available in recovery_operations_with_ci.csv"))
        expected = MRO_EXPECTED.get((model, dataset_display(dataset)))
        if expected:
            expected_operation, expected_rest, expected_acc = expected
            validation.append({"table_name": "minimal_recovery_operation", "model": model, "dataset": dataset_display(dataset), "metric": "operation", "paper_value": expected_operation, "recomputed_value": CONDITION_LABELS[selected], "absolute_difference": 0 if expected_operation == CONDITION_LABELS[selected] else np.nan, "status": "PASS" if expected_operation == CONDITION_LABELS[selected] else "FAIL"})
            validation.append({"table_name": "minimal_recovery_operation", "model": model, "dataset": dataset_display(dataset), "metric": "accuracy", "paper_value": expected_acc, "recomputed_value": float(point), "absolute_difference": abs(float(point) - expected_acc), "status": "PASS" if abs(float(point) - expected_acc) <= 0.0015 else "FAIL"})
            rest_value = np.nan if not values["restoration_ratio_informative"] else values["restoration_ratio"]
            if pd.isna(expected_rest) and pd.isna(rest_value):
                status = "PASS"
                diff = np.nan
            elif pd.isna(expected_rest) or pd.isna(rest_value):
                status = "FAIL"
                diff = np.nan
            else:
                diff = abs(float(rest_value) - float(expected_rest))
                status = "PASS" if diff <= 0.0015 else "FAIL"
            validation.append({"table_name": "minimal_recovery_operation", "model": model, "dataset": dataset_display(dataset), "metric": "restoration_ratio", "paper_value": expected_rest, "recomputed_value": rest_value, "absolute_difference": diff, "status": status})

    for condition, expected in RECOVERY_SUMMARY_EXPECTED.items():
        condition_values = [point_by_pair_condition[(model_key, dataset, condition)] for model_key, _, dataset in MODEL_DATASETS if (model_key, dataset, condition) in point_by_pair_condition]
        classifications = [classify_recovery(values) for values in condition_values]
        clean_like = int(sum(label == "restored_clean_context_like" for label in classifications))
        genuine = int(sum(label == "genuine_repair_like" for label in classifications))
        weak_mixed_worse = int(len(classifications) - clean_like - genuine)
        summary_validation_rows.append({"condition": condition, "clean_like": clean_like, "genuine": genuine, "weak_mixed_worse": weak_mixed_worse})
        for metric, recomputed, paper in [
            ("clean_like_count", clean_like, expected[0]),
            ("genuine_count", genuine, expected[1]),
            ("weak_mixed_worse_count", weak_mixed_worse, expected[2]),
        ]:
            status = "PASS" if abs(float(recomputed) - paper) <= 0.0015 else "FAIL"
            validation.append({"table_name": "recovery_hierarchy_summary", "model": "ALL", "dataset": CONDITION_LABELS[condition], "metric": metric, "paper_value": paper, "recomputed_value": recomputed, "absolute_difference": np.nan if pd.isna(recomputed) else abs(float(recomputed) - paper), "status": status})
    return pd.DataFrame(operation_rows), pd.DataFrame(minimal_rows), pd.DataFrame(summary_validation_rows)


def analyze_relabel(pressure_root: Path, samples: int, seed: int, validation: List[Dict[str, Any]], missing: List[str], used: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows = []
    semantic_rows = []
    rng = np.random.default_rng(seed + 5)
    summary_path = pressure_root / "aggregate" / "relabel_model_summary.csv"
    relabel_summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    if summary_path.is_file():
        used.append(str(summary_path))
    else:
        missing.append(f"Missing relabel aggregate summary for chance rates: {summary_path}")
    for model_key, model, dataset in MODEL_DATASETS:
        path = pressure_root / "relabel_intervention" / model_key / dataset / "per_item_results.jsonl"
        if not path.is_file():
            missing.append(f"Missing relabel item artifact: {path}")
            for table, metrics in [("relabel_effect", ["baseline_pw", "shuffle_pw", "wrong_reduction"]), ("semantic_letter", ["semantic_following_rate", "old_letter_following_rate", "chance_old_letter_rate", "new_wrong_following_rate"] )]:
                for metric in metrics:
                    add_validation_missing(validation, table, model, dataset_display(dataset), metric)
            continue
        used.append(str(path))
        df = read_jsonl(path)
        require_columns(df, ["original_index", "condition", "p_advocated_wrong", "semantic_wrong_follow", "old_letter_follow", "new_wrong_letter_follow", "model_key", "dataset"], path)
        df = df[df["skipped"].astype(bool) == False].copy()  # noqa: E712
        pivot = df.pivot_table(index="original_index", columns="condition", values="p_advocated_wrong", aggfunc="first")
        if "baseline_strong_reset" not in pivot.columns or "option_relabel_shuffle_reset" not in pivot.columns:
            missing.append(f"Relabel file missing baseline/shuffle conditions: {path}")
            continue
        idx = bootstrap_indices(len(pivot), samples, rng)
        baseline = pivot["baseline_strong_reset"].astype(float).to_numpy()
        shuffle = pivot["option_relabel_shuffle_reset"].astype(float).to_numpy()
        metrics = {"baseline_pw": baseline, "shuffle_pw": shuffle, "wrong_reduction": baseline - shuffle}
        for metric, arr in metrics.items():
            point, low, high = mean_ci(arr, idx)
            effect_rows.append(metric_row("relabel_effect", model_key, dataset, "option_relabel_shuffle_reset", metric, point, low, high, len(pivot), samples, seed))
            add_validation(validation, "relabel_effect", model, dataset_display(dataset), metric, point)

        shuffle_rows = df[df["condition"] == "option_relabel_shuffle_reset"].drop_duplicates("original_index")
        idx2 = bootstrap_indices(len(shuffle_rows), samples, rng)
        empirical_metrics = {
            "semantic_following_rate": shuffle_rows["semantic_wrong_follow"].astype(bool).astype(float).to_numpy(),
            "old_letter_following_rate": shuffle_rows["old_letter_follow"].astype(bool).astype(float).to_numpy(),
            "new_wrong_following_rate": shuffle_rows["new_wrong_letter_follow"].astype(bool).astype(float).to_numpy(),
        }
        for metric, arr in empirical_metrics.items():
            point, low, high = mean_ci(arr, idx2)
            semantic_rows.append(metric_row("semantic_letter", model_key, dataset, "option_relabel_shuffle_reset", metric, point, low, high, len(shuffle_rows), samples, seed))
            add_validation(validation, "semantic_letter", model, dataset_display(dataset), metric, point)
        chance = np.nan
        if not relabel_summary.empty:
            chance_rows = relabel_summary[(relabel_summary["model_key"] == model_key) & (relabel_summary["dataset"] == dataset) & (relabel_summary["condition"] == "option_relabel_shuffle_reset")]
            if not chance_rows.empty and "old_letter_chance_rate" in chance_rows.columns:
                chance = float(chance_rows.iloc[0]["old_letter_chance_rate"])
        semantic_rows.append(metric_row("semantic_letter", model_key, dataset, "option_relabel_shuffle_reset", "chance_old_letter_rate", chance, np.nan, np.nan, len(shuffle_rows), samples, seed, informative=False, notes="Deterministic expected value from relabel aggregate; CI not meaningful"))
        add_validation(validation, "semantic_letter", model, dataset_display(dataset), "chance_old_letter_rate", chance)
    return pd.DataFrame(effect_rows), pd.DataFrame(semantic_rows)


def format_ci(row: pd.Series) -> str:
    estimate = row.get("estimate")
    if pd.isna(estimate):
        return "--"
    if not bool(row.get("informative", True)) or pd.isna(row.get("ci_low")) or pd.isna(row.get("ci_high")):
        return f"{float(estimate):.3f}"
    return f"{float(estimate):.3f} [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"


def table_wide(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    sub = df[df["table_name"] == table_name].copy()
    sub["value"] = sub.apply(format_ci, axis=1)
    return sub.pivot_table(index=["model", "dataset", "condition_or_operation"], columns="metric", values="value", aggfunc="first").reset_index()


def write_latex_outputs(output_dir: Path, pressure: pd.DataFrame, controls: pd.DataFrame, decomp: pd.DataFrame, recovery: pd.DataFrame, minimal: pd.DataFrame, relabel: pd.DataFrame, semantic: pd.DataFrame) -> None:
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("pressure_ramp_with_ci.tex", table_wide(pressure, "pressure_ramp"), "Pressure-ramp metrics with paired bootstrap 95\\% CIs.", "tab:pressure-ramp-ci"),
        ("behavioral_controls_with_ci.tex", table_wide(controls, "behavioral_controls"), "Behavioral controls with paired bootstrap 95\\% CIs.", "tab:behavioral-controls-ci"),
        ("failure_decomposition_with_ci.tex", table_wide(decomp, "failure_decomposition"), "Failure decomposition with paired bootstrap 95\\% CIs.", "tab:failure-decomposition-ci"),
        ("recovery_operations_with_ci.tex", table_wide(recovery[recovery["metric"].isin(["p_wrong", "p_correct", "accuracy", "hysteresis_gap", "restoration_ratio"])], "recovery_operations"), "Recovery operations with paired bootstrap 95\\% CIs.", "tab:recovery-operations-ci"),
        ("minimal_recovery_operation_with_ci.tex", table_wide(minimal, "minimal_recovery_operation"), "Minimal recovery operation with paired bootstrap 95\\% CIs.", "tab:minimal-recovery-ci"),
        ("relabel_effect_with_ci.tex", table_wide(relabel, "relabel_effect"), "Relabel effect with paired bootstrap 95\\% CIs.", "tab:relabel-effect-ci"),
        ("semantic_letter_with_ci.tex", table_wide(semantic, "semantic_letter"), "Semantic-letter diagnostic with paired bootstrap 95\\% CIs.", "tab:semantic-letter-ci"),
    ]
    for filename, table, caption, label in specs:
        table.to_latex(latex_dir / filename, index=False, escape=True, caption=caption, label=label)
        compact = table.copy()
        for column in compact.columns:
            if column not in {"model", "dataset", "condition_or_operation"}:
                compact[column] = compact[column].astype(str).str.replace(r" \[.*\]", "", regex=True)
        compact_name = filename.replace("_with_ci.tex", "_compact.tex")
        compact.to_latex(latex_dir / compact_name, index=False, escape=True, caption=caption.replace(" with paired bootstrap 95\\% CIs", " point estimates"), label=label.replace("-ci", "-compact"))


def validate_remaining(validation: List[Dict[str, Any]]) -> pd.DataFrame:
    seen = {(row["table_name"], row["model"], row["dataset"], row["metric"]) for row in validation}
    for table, model, dataset, metric in PAPER_VALUES:
        if (table, model, dataset, metric) not in seen:
            validation.append({"table_name": table, "model": model, "dataset": dataset, "metric": metric, "paper_value": PAPER_VALUES[(table, model, dataset, metric)], "recomputed_value": np.nan, "absolute_difference": np.nan, "status": "MISSING"})
    return pd.DataFrame(validation).sort_values(["table_name", "model", "dataset", "metric"])


def validate_threshold_sensitivity(path: Path, validation: List[Dict[str, Any]], missing: List[str]) -> None:
    if not path.is_file():
        return
    df = pd.read_csv(path)
    columns = {
        "r0": "r0_ordinary_reset_recovered",
        "r1": "r1_user_retraction_recovered",
        "r2": "r2_system_reset_recovered",
        "r3_fresh": "r3_fresh_deletion_recovered",
        "r3_truncation": "r3_context_truncation_recovered",
        "r4": "r4_neutral_summary_recovered",
        "r5": "r5_factual_reconstruction_recovered",
        "r6": "r6_self_verification_recovered",
    }
    for family, expected in THRESHOLD_EXPECTED.items():
        rows = df[df["threshold_family"] == family]
        if rows.empty:
            missing.append(f"Missing threshold sensitivity family in {path}: {family}")
            continue
        record = rows.iloc[0]
        for metric, column in columns.items():
            value = float(record[column])
            paper = float(expected[metric])
            diff = abs(value - paper)
            validation.append({"table_name": "threshold_sensitivity", "model": "ALL", "dataset": family, "metric": metric, "paper_value": paper, "recomputed_value": value, "absolute_difference": diff, "status": "PASS" if diff <= 0.0015 else "FAIL"})


def write_missing_report(path: Path, missing: Sequence[str]) -> None:
    lines = ["# Missing Artifacts", ""]
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("No required item-level artifacts were missing for the computed tables.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ci_report(path: Path, used: Sequence[str], command: str, samples: int, seed: int, validation: pd.DataFrame, missing: Sequence[str]) -> None:
    counts = validation["status"].value_counts().to_dict() if not validation.empty else {}
    failures = validation[validation["status"] == "FAIL"] if not validation.empty else pd.DataFrame()
    miss = validation[validation["status"] == "MISSING"] if not validation.empty else pd.DataFrame()
    lines = [
        "# Paper CI Report",
        "",
        "This run computed paired bootstrap confidence intervals from existing item-level artifacts only. No model inference, model API calls, or dataset regeneration were performed.",
        "",
        "## Command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Bootstrap Settings",
        "",
        f"- unit: item ID",
        f"- bootstrap samples: {samples}",
        f"- seed: {seed}",
        "- interval: percentile 95% CI [2.5%, 97.5%]",
        "- condition pairing: all condition values for a sampled item are resampled together",
        "",
        "## Artifact Files Used",
        "",
    ]
    lines.extend(f"- `{item}`" for item in sorted(set(used)))
    lines.extend([
        "",
        "## Validation Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        f"- MISSING: {counts.get('MISSING', 0)}",
        "",
    ])
    if not failures.empty:
        lines.append("## Mismatches Against Paper Values")
        lines.append("")
        for _, row in failures.iterrows():
            lines.append(f"- {row['table_name']} / {row['model']} / {row['dataset']} / {row['metric']}: paper={row['paper_value']}, recomputed={row['recomputed_value']}, diff={row['absolute_difference']}")
        lines.append("")
    if not miss.empty:
        lines.append("## Validation Values Missing From Item-Level Recompute")
        lines.append("")
        for _, row in miss.iterrows():
            lines.append(f"- {row['table_name']} / {row['model']} / {row['dataset']} / {row['metric']}")
        lines.append("")
    lines.extend([
        "## Missing Item-Level Artifacts",
        "",
    ])
    lines.extend((f"- {item}" for item in missing),)
    if not missing:
        lines.append("No required item-level artifacts were missing for the computed tables.")
    lines.extend([
        "",
        "## Notes On Metrics Without Meaningful CIs",
        "",
        "- Semantic-letter chance rates are deterministic expected values from the relabel aggregate, not observed Bernoulli outcomes; CIs are reported as NA.",
        "- Restoration ratios are marked non-informative and reported as NA when the ordinary-reset hysteresis gap is below 0.05.",
        "- Clean-context-like flags are deterministic aggregate classifications; the report does not bootstrap classifier-count uncertainty.",
        "",
        "## Recommended Paper Updates",
        "",
        "Update the pressure-ramp, behavioral-control, failure-decomposition, recovery-operation, minimal-recovery, relabel-effect, and semantic-letter tables from `latex_tables/`. Use compact point-estimate snippets in the main paper if space is tight and full CI snippets in the appendix.",
        "",
        "## Experimental Setup Paragraph",
        "",
        "Confidence intervals were computed offline from item-level artifacts using paired bootstrap resampling over evaluation items. For each model-dataset pair, item IDs were sampled with replacement and all condition measurements for a sampled item were kept paired across conditions. Each reported interval is the percentile 95% interval over 10,000 bootstrap replicates with seed 12345. No model inference or dataset regeneration was performed for CI computation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pressure_root, recovery_root = resolve_roots(input_root)
    validation: List[Dict[str, Any]] = []
    missing: List[str] = []
    used: List[str] = []

    pressure = analyze_pressure(pressure_root, args.bootstrap_samples, args.seed, validation, missing, used)
    controls = analyze_controls(pressure_root, args.bootstrap_samples, args.seed, validation, missing, used)
    raw = load_recovery_raw(recovery_root, used, missing)
    decomposition = analyze_decomposition(raw, args.bootstrap_samples, args.seed, validation)
    recovery_ops, minimal, _ = analyze_recovery(raw, args.bootstrap_samples, args.seed, validation)
    relabel, semantic = analyze_relabel(pressure_root, args.bootstrap_samples, args.seed, validation, missing, used)

    threshold_src = recovery_root / "recovery_threshold_sensitivity.csv"
    threshold_dst = output_dir / "recovery_threshold_sensitivity.csv"
    if threshold_src.is_file():
        shutil.copyfile(threshold_src, threshold_dst)
        used.append(str(threshold_src))
        validate_threshold_sensitivity(threshold_dst, validation, missing)
    else:
        missing.append(f"Missing threshold sensitivity artifact: {threshold_src}")
        pd.DataFrame().to_csv(threshold_dst, index=False)

    outputs = [
        (pressure, "pressure_ramp_with_ci.csv"),
        (controls, "behavioral_controls_with_ci.csv"),
        (decomposition, "failure_decomposition_with_ci.csv"),
        (recovery_ops, "recovery_operations_with_ci.csv"),
        (minimal, "minimal_recovery_operation_with_ci.csv"),
        (relabel, "relabel_effect_with_ci.csv"),
        (semantic, "semantic_letter_with_ci.csv"),
    ]
    for df, filename in outputs:
        if not df.empty:
            df = df.sort_values(by=list(df.columns[: min(5, len(df.columns))]))
        df.to_csv(output_dir / filename, index=False)

    validation_df = validate_remaining(validation)
    validation_df.to_csv(output_dir / "validation_against_paper.csv", index=False)
    write_missing_report(output_dir / "missing_artifacts.md", missing)
    write_latex_outputs(output_dir, pressure, controls, decomposition, recovery_ops, minimal, relabel, semantic)
    command = f"python src/original_research_scripts/paper_bootstrap_cis.py --input-root {args.input_root} --output-dir {args.output_dir} --bootstrap-samples {args.bootstrap_samples} --seed {args.seed}"
    write_ci_report(output_dir / "ci_report.md", used, command, args.bootstrap_samples, args.seed, validation_df, missing)
    print(f"Wrote paper CI outputs to {output_dir}")
    print(validation_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()