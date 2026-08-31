#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from paper_readiness_common_v1 import (
    CCTP_ROOT,
    HIERARCHY_FAMILIES,
    HIERARCHY_ROOT,
    PRESSURE_SUITE_ROOT,
    TRUNCATION_ROOT,
    add_ci_row,
    bootstrap_mean,
    ci_from_bootstrap,
    hierarchy_counts,
    load_cctp_items,
    load_hierarchy_raw,
    load_hierarchy_summary,
    load_pressure_items,
    load_truncation_summary,
    markdown_table,
    output_root_from_args,
    parser_with_output,
    rel,
    write_dataframe,
    write_json,
)


def paired_group_indices(n: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(samples, n))


def pressure_cis(samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 11)
    data = load_pressure_items()
    rows: List[Dict[str, Any]] = []
    source = PRESSURE_SUITE_ROOT
    for (model_key, dataset), group in data.groupby(["model_key", "dataset"], dropna=False):
        n = len(group)
        idx = paired_group_indices(n, samples, rng)
        arrays = {
            "clean_advocated_wrong_probability": group["clean_p_wrong"].to_numpy(float),
            "pressure_advocated_wrong_probability": group["strong_pressure_p_wrong"].to_numpy(float),
            "reset_advocated_wrong_probability": group["strong_reset_p_wrong"].to_numpy(float),
            "hysteresis_gap": (group["strong_reset_p_wrong"] - group["clean_p_wrong"]).to_numpy(float),
            "positive_hysteresis_rate": group["positive_hysteresis_vs_initial_clean"].astype(bool).astype(float).to_numpy(),
            "conditional_lock_in": group["lock_in"].astype(bool).astype(float).to_numpy(),
        }
        for metric, values in arrays.items():
            boot = values[idx].mean(axis=1)
            low, high = ci_from_bootstrap(boot)
            add_ci_row(rows, "pressure_ramp", {"model_key": model_key, "dataset": dataset}, metric, float(values.mean()), low, high, n, samples, seed, source)
    return pd.DataFrame(rows)


def failure_cis(samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 22)
    raw = load_hierarchy_raw()
    base = raw[raw["condition"].eq("ordinary_strong_reset")].drop_duplicates(["model_key", "dataset", "item_id"])
    rows: List[Dict[str, Any]] = []
    source = HIERARCHY_ROOT / "recovery_hierarchy_raw.jsonl"
    for (model_key, dataset), group in base.groupby(["model_key", "dataset"], dropna=False):
        clean = group["clean_correct"].astype(bool).to_numpy()
        reset = group["ordinary_reset_correct"].astype(bool).to_numpy()
        reset_wrong = group["ordinary_reset_wrong_follow"].astype(bool).to_numpy()
        arrays = {
            "stable_correct": (clean & reset).astype(float),
            "history_contamination": (clean & ~reset).astype(float),
            "context_induced_correction": (~clean & reset).astype(float),
            "task_failure": (~clean & ~reset).astype(float),
        }
        n = len(group)
        idx = paired_group_indices(n, samples, rng)
        for metric, values in arrays.items():
            boot = values[idx].mean(axis=1)
            low, high = ci_from_bootstrap(boot)
            add_ci_row(rows, "history_contamination", {"model_key": model_key, "dataset": dataset}, metric, float(values.mean()), low, high, n, samples, seed, source)
        denom = clean.sum()
        estimate = float((clean & reset_wrong).sum() / denom) if denom else np.nan
        denom_boot = clean[idx].sum(axis=1)
        num_boot = (clean[idx] & reset_wrong[idx]).sum(axis=1)
        boot = np.divide(num_boot, denom_boot, out=np.full(samples, np.nan), where=denom_boot > 0)
        low, high = ci_from_bootstrap(boot)
        add_ci_row(rows, "history_contamination", {"model_key": model_key, "dataset": dataset}, "contamination_conditioned_on_clean_correct", estimate, low, high, n, samples, seed, source, notes="Conditional denominator recomputed in each bootstrap sample.")
    return pd.DataFrame(rows)


def recovery_cis(samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 33)
    raw = load_hierarchy_raw()
    summary = load_hierarchy_summary()
    rows: List[Dict[str, Any]] = []
    source = HIERARCHY_ROOT / "recovery_hierarchy_raw.jsonl"
    for (model_key, dataset, condition), group in raw[raw["is_main_condition"].astype(bool)].groupby(["model_key", "dataset", "condition"], dropna=False):
        n = len(group)
        idx = paired_group_indices(n, samples, rng)
        metric_arrays = {
            "intervention_p_wrong": group["intervention_p_wrong"].to_numpy(float),
            "intervention_p_correct": group["intervention_p_correct"].to_numpy(float),
            "accuracy": group["intervention_correct"].astype(bool).astype(float).to_numpy(),
            "hysteresis_gap": (group["intervention_p_wrong"] - group["clean_p_wrong"]).to_numpy(float),
            "entropy": group["intervention_entropy"].to_numpy(float),
            "max_probability": group["intervention_max_prob"].to_numpy(float),
        }
        for metric, values in metric_arrays.items():
            boot = values[idx].mean(axis=1)
            low, high = ci_from_bootstrap(boot)
            add_ci_row(rows, "recovery_hierarchy", {"model_key": model_key, "dataset": dataset, "condition": condition}, metric, float(values.mean()), low, high, n, samples, seed, source)
        reset = group["ordinary_reset_p_wrong"].to_numpy(float)
        clean = group["clean_p_wrong"].to_numpy(float)
        intervention = group["intervention_p_wrong"].to_numpy(float)
        denom = reset[idx].mean(axis=1) - clean[idx].mean(axis=1)
        boot = np.divide(reset[idx].mean(axis=1) - intervention[idx].mean(axis=1), denom, out=np.full(samples, np.nan), where=np.abs(denom) >= 0.05)
        low, high = ci_from_bootstrap(boot)
        estimate = (reset.mean() - intervention.mean()) / (reset.mean() - clean.mean()) if abs(reset.mean() - clean.mean()) >= 0.05 else np.nan
        add_ci_row(rows, "recovery_hierarchy", {"model_key": model_key, "dataset": dataset, "condition": condition}, "restoration_ratio", float(estimate), low, high, n, samples, seed, source, notes="Conditional denominator recomputed in each bootstrap sample; NA when ordinary-reset gap < 0.05.")
    counts = hierarchy_counts(summary, HIERARCHY_FAMILIES["main"])
    for condition, count in counts.items():
        add_ci_row(rows, "recovery_hierarchy", {"model_key": "ALL", "dataset": "ALL", "condition": condition}, "recovered_pair_count_descriptive", float(count), np.nan, np.nan, int(summary[["model_key", "dataset"]].drop_duplicates().shape[0]), samples, seed, HIERARCHY_ROOT / "recovery_hierarchy_summary.csv", notes="Descriptive deterministic count from aggregate criterion; not bootstrapped.")
    return pd.DataFrame(rows)


def truncation_cis(samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 44)
    summary = load_truncation_summary()
    rows: List[Dict[str, Any]] = []
    source = TRUNCATION_ROOT / "expanded_truncation_summary.csv"
    condition_map = {
        "full_history_reset": "full_history_gap",
        "remove_strong_only": "drop_strong_gap",
        "remove_medium_strong": "drop_medium_strong_gap",
        "remove_all_pressure": "drop_all_pressure_gap",
        "reset_only": "reset_only_gap",
        "fresh_context": "fresh_gap",
    }
    for condition, metric in condition_map.items():
        values = summary[summary["condition"].eq(condition)]["hysteresis_gap_mean"].to_numpy(float)
        estimate, low, high = bootstrap_mean(values, samples, rng)
        add_ci_row(rows, "truncation_boundary", {"condition": condition}, metric, estimate, low, high, len(values), samples, seed, source, notes="Bootstrap over model-dataset aggregate rows.")
    pivot = summary.pivot_table(index=["model", "dataset"], columns="condition", values="hysteresis_gap_mean", aggfunc="first")
    reductions = {
        "full_minus_drop_all_pressure": pivot.get("full_history_reset") - pivot.get("remove_all_pressure"),
        "full_minus_reset_only": pivot.get("full_history_reset") - pivot.get("reset_only"),
        "drop_medium_strong_minus_drop_all_pressure": pivot.get("remove_medium_strong") - pivot.get("remove_all_pressure"),
    }
    for metric, series in reductions.items():
        values = series.dropna().to_numpy(float)
        estimate, low, high = bootstrap_mean(values, samples, rng)
        add_ci_row(rows, "truncation_boundary", {"condition": "pairwise_reduction"}, metric, estimate, low, high, len(values), samples, seed, source, notes="Bootstrap over model-dataset aggregate rows.")
    return pd.DataFrame(rows)


def cctp_cis(samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 55)
    item = load_cctp_items()
    primary = item[(item["condition"].eq("cctp_logit_linear_min_alpha")) & (item["pressure_type"].eq("wrong_pressure"))]
    rows: List[Dict[str, Any]] = []
    source = CCTP_ROOT / "combined_item_level.csv"
    for dataset, group in primary.groupby("dataset", dropna=False):
        for metric, column in [("mean_alpha_star", "alpha_star"), ("mean_history_retention", "history_retention")]:
            estimate, low, high = bootstrap_mean(group[column].to_numpy(float), samples, rng)
            add_ci_row(rows, "cctp_alpha", {"dataset": dataset, "model_key": "ALL"}, metric, estimate, low, high, len(group), samples, seed, source)
    dataset_groups = {dataset: group["alpha_star"].to_numpy(float) for dataset, group in primary.groupby("dataset", dropna=False)}
    if {"mmlu_pro", "truthfulqa_mc"} <= set(dataset_groups):
        m = dataset_groups["mmlu_pro"]
        t = dataset_groups["truthfulqa_mc"]
        m_idx = rng.integers(0, m.size, size=(samples, m.size))
        t_idx = rng.integers(0, t.size, size=(samples, t.size))
        boot = m[m_idx].mean(axis=1) - t[t_idx].mean(axis=1)
        low, high = ci_from_bootstrap(boot)
        add_ci_row(rows, "cctp_alpha", {"dataset": "mmlu_minus_truthfulqa", "model_key": "ALL"}, "mean_alpha_difference", float(m.mean() - t.mean()), low, high, int(m.size + t.size), samples, seed, source)
    for (model_key, dataset), group in primary.groupby(["model_key", "dataset"], dropna=False):
        estimate, low, high = bootstrap_mean(group["alpha_star"].to_numpy(float), samples, rng)
        add_ci_row(rows, "cctp_alpha", {"dataset": dataset, "model_key": model_key}, "mean_alpha_star_by_model_dataset", estimate, low, high, len(group), samples, seed, source)
    return pd.DataFrame(rows)


def write_report(output_root: Path, frames: Dict[str, pd.DataFrame], samples: int, seed: int) -> None:
    lines = [
        "# Bootstrap CI Report",
        "",
        f"- bootstrap samples: {samples}",
        f"- seed: {seed}",
        "- method: paired bootstrap over item IDs within model-dataset groups where item-level data are available",
        "- truncation boundary CIs bootstrap over model-dataset aggregate rows because the expanded item-level artifact does not expose all panel metrics in a wide paired table",
        "",
        "## Stability Notes",
        "",
        "Pressure-ramp, failure-decomposition, recovery-hierarchy continuous metrics, and CCTP alpha summaries have paired bootstrap intervals. Recovered pair counts are deterministic descriptive counts from aggregate criteria unless explicitly modeled otherwise.",
        "",
    ]
    for name, frame in frames.items():
        lines.extend([f"## {name}", "", markdown_table(frame.head(12).to_dict("records"), list(frame.columns[: min(10, len(frame.columns))])), ""])
    (output_root / "bootstrap_ci_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = parser_with_output("Generate paper readiness bootstrap confidence intervals")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    output_root = output_root_from_args(args)
    frames = {
        "pressure_ramp_cis": pressure_cis(args.bootstrap_samples, args.seed),
        "history_contamination_cis": failure_cis(args.bootstrap_samples, args.seed),
        "recovery_hierarchy_cis": recovery_cis(args.bootstrap_samples, args.seed),
        "truncation_boundary_cis": truncation_cis(args.bootstrap_samples, args.seed),
        "cctp_alpha_cis": cctp_cis(args.bootstrap_samples, args.seed),
    }
    write_json(output_root / "bootstrap_config.json", {"bootstrap_samples": args.bootstrap_samples, "seed": args.seed, "method": "paired item bootstrap within model-dataset groups where item-level data are available"})
    for name, frame in frames.items():
        write_dataframe(output_root / f"{name}.csv", frame)
    write_report(output_root, frames, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
