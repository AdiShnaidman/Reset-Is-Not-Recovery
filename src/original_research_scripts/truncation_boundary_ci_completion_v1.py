#!/usr/bin/env python3
"""Complete truncation-boundary confidence intervals from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITION_LABELS = {
    "full_history_reset": "Full pressure history",
    "remove_strong_only": "Drop strongest turn",
    "remove_medium_strong": "Drop medium+strong turns",
    "remove_all_pressure": "Drop all pressure turns",
    "reset_only": "Reset only",
}

CONDITION_ORDER = list(CONDITION_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/expanded_truncation_boundary_v1"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return percentile_ci(means)


def item_level_gaps(item_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_key", "dataset", "item_id"]
    clean = item_rows[item_rows["condition"] == "fresh_context"][
        keys + ["p_advocated_wrong"]
    ].rename(columns={"p_advocated_wrong": "p_clean_wrong"})
    merged = item_rows[item_rows["condition"].isin(CONDITION_ORDER)][
        keys + ["condition", "p_advocated_wrong"]
    ].merge(clean, on=keys, how="inner", validate="many_to_one")
    merged["gap"] = merged["p_advocated_wrong"] - merged["p_clean_wrong"]
    return merged


def condition_summary(
    gaps: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        values = gaps.loc[gaps["condition"] == condition, "gap"].to_numpy(float)
        ci_low, ci_high = bootstrap_mean_ci(values, samples, seed)
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "mean_gap": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_items": int(len(values)),
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
                "bootstrap_unit": "item",
            }
        )
    return pd.DataFrame(rows)


def aggregate_row_summary(
    summary_rows: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        values = summary_rows.loc[
            summary_rows["condition"] == condition, "hysteresis_gap_mean"
        ].to_numpy(float)
        ci_low, ci_high = bootstrap_mean_ci(values, samples, seed)
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "mean_gap": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_items": int(len(values)),
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
                "bootstrap_unit": "model_dataset_aggregate_row",
            }
        )
    return pd.DataFrame(rows)


def format_interval(row: pd.Series) -> str:
    return f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def write_main_table(summary: pd.DataFrame, path: Path, caption_note: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Context retained & Mean gap & 95\% CI \\",
        r"\hline",
    ]
    for condition in CONDITION_ORDER:
        row = summary.loc[summary["condition"] == condition].iloc[0]
        lines.append(
            f"{row['label']} & {row['mean_gap']:.3f} & {format_interval(row)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Truncation-boundary analysis on high-hysteresis pairs. "
                r"Partial removal helps, but clean-like behavior appears only when all "
                rf"pressure-bearing turns are removed. {caption_note}" + "}"
            ),
            r"\label{tab:truncation-boundary}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(
    item_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    path: Path,
    source_paths: Iterable[Path],
) -> None:
    lines = [
        "# Truncation Boundary CI Completion",
        "",
        "## Inputs",
    ]
    for source_path in source_paths:
        lines.append(f"- `{source_path}`")
    lines.extend(
        [
            "",
            "## Item-Level Paired Bootstrap",
            "",
            "| Condition | Mean gap | 95% CI | n |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in item_summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['mean_gap']:.3f} | {format_interval(row)} | {int(row['n_items'])} |"
        )
    lines.extend(
        [
            "",
            "## Model-Dataset Aggregate-Row Bootstrap",
            "",
            "| Condition | Mean gap | 95% CI | n rows |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in aggregate_summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['mean_gap']:.3f} | {format_interval(row)} | {int(row['n_items'])} |"
        )
    lines.extend(
        [
            "",
            "The previously reported endpoint intervals match the aggregate-row bootstrap, not the item-level paired bootstrap.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    item_path = input_root / "expanded_truncation_item_level.jsonl"
    summary_path = input_root / "expanded_truncation_summary.csv"
    if not item_path.exists() or not summary_path.exists():
        missing = [str(path) for path in [item_path, summary_path] if not path.exists()]
        raise FileNotFoundError("Missing required saved artifact(s): " + ", ".join(missing))

    item_rows = read_jsonl(item_path)
    summary_rows = pd.read_csv(summary_path)
    gaps = item_level_gaps(item_rows)
    item_summary = condition_summary(gaps, args.bootstrap_samples, args.bootstrap_seed)
    aggregate_summary = aggregate_row_summary(
        summary_rows, args.bootstrap_samples, args.bootstrap_seed
    )

    csv_columns = [
        "condition",
        "mean_gap",
        "ci_low",
        "ci_high",
        "n_items",
        "bootstrap_samples",
        "bootstrap_seed",
    ]
    item_summary[csv_columns].to_csv(
        input_root / "truncation_boundary_ci_summary.csv", index=False
    )
    aggregate_summary[csv_columns].to_csv(
        input_root / "truncation_boundary_aggregate_row_ci_summary.csv", index=False
    )
    item_summary.to_csv(input_root / "truncation_boundary_ci_summary_detailed.csv", index=False)
    aggregate_summary.to_csv(
        input_root / "truncation_boundary_aggregate_row_ci_summary_detailed.csv",
        index=False,
    )
    gaps.to_csv(input_root / "truncation_boundary_item_level_gaps.csv", index=False)
    write_main_table(
        item_summary,
        input_root / "truncation_boundary_main_table_completed_item_bootstrap.tex",
        "Intervals use paired item-level bootstrap over saved item rows.",
    )
    write_main_table(
        aggregate_summary,
        input_root / "truncation_boundary_main_table_completed_aggregate_bootstrap.tex",
        "Intervals use bootstrap over model--dataset aggregate rows, matching the existing endpoint intervals.",
    )
    write_main_table(
        aggregate_summary,
        input_root / "truncation_boundary_main_table_completed.tex",
        "Intervals use bootstrap over model--dataset aggregate rows, matching the existing endpoint intervals.",
    )
    write_markdown_report(
        item_summary,
        aggregate_summary,
        input_root / "truncation_boundary_ci_completion_report.md",
        [item_path, summary_path],
    )


if __name__ == "__main__":
    main()