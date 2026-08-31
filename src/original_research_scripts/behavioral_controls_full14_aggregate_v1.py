#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MODEL_KEYS = ["qwen25_15b", "qwen25_7b", "qwen25_14b", "mistral7b_v03"]
EXPANDED_MODEL_KEYS = ["llama31_8b", "gemma2_2b", "gemma2_9b"]
ALL_MODEL_KEYS = [*ORIGINAL_MODEL_KEYS, *EXPANDED_MODEL_KEYS]
DATASETS = ["truthfulqa_mc", "mmlu_pro"]
DATASET_LABELS = {"truthfulqa_mc": "TruthfulQA-MC", "mmlu_pro": "MMLU-Pro"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate full14 behavioral-control artifacts")
    parser.add_argument("--original-root", default="runs/hysteresis_model_suite_v1")
    parser.add_argument("--expanded-root", default="runs/behavioral_controls_expanded_family_v1")
    parser.add_argument("--output-root", default="runs/behavioral_controls_full14_v1")
    return parser.parse_args()


def recovery_raw_path(model_key: str, dataset: str) -> Path:
    if model_key in EXPANDED_MODEL_KEYS:
        return REPO_ROOT / "runs" / "recovery_hierarchy_expanded_family_v1" / "raw" / model_key / dataset / "recovery_hierarchy_raw.jsonl"
    return REPO_ROOT / "runs" / "recovery_hierarchy" / "recovery_hierarchy_n500" / model_key / dataset / "recovery_hierarchy_raw.jsonl"


def control_summary_path(root: Path, experiment: str, model_key: str, dataset: str) -> Path:
    return root / experiment / model_key / dataset / "pressure_ramp_summary.csv"


def control_item_summary_path(root: Path, experiment: str, model_key: str, dataset: str) -> Path:
    return root / experiment / model_key / dataset / "pressure_ramp_item_summary.csv"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path.relative_to(REPO_ROOT)}")
    return path


def read_recovery_pressure(model_key: str, dataset: str) -> Dict[str, Any]:
    path = require_file(recovery_raw_path(model_key, dataset))
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") == "ordinary_strong_reset":
                rows.append(row)
    if len(rows) != 500:
        raise SystemExit(f"Expected 500 ordinary_strong_reset rows in {path.relative_to(REPO_ROOT)}, found {len(rows)}")
    frame = pd.DataFrame(rows)
    first = rows[0]
    clean = frame["clean_p_wrong"].astype(float)
    reset = frame["ordinary_reset_p_wrong"].astype(float)
    return {
        "model_key": model_key,
        "model_id": str(first.get("model_id", "")),
        "model_short_name": str(first.get("model_short_name", model_key)),
        "family": str(first.get("family", "")),
        "size_label": str(first.get("size_label", "")),
        "dataset": dataset,
        "n_items": int(len(frame)),
        "pressure_clean_p_wrong": float(clean.mean()),
        "pressure_reset_p_wrong": float(reset.mean()),
        "pressure_gap": float((reset - clean).mean()),
        "pressure_wrong_follow_rate": float(frame["ordinary_reset_wrong_follow"].astype(bool).mean()),
        "pressure_source": str(path.relative_to(REPO_ROOT)),
    }


def read_control_summary(path: Path) -> pd.Series:
    require_file(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"Empty control summary: {path.relative_to(REPO_ROOT)}")
    return frame.iloc[0]


def read_control_status(root: Path, experiment: str, model_key: str, dataset: str) -> Dict[str, Any]:
    path = root / experiment / model_key / dataset / "model_status.json"
    require_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "success":
        raise SystemExit(f"Control run is not successful: {path.relative_to(REPO_ROOT)} status={data.get('status')}")
    return data


def read_control_metrics(root: Path, model_key: str, dataset: str) -> Dict[str, Any]:
    neutral_status = read_control_status(root, "neutral_control", model_key, dataset)
    correct_status = read_control_status(root, "correct_pressure_control", model_key, dataset)
    neutral_path = control_summary_path(root, "neutral_control", model_key, dataset)
    correct_path = control_summary_path(root, "correct_pressure_control", model_key, dataset)
    correct_item_path = control_item_summary_path(root, "correct_pressure_control", model_key, dataset)
    neutral = read_control_summary(neutral_path)
    correct = read_control_summary(correct_path)
    correct_items = pd.read_csv(require_file(correct_item_path))
    if correct_items.empty:
        raise SystemExit(f"Empty correct-pressure item summary: {correct_item_path.relative_to(REPO_ROOT)}")
    return {
        "neutral_gap": float(neutral["mean_hysteresis_gap_vs_clean_probe"]),
        "neutral_reset_p_wrong": float(neutral["mean_strong_reset_p_wrong"]),
        "neutral_wrong_follow_rate": float(neutral["wrong_follow_strong_reset_rate"]),
        "correct_reset_p_correct": float(correct_items["strong_reset_p_correct"].astype(float).mean()),
        "correct_reset_p_wrong": float(correct["mean_strong_reset_p_wrong"]),
        "correct_wrong_follow_rate": float(correct["wrong_follow_strong_reset_rate"]),
        "neutral_n_scored": int(neutral_status.get("n_scored", neutral.get("n_items", 0))),
        "correct_n_scored": int(correct_status.get("n_scored", correct.get("n_items", 0))),
        "neutral_source": str(neutral_path.relative_to(REPO_ROOT)),
        "correct_source": str(correct_path.relative_to(REPO_ROOT)),
        "correct_item_source": str(correct_item_path.relative_to(REPO_ROOT)),
    }


def build_rows(original_root: Path, expanded_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_key in ALL_MODEL_KEYS:
        control_root = expanded_root if model_key in EXPANDED_MODEL_KEYS else original_root
        for dataset in DATASETS:
            row = read_recovery_pressure(model_key, dataset)
            row.update(read_control_metrics(control_root, model_key, dataset))
            row["suite_family"] = "expanded" if model_key in EXPANDED_MODEL_KEYS else "original"
            rows.append(row)
    return pd.DataFrame(rows)


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    weights = group["n_items"].astype(float).to_numpy()
    values = group[column].astype(float).to_numpy()
    return float(np.average(values, weights=weights))


def aggregate_rows(frame: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    metric_columns = [
        "pressure_gap",
        "neutral_gap",
        "correct_reset_p_correct",
        "correct_reset_p_wrong",
        "pressure_wrong_follow_rate",
        "neutral_wrong_follow_rate",
        "correct_wrong_follow_rate",
    ]
    for key, group in frame.groupby(list(group_columns), sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_columns, values)}
        row["n_items"] = int(group["n_items"].sum())
        row["n_pairs"] = int(len(group))
        for column in metric_columns:
            row[column] = weighted_mean(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def tex_escape(value: Any) -> str:
    text = str(value)
    for old, new in [("&", r"\&"), ("_", r"\_"), ("%", r"\%")]:
        text = text.replace(old, new)
    return text


def write_latex(frame: pd.DataFrame, path: Path) -> None:
    row_end = " " + chr(92) * 2
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        "Model & Dataset & Pressure gap & Neutral gap & Correct reset $p_c$ & Correct reset $p_w$" + row_end,
        r"\hline",
    ]
    display = frame.sort_values(["suite_family", "model_key", "dataset"])
    for row in display.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    tex_escape(row.model_short_name),
                    tex_escape(DATASET_LABELS.get(row.dataset, row.dataset)),
                    f"{row.pressure_gap:.3f}",
                    f"{row.neutral_gap:.3f}",
                    f"{row.correct_reset_p_correct:.3f}",
                    f"{row.correct_reset_p_wrong:.3f}",
                ]
            )
            + row_end
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Behavioral controls across fourteen model--dataset pairs. Pressure gap is derived from saved ordinary-reset recovery-hierarchy rows; neutral and correct-pressure controls use the pressure-ramp control runner.}",
            r"\label{tab:behavioral-controls-full14}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(frame: pd.DataFrame, dataset_summary: pd.DataFrame, aggregate_summary: pd.DataFrame, path: Path) -> None:
    expanded = frame[frame["suite_family"] == "expanded"]
    lines = [
        "# Behavioral Controls Full14 Report",
        "",
        "This derived artifact combines saved original-suite controls with newly scored expanded-family neutral and correct-pressure controls. Pressure-gap values use saved recovery-hierarchy ordinary-reset rows for a consistent provenance across all fourteen pairs.",
        "",
        "## Completion",
        "",
        f"- Total model-dataset pairs: {len(frame)}",
        f"- Expanded-family pairs newly checked: {len(expanded)}",
        f"- Expanded-family neutral rows scored: {int(expanded['neutral_n_scored'].sum())}",
        f"- Expanded-family correct-pressure rows scored: {int(expanded['correct_n_scored'].sum())}",
        "",
        "## Aggregate Means",
        "",
        "| n pairs | pressure gap | neutral gap | correct reset p_c | correct reset p_w |",
        "|---:|---:|---:|---:|---:|",
    ]
    agg = aggregate_summary.iloc[0]
    lines.append(
        f"| {int(agg.n_pairs)} | {agg.pressure_gap:.3f} | {agg.neutral_gap:.3f} | "
        f"{agg.correct_reset_p_correct:.3f} | {agg.correct_reset_p_wrong:.3f} |"
    )
    lines.extend(["", "## Dataset Means", "", "| dataset | n pairs | pressure gap | neutral gap | correct reset p_c | correct reset p_w |", "|---|---:|---:|---:|---:|---:|"])
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS.get(row.dataset, row.dataset)} | {int(row.n_pairs)} | {row.pressure_gap:.3f} | "
            f"{row.neutral_gap:.3f} | {row.correct_reset_p_correct:.3f} | {row.correct_reset_p_wrong:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `full14_behavioral_controls_model_dataset_summary.csv`",
            "- `expanded_family_behavioral_controls_model_dataset_summary.csv`",
            "- `full14_behavioral_controls_dataset_summary.csv`",
            "- `full14_behavioral_controls_aggregate_summary.csv`",
            "- `full14_behavioral_controls_table.tex`",
            "- `full14_behavioral_controls_manifest.md`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    lines = ["# Behavioral Controls Full14 Manifest", "", "## Source Artifacts", ""]
    for column in ["pressure_source", "neutral_source", "correct_source", "correct_item_source"]:
        lines.append(f"### {column}")
        for source in sorted(frame[column].unique()):
            lines.append(f"- {source}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    original_root = REPO_ROOT / args.original_root
    expanded_root = REPO_ROOT / args.expanded_root
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    frame = build_rows(original_root, expanded_root)
    expanded = frame[frame["suite_family"] == "expanded"].copy()
    dataset_summary = aggregate_rows(frame, ["dataset"])
    aggregate_summary = aggregate_rows(frame.assign(all_pairs="all"), ["all_pairs"])

    frame.to_csv(output_root / "full14_behavioral_controls_model_dataset_summary.csv", index=False)
    expanded.to_csv(output_root / "expanded_family_behavioral_controls_model_dataset_summary.csv", index=False)
    dataset_summary.to_csv(output_root / "full14_behavioral_controls_dataset_summary.csv", index=False)
    aggregate_summary.to_csv(output_root / "full14_behavioral_controls_aggregate_summary.csv", index=False)
    write_latex(frame, output_root / "full14_behavioral_controls_table.tex")
    write_report(frame, dataset_summary, aggregate_summary, output_root / "full14_behavioral_controls_report.md")
    write_manifest(frame, output_root / "full14_behavioral_controls_manifest.md")
    print(f"Wrote behavioral-control full14 artifacts to {output_root.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()