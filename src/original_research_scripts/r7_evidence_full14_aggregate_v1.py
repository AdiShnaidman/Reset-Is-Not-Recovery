#!/usr/bin/env python3
"""Aggregate full-14 R7 trusted-evidence recovery outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import groundlm_evidence_recovery_v1 as r7


MODEL_SUITES = {
    **{model_key: "original" for model_key in r7.ORIGINAL_MODEL_KEYS},
    **{model_key: "expanded" for model_key in r7.EXPANDED_MODEL_KEYS},
}
TABLE_CONDITIONS = ["ordinary_strong_reset", r7.EVIDENCE_CONDITION]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("runs/groundlm_evidence_recovery_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/r7_evidence_full14_v1"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(r7.REPO_ROOT))
    except ValueError:
        return str(path)


def expected_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for model_key, suite in MODEL_SUITES.items():
        for dataset in r7.DATASETS:
            pairs.append((suite, model_key, dataset))
    return pairs


def r7_raw_path(source_root: Path, suite: str, model_key: str, dataset: str) -> Path:
    return source_root / suite / model_key / dataset / "evidence_grounded_recovery_raw.jsonl"


def validate_outputs(source_root: Path, output_root: Path, allow_partial: bool) -> pd.DataFrame:
    rows = []
    missing = []
    for suite, model_key, dataset in expected_pairs():
        path = r7_raw_path(source_root, suite, model_key, dataset)
        status_path = path.parent / "run_status.json"
        n_rows = 0
        status = "missing"
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                n_rows = sum(1 for line in handle if line.strip())
            status = "complete" if n_rows == 500 else "row_count_mismatch"
        status_payload: dict[str, Any] = {}
        if status_path.is_file():
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": suite,
                "model_key": model_key,
                "dataset": dataset,
                "raw_path": rel(path),
                "status_path": rel(status_path),
                "status": status,
                "raw_rows": n_rows,
                "run_status": status_payload.get("status", ""),
            }
        )
        if status != "complete":
            missing.append(f"{suite}/{model_key}/{dataset}: {status} ({n_rows} rows)")
    status_frame = pd.DataFrame(rows)
    status_frame.to_csv(output_root / "full14_pair_completion_status.csv", index=False)
    if missing and not allow_partial:
        raise SystemExit("Incomplete R7 outputs:\n" + "\n".join(missing))
    return status_frame


def collect_full14_rows(source_root: Path) -> pd.DataFrame:
    frames = []
    for suite in ["original", "expanded"]:
        frames.append(r7.collect_comparison_rows(source_root, suite))
    frame = pd.concat(frames, ignore_index=True)
    frame["suite"] = frame["model_key"].map(MODEL_SUITES)
    return frame


def subset_table_conditions(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary["condition"].isin(TABLE_CONDITIONS)].copy()
    order = {condition: index for index, condition in enumerate(TABLE_CONDITIONS)}
    table["condition_order"] = table["condition"].map(order)
    return table.sort_values(["condition_order"])


def delta_rows(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    metrics = [
        "accuracy",
        "mean_p_wrong",
        "mean_p_correct",
        "mean_hysteresis_gap",
        "wrong_following_rate",
        "history_contamination_rate_clean_correct",
    ]
    grouped = [((), summary)] if not group_cols else summary.groupby(group_cols, sort=True, dropna=False)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, key_tuple))
        by_condition = {row.condition: row for row in group.itertuples(index=False)}
        if "ordinary_strong_reset" not in by_condition or r7.EVIDENCE_CONDITION not in by_condition:
            continue
        r0_row = by_condition["ordinary_strong_reset"]
        r7_row = by_condition[r7.EVIDENCE_CONDITION]
        for metric in metrics:
            record[f"r0_{metric}"] = float(getattr(r0_row, metric))
            record[f"r7_{metric}"] = float(getattr(r7_row, metric))
            record[f"delta_{metric}"] = float(getattr(r7_row, metric) - getattr(r0_row, metric))
        record["n_items"] = int(getattr(r7_row, "n_items"))
        rows.append(record)
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame) -> str:
    table = subset_table_conditions(summary)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Condition & Acc. & Mean $p(w_i)$ & Gap & Wrong-follow & Contam. \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{row.condition_label} & {row.accuracy:.3f} & {row.mean_p_wrong:.3f} & "
            f"{row.mean_hysteresis_gap:.3f} & {100.0 * row.wrong_following_rate:.1f}"
            r"\% & "
            f"{100.0 * row.history_contamination_rate_clean_correct:.1f}"
            r"\% \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Full fourteen-pair R7 trusted-evidence recovery. R0 is the ordinary reset with pressure history retained; R7 retains the history but adds the trusted evidence block.}",
            r"\label{tab:r7-evidence-full14}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    output_root: Path,
    status: pd.DataFrame,
    overall_delta: pd.DataFrame,
    dataset_delta: pd.DataFrame,
) -> None:
    complete = status[status["status"].eq("complete")]
    failed = status[~status["status"].eq("complete")]
    lines = [
        "# R7 Evidence Full-14 Report",
        "",
        "## Inputs",
        "",
        "- Existing recovery hierarchy raw rows from `runs/recovery_hierarchy/recovery_hierarchy_n500` and `runs/recovery_hierarchy_expanded_family_v1/raw`.",
        "- Existing R7 trusted-evidence raw rows from `runs/groundlm_evidence_recovery_v1`.",
        "- Accepted n=500 item subsets from `runs/hysteresis_model_suite_v1/subsets`.",
        "- R7 evidence generated automatically by the existing rule: dataset reference/explanation/evidence field when available, otherwise deterministic constructed gold evidence from the saved question and benchmark-correct option text.",
        "- No manual evidence was written.",
        "",
        "## Completion",
        "",
        f"- Complete pairs: {len(complete)}/14",
        f"- Failed or incomplete pairs: {len(failed)}/14",
        "",
    ]
    if not failed.empty:
        lines.extend(["Incomplete pairs:", ""])
        for row in failed.itertuples(index=False):
            lines.append(f"- {row.suite}/{row.model_key}/{row.dataset}: {row.status}, rows={row.raw_rows}")
        lines.append("")
    if not overall_delta.empty:
        row = overall_delta.iloc[0]
        lines.extend(
            [
                "## Full-14 R0 vs R7",
                "",
                f"- R0 accuracy: {row['r0_accuracy']:.3f}",
                f"- R7 accuracy: {row['r7_accuracy']:.3f}",
                f"- R0 mean p(w_i): {row['r0_mean_p_wrong']:.3f}",
                f"- R7 mean p(w_i): {row['r7_mean_p_wrong']:.3f}",
                f"- Mean p(w_i) change: {row['delta_mean_p_wrong']:.3f}",
                f"- R0 hysteresis gap: {row['r0_mean_hysteresis_gap']:.3f}",
                f"- R7 hysteresis gap: {row['r7_mean_hysteresis_gap']:.3f}",
                f"- Gap change: {row['delta_mean_hysteresis_gap']:.3f}",
                f"- R0 wrong-following: {100.0 * row['r0_wrong_following_rate']:.1f}%",
                f"- R7 wrong-following: {100.0 * row['r7_wrong_following_rate']:.1f}%",
                f"- R0 contamination: {100.0 * row['r0_history_contamination_rate_clean_correct']:.1f}%",
                f"- R7 contamination: {100.0 * row['r7_history_contamination_rate_clean_correct']:.1f}%",
                "",
            ]
        )
    if not dataset_delta.empty:
        lines.extend(["## Dataset Deltas", "", "| Dataset | R0 acc. | R7 acc. | R0 p(w_i) | R7 p(w_i) | R0 wrong-follow | R7 wrong-follow |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in dataset_delta.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.r0_accuracy:.3f} | {row.r7_accuracy:.3f} | "
                f"{row.r0_mean_p_wrong:.3f} | {row.r7_mean_p_wrong:.3f} | "
                f"{100.0 * row.r0_wrong_following_rate:.1f}% | {100.0 * row.r7_wrong_following_rate:.1f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## Runtime Notes",
            "",
            "Each model-dataset pair was scored with the existing `groundlm_evidence_recovery_v1.py` runner. Gemma chat-template compatibility uses the existing system-prompt folding path in that runner.",
            "",
        ]
    )
    (output_root / "full14_r7_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root
    status = validate_outputs(source_root, output_root, args.allow_partial)
    frame = collect_full14_rows(source_root)
    frame.to_csv(output_root / "full14_comparison_item_rows.csv", index=False)
    pair = r7.summarize(
        frame,
        ["suite", "model_key", "model_short_name", "dataset", "condition", "condition_label"],
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    by_model = r7.summarize(
        frame,
        ["model_key", "model_short_name", "condition", "condition_label"],
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    by_dataset = r7.summarize(
        frame,
        ["dataset", "condition", "condition_label"],
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    overall = r7.summarize(
        frame,
        ["condition", "condition_label"],
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    pair.to_csv(output_root / "full14_pair_summary.csv", index=False)
    by_model.to_csv(output_root / "full14_model_summary.csv", index=False)
    by_dataset.to_csv(output_root / "full14_dataset_summary.csv", index=False)
    overall.to_csv(output_root / "full14_aggregate_summary.csv", index=False)
    overall_delta = delta_rows(overall, [])
    dataset_delta = delta_rows(by_dataset, ["dataset"])
    model_delta = delta_rows(by_model, ["model_key", "model_short_name"])
    pair_delta = delta_rows(pair, ["suite", "model_key", "model_short_name", "dataset"])
    overall_delta.to_csv(output_root / "full14_r0_vs_r7_delta.csv", index=False)
    dataset_delta.to_csv(output_root / "full14_dataset_delta.csv", index=False)
    model_delta.to_csv(output_root / "full14_model_delta.csv", index=False)
    pair_delta.to_csv(output_root / "full14_pair_delta.csv", index=False)
    (output_root / "full14_table6_r7_replacement.tex").write_text(latex_table(overall), encoding="utf-8")
    write_report(output_root, status, overall_delta, dataset_delta)


if __name__ == "__main__":
    main()