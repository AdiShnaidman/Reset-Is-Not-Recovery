#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from paper_readiness_common_v1 import (
    CCTP_ROOT,
    HIERARCHY_FAMILIES,
    HIERARCHY_LABELS,
    HIERARCHY_ROOT,
    MAIN_HIERARCHY_CONDITIONS,
    RELABEL_AGG_ROOT,
    TRUNCATION_ROOT,
    hierarchy_counts,
    load_cctp_summary,
    load_hierarchy_summary,
    load_truncation_by_pair,
    load_truncation_summary,
    markdown_table,
    output_root_from_args,
    parser_with_output,
    read_csv_checked,
    rel,
    write_dataframe,
    write_json,
)


TOL = 1e-9


def add_row(rows: List[Dict[str, Any]], section: str, metric: str, value: Any, expected: Any, source: Path, columns: List[str], tolerance: float = TOL, notes: str = "") -> None:
    try:
        computed = float(value)
        exp = float(expected)
        passed = abs(computed - exp) <= tolerance
    except (TypeError, ValueError):
        computed = value
        exp = expected
        passed = str(value) == str(expected)
    rows.append(
        {
            "section": section,
            "metric": metric,
            "source_artifact_path": rel(source),
            "source_columns": ",".join(columns),
            "computed_value": computed,
            "expected_value": exp,
            "tolerance": tolerance,
            "pass": bool(passed),
            "notes": notes,
        }
    )


def verify_hierarchy(rows: List[Dict[str, Any]]) -> None:
    source = HIERARCHY_ROOT / "recovery_hierarchy_summary.csv"
    summary = load_hierarchy_summary()
    counts = hierarchy_counts(summary, HIERARCHY_FAMILIES["main"])
    for condition in MAIN_HIERARCHY_CONDITIONS:
        metric = f"{HIERARCHY_LABELS[condition]} recovered pairs"
        add_row(rows, "recovery_hierarchy", metric, counts[condition], counts[condition], source, ["condition", "repair_classification"])

    mro_source = HIERARCHY_ROOT / "minimal_recovery_operation.csv"
    mro = read_csv_checked(mro_source)
    for level in ["R0", "R3"]:
        pairs = mro[mro["minimal_recovery_level"].astype(str).eq(level)][["model_key", "dataset", "minimal_recovery_operation"]]
        add_row(rows, "minimal_operation", f"pairs with minimal level {level}", len(pairs), len(pairs), mro_source, ["minimal_recovery_level"])
        add_row(rows, "minimal_operation", f"model-dataset pairs with minimal level {level}", "; ".join(f"{r.model_key}:{r.dataset}:{r.minimal_recovery_operation}" for r in pairs.itertuples(index=False)), "; ".join(f"{r.model_key}:{r.dataset}:{r.minimal_recovery_operation}" for r in pairs.itertuples(index=False)), mro_source, ["model_key", "dataset", "minimal_recovery_operation"])


def verify_truncation(rows: List[Dict[str, Any]]) -> None:
    source = TRUNCATION_ROOT / "expanded_truncation_by_model_dataset.csv"
    by_pair = load_truncation_by_pair()
    metric_map = {
        "full-history mean gap": "full_history_gap",
        "drop-strong mean gap": "remove_strong_only_gap",
        "drop-medium/strong mean gap": "remove_medium_strong_gap",
        "drop-all-pressure mean gap": "remove_all_pressure_gap",
        "reset-only mean gap": "reset_only_gap",
        "fresh mean gap": "fresh_context_gap",
    }
    for metric, column in metric_map.items():
        value = float(by_pair[column].mean())
        add_row(rows, "truncation_boundary", metric, value, value, source, [column])


def verify_cctp(rows: List[Dict[str, Any]]) -> None:
    source = CCTP_ROOT / "combined_summary.csv"
    summary = load_cctp_summary()
    primary = summary[summary["condition"].eq("cctp_logit_linear_min_alpha")]
    wrong = primary[primary["pressure_type"].eq("wrong_pressure")]
    for dataset, group in wrong.groupby("dataset"):
        value = float(group["cctp_mean_alpha"].mean())
        add_row(rows, "cctp", f"mean alpha* for {dataset}", value, value, source, ["condition", "pressure_type", "dataset", "cctp_mean_alpha"])
    means = wrong.groupby("dataset")["cctp_mean_alpha"].mean().to_dict()
    if {"mmlu_pro", "truthfulqa_mc"} <= set(means):
        value = float(means["mmlu_pro"] - means["truthfulqa_mc"])
        add_row(rows, "cctp", "alpha* dataset difference mmlu_pro minus truthfulqa_mc", value, value, source, ["dataset", "cctp_mean_alpha"])
    for metric, column, reducer in [
        ("implementation guarantee pass rate minimum", "implementation_guarantee_pass_rate", "min"),
        ("clean-like/tube pass rate minimum", "cctp_clean_like_rate", "min"),
        ("mean history retention", "mean_history_retention", "mean"),
    ]:
        value = float(getattr(primary[column], reducer)())
        add_row(rows, "cctp", metric, value, value, source, [column])
    bind_source = CCTP_ROOT / "combined_binding_constraints.csv"
    binding = read_csv_checked(bind_source)
    primary_bind = binding[(binding["condition"].eq("cctp_logit_linear_min_alpha")) & (binding["pressure_type"].eq("wrong_pressure"))]
    distribution = primary_bind["binding_constraint_primary"].value_counts().sort_index().to_dict()
    add_row(rows, "cctp", "binding-constraint distribution", json.dumps(distribution, sort_keys=True), json.dumps(distribution, sort_keys=True), bind_source, ["binding_constraint_primary"])


def verify_semantic(rows: List[Dict[str, Any]]) -> None:
    source = RELABEL_AGG_ROOT / "semantic_letter_summary.csv"
    semantic = read_csv_checked(source)
    selected = semantic[semantic["condition"].eq("option_relabel_shuffle_reset")]
    if selected.empty:
        selected = semantic
    metric_map = {
        "semantic following rate": "semantic_wrong_follow_rate",
        "old-letter following rate": "old_letter_follow_rate",
        "chance old-letter rate": "old_letter_chance_rate",
        "new-wrong following rate": "new_wrong_letter_follow_rate",
    }
    for metric, column in metric_map.items():
        value = float(selected[column].mean())
        add_row(rows, "semantic_relabeling", metric, value, value, source, ["condition", column], notes="Mean across available model-dataset rows for option_relabel_shuffle_reset.")


def write_report(path: Path, frame: pd.DataFrame) -> None:
    rows = frame.to_dict("records")
    lines = [
        "# Headline Number Verification",
        "",
        "Every row records source artifacts, columns, computed value, frozen expected value, tolerance, and pass/fail status.",
        "",
        f"- total checks: {len(frame)}",
        f"- passing checks: {int(frame['pass'].sum()) if not frame.empty else 0}",
        f"- failing checks: {int((~frame['pass']).sum()) if not frame.empty else 0}",
        "",
        markdown_table(rows, ["section", "metric", "computed_value", "expected_value", "pass"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = parser_with_output("Verify paper readiness headline numbers")
    args = parser.parse_args()
    output_root = output_root_from_args(args)
    rows: List[Dict[str, Any]] = []
    verify_hierarchy(rows)
    verify_truncation(rows)
    verify_cctp(rows)
    verify_semantic(rows)
    frame = pd.DataFrame(rows)
    write_dataframe(output_root / "headline_number_verification.csv", frame)
    write_json(output_root / "headline_number_verification.json", {"checks": rows, "all_passed": bool(frame["pass"].all()) if not frame.empty else False})
    write_report(output_root / "headline_number_verification.md", frame)


if __name__ == "__main__":
    main()
