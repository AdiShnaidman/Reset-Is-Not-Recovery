#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from paper_readiness_common_v1 import (
    HIERARCHY_FAMILIES,
    HIERARCHY_LABELS,
    MAIN_HIERARCHY_CONDITIONS,
    hierarchy_counts,
    hierarchy_minimal_counts,
    load_hierarchy_summary,
    markdown_table,
    output_root_from_args,
    parser_with_output,
    write_dataframe,
    write_latex_table,
)


LAMBDA_VALUES = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]


def row_for_threshold(name: str, thresholds: Dict[str, float], summary: pd.DataFrame) -> Dict[str, Any]:
    counts = hierarchy_counts(summary, thresholds)
    mro = hierarchy_minimal_counts(summary, thresholds)
    row: Dict[str, Any] = {"threshold_family": name, **thresholds}
    for condition in MAIN_HIERARCHY_CONDITIONS:
        row[f"{condition}_recovered"] = counts[condition]
        row[f"mro_{condition}"] = mro[condition]
    row["mro_not_restored"] = mro["not_restored"]
    row["instruction_level_max"] = max(counts["explicit_user_retraction"], counts["system_level_reset"])
    row["state_changing_max"] = max(counts["fresh_context_deletion"], counts["context_truncation"], counts["factual_state_reconstruction"])
    row["neutral_or_reconstruction_max"] = max(counts["neutral_summary_replacement"], counts["factual_state_reconstruction"])
    row["qualitative_ordering_stable"] = bool(row["state_changing_max"] >= row["neutral_or_reconstruction_max"] >= row["instruction_level_max"])
    return row


def main() -> None:
    parser = parser_with_output("Run paper readiness hierarchy threshold sensitivity sweep")
    args = parser.parse_args()
    output_root = output_root_from_args(args)
    summary = load_hierarchy_summary()
    family_rows = [row_for_threshold(name, thresholds, summary) for name, thresholds in HIERARCHY_FAMILIES.items()]
    strict_main_loose = pd.DataFrame(family_rows)
    main = HIERARCHY_FAMILIES["main"]
    sweep_rows = []
    for value in LAMBDA_VALUES:
        thresholds = {key: threshold * value for key, threshold in main.items()}
        row = row_for_threshold(f"lambda_{value:.2f}", thresholds, summary)
        row["lambda"] = value
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)
    write_dataframe(output_root / "threshold_sensitivity_strict_main_loose.csv", strict_main_loose)
    write_dataframe(output_root / "threshold_sensitivity_multiplier_sweep.csv", sweep)
    tex_rows = []
    for row in family_rows:
        tex_rows.append({
            "family": row["threshold_family"],
            "R0": row["ordinary_strong_reset_recovered"],
            "R1": row["explicit_user_retraction_recovered"],
            "R2": row["system_level_reset_recovered"],
            "R3_delete": row["fresh_context_deletion_recovered"],
            "R3_truncate": row["context_truncation_recovered"],
            "R4": row["neutral_summary_replacement_recovered"],
            "R5": row["factual_state_reconstruction_recovered"],
            "R6": row["self_verification_final_label_recovered"],
        })
    write_latex_table(output_root / "threshold_sensitivity_table.tex", tex_rows, ["family", "R0", "R1", "R2", "R3_delete", "R3_truncate", "R4", "R5", "R6"], "Recovery count threshold sensitivity.", "tab:threshold-sensitivity")
    stable_rows = int(strict_main_loose["qualitative_ordering_stable"].sum())
    lambda_stable = int(sweep["qualitative_ordering_stable"].sum())
    lines = [
        "# Threshold Sensitivity Report",
        "",
        f"- strict/main/loose stable rows: {stable_rows}/{len(strict_main_loose)}",
        f"- multiplier sweep stable rows: {lambda_stable}/{len(sweep)}",
        "",
        "Qualitative ordering criterion: state-changing operations strongest, neutral summary/factual reconstruction intermediate, instruction-level repairs weak.",
        "",
        "## Strict/Main/Loose",
        "",
        markdown_table(tex_rows, ["family", "R0", "R1", "R2", "R3_delete", "R3_truncate", "R4", "R5", "R6"]),
        "",
        "## Multiplier Sweep",
        "",
        markdown_table(sweep[["lambda", "instruction_level_max", "neutral_or_reconstruction_max", "state_changing_max", "qualitative_ordering_stable"]].to_dict("records"), ["lambda", "instruction_level_max", "neutral_or_reconstruction_max", "state_changing_max", "qualitative_ordering_stable"]),
        "",
    ]
    (output_root / "threshold_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
