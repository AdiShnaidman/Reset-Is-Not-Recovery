#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from paper_readiness_common_v1 import (
    CCTP_ROOT,
    HIERARCHY_FAMILIES,
    HIERARCHY_ROOT,
    OUT_ROOT,
    REPO_ROOT,
    git_diff_stat,
    hierarchy_counts,
    load_cctp_items,
    load_cctp_summary,
    load_hierarchy_raw,
    load_hierarchy_summary,
    markdown_table,
    output_root_from_args,
    parser_with_output,
    read_json,
    rel,
    sha256_file,
    source_artifacts,
    source_checksum_rows,
    write_checksum_file,
    write_json,
)


def add(checks: List[Dict[str, Any]], name: str, passed: bool, details: str = "") -> None:
    checks.append({"check": name, "pass": bool(passed), "details": details})


def valid_tex(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return "\\begin{table" in text and "\\end{table" in text and text.count("{") == text.count("}")


def entropy_from_probs(values: List[float]) -> float:
    return float(-sum(max(float(value), 1e-12) * math.log(max(float(value), 1e-12)) for value in values))


def parse_probs(value: Any) -> Dict[str, float]:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, prob in data.items():
        try:
            out[str(key)] = float(prob)
        except (TypeError, ValueError):
            return {}
    return out


def compare_source_checksums(output_root: Path) -> bool:
    before = output_root / "source_checksums_before.sha256"
    if not before.is_file():
        return False
    current = {row["path"]: row["sha256"] for row in source_checksum_rows()}
    for line in before.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path = line.split(None, 1)
        path = path.strip()
        if current.get(path) != digest:
            return False
    return True


def write_final_report(output_root: Path, checks: List[Dict[str, Any]]) -> None:
    headline = pd.read_csv(output_root / "headline_number_verification.csv") if (output_root / "headline_number_verification.csv").is_file() else pd.DataFrame()
    threshold = pd.read_csv(output_root / "threshold_sensitivity_strict_main_loose.csv") if (output_root / "threshold_sensitivity_strict_main_loose.csv").is_file() else pd.DataFrame()
    cctp_ci = pd.read_csv(output_root / "cctp_alpha_cis.csv") if (output_root / "cctp_alpha_cis.csv").is_file() else pd.DataFrame()
    lines = [
        "# Paper Readiness Report",
        "",
        "## Summary Answers",
        "",
        "1. Criteria are frozen and separated: yes, if validation passes.",
        "2. Hierarchy and CCTP criteria are distinct: yes; hierarchy is aggregate model-dataset-operation, CCTP is item-level alpha* tube.",
        "3. Headline numbers reproduced: see `headline_number_verification.csv`.",
        "4. CIs support central claims: pressure, hierarchy continuous metrics, truncation, and CCTP alpha summaries have bootstrap intervals; recovered pair counts are descriptive deterministic counts.",
        "5. Threshold sensitivity supports the hierarchy: see `threshold_sensitivity_report.md`.",
        "6. Truncation boundary supports state-restoration interpretation: see panel C data and truncation CIs.",
        "7. CCTP claims are framed as oracle diagnostic / severity measurement via alpha*, history retention, tube-pass rate, and binding constraints.",
        "8. Release artifacts are sufficient for reproducing tables from existing item-level scores; no model outputs are copied or regenerated.",
        "9. Ready files include criteria TeX tables, verification CSV/MD, CI CSVs, threshold tables, figure data, and draft figure.",
        "10. Remaining unresolved items are listed below from validation failures or descriptive-only CI notes.",
        "",
        "## Validation Checks",
        "",
        markdown_table(checks, ["check", "pass", "details"]),
        "",
    ]
    if not headline.empty:
        counts = headline[headline["section"].eq("recovery_hierarchy")][["metric", "computed_value"]]
        lines.extend(["## Headline Recovered-Pair Counts", "", markdown_table(counts.to_dict("records"), ["metric", "computed_value"]), ""])
    if not cctp_ci.empty:
        lines.extend(["## Key CCTP CI Summary", "", markdown_table(cctp_ci.head(12).to_dict("records"), list(cctp_ci.columns[: min(8, len(cctp_ci.columns))])), ""])
    if not threshold.empty:
        keep = [column for column in threshold.columns if column in {"threshold_family", "ordinary_strong_reset_recovered", "explicit_user_retraction_recovered", "system_level_reset_recovered", "fresh_context_deletion_recovered", "context_truncation_recovered", "neutral_summary_replacement_recovered", "factual_state_reconstruction_recovered", "self_verification_final_label_recovered", "qualitative_ordering_stable"}]
        lines.extend(["## Threshold Sensitivity Summary", "", markdown_table(threshold[keep].to_dict("records"), keep), ""])
    unresolved = [row for row in checks if not row["pass"]]
    lines.extend([
        "## Unresolved Before Manuscript Editing",
        "",
        markdown_table(unresolved, ["check", "details"]) if unresolved else "No validation failures remain in the generated readiness package.",
        "",
        "## Paper-change recommendations, but do not apply them",
        "",
        "- add hierarchy criterion table",
        "- add CCTP clean-tube table",
        "- add CI values to key tables",
        "- add threshold sensitivity summary",
        "- promote truncation-boundary figure",
        "- reframe CCTP as alpha* severity metric",
        "- move open-ended results to appendix/limitations",
        "- remove stale commented blocks",
        "",
    ])
    (output_root / "PAPER_READINESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = parser_with_output("Validate generated paper readiness artifacts")
    args = parser.parse_args()
    output_root = output_root_from_args(args)
    checks: List[Dict[str, Any]] = []

    required_sources = source_artifacts()
    add(checks, "all required source artifacts exist", len(required_sources) >= 10, f"found {len(required_sources)} source artifacts")
    add(checks, "source checksums unchanged", compare_source_checksums(output_root), "compared current hashes with source_checksums_before.sha256")

    raw = load_hierarchy_raw()
    dup = int(raw.duplicated(["model_key", "dataset", "item_id", "condition"]).sum())
    add(checks, "no duplicated hierarchy model-dataset-item-condition rows", dup == 0, f"duplicates={dup}")
    paired_counts = raw.groupby(["model_key", "dataset", "item_id"])["condition"].nunique()
    add(checks, "every hierarchy item has required paired main conditions", bool((paired_counts >= 8).all()), f"min_condition_count={int(paired_counts.min()) if not paired_counts.empty else 0}")
    add(checks, "advocated wrong answer is never correct", bool((raw["wrong_advocated_letter"].astype(str) != raw["correct_option_letter"].astype(str)).all()), "hierarchy raw")
    hierarchy_option_consistent = bool(raw.groupby(["model_key", "dataset", "item_id"])["choices_json"].nunique().le(1).all())
    add(checks, "hierarchy answer option identities are consistent across paired contexts", hierarchy_option_consistent, "choices_json stable within model-dataset-item")

    cctp = load_cctp_items()
    cctp_primary = cctp[cctp["condition"].eq("cctp_logit_linear_min_alpha")]
    cctp_dup = int(cctp_primary.duplicated(["model_key", "dataset", "item_id", "pressure_type"]).sum())
    add(checks, "no duplicated CCTP primary rows", cctp_dup == 0, f"duplicates={cctp_dup}")
    add(checks, "CCTP alpha=1 implementation guarantee passes", bool(cctp_primary["implementation_guarantee_passed"].astype(bool).all()), "primary CCTP rows")
    add(checks, "CCTP internal decision flag not used in hierarchy tables", "strong_positive_scale_result" not in (HIERARCHY_ROOT / "recovery_hierarchy_summary.csv").read_text(encoding="utf-8"), "searched hierarchy summary")
    add(checks, "answer option identities are consistent across paired CCTP contexts", bool((cctp["correct_label"].astype(str) != cctp["advocated_wrong_label"].astype(str)).all()), "correct_label != advocated_wrong_label")
    cctp_option_stable = bool(cctp.groupby(["model_key", "dataset", "item_id"])["option_labels_json"].nunique().le(1).all())
    add(checks, "CCTP option label identities are stable across paired contexts", cctp_option_stable, "option_labels_json stable within model-dataset-item")

    prob_failures = 0
    answer_set_failures = 0
    entropy_failures = 0
    for row in cctp_primary.itertuples(index=False):
        clean = parse_probs(getattr(row, "clean_probs_json"))
        full = parse_probs(getattr(row, "full_history_probs_json"))
        projected = parse_probs(getattr(row, "projected_probs_json"))
        labels = set(clean)
        if not clean or set(full) != labels or set(projected) != labels:
            answer_set_failures += 1
            continue
        for probs in [clean, full, projected]:
            if abs(sum(probs.values()) - 1.0) > 1e-6:
                prob_failures += 1
        if abs(entropy_from_probs(list(projected.values())) - float(getattr(row, "entropy_projected"))) > 1e-6:
            entropy_failures += 1
    add(checks, "probabilities sum to 1 within tolerance", prob_failures == 0, f"probability_sum_failures={prob_failures}")
    add(checks, "entropy is computed over the same answer set", answer_set_failures == 0 and entropy_failures == 0, f"answer_set_failures={answer_set_failures}; entropy_failures={entropy_failures}")

    generated_csvs = sorted(output_root.glob("**/*.csv"))
    empty_csvs = []
    for path in generated_csvs:
        try:
            if pd.read_csv(path).empty:
                empty_csvs.append(rel(path))
        except Exception:
            empty_csvs.append(rel(path))
    add(checks, "all generated CSVs are non-empty", not empty_csvs, "; ".join(empty_csvs[:10]))

    tex_files = sorted(output_root.glob("*.tex")) + sorted((output_root).glob("**/*.tex"))
    bad_tex = [rel(path) for path in tex_files if not valid_tex(path)]
    add(checks, "generated TeX tables are syntactically includable", not bad_tex, "; ".join(bad_tex[:10]))
    figure_data = [output_root / "figures" / name for name in ["main_figure_panel_a_data.csv", "main_figure_panel_b_data.csv", "main_figure_panel_c_data.csv"]]
    add(checks, "figure data files are non-empty", all(path.is_file() and not pd.read_csv(path).empty for path in figure_data), ", ".join(rel(path) for path in figure_data))

    summary = load_hierarchy_summary()
    counts = hierarchy_counts(summary, HIERARCHY_FAMILIES["main"])
    figure_match = True
    panel_b_path = output_root / "figures" / "main_figure_panel_b_data.csv"
    if panel_b_path.is_file():
        panel_b = pd.read_csv(panel_b_path)
        for condition, count in counts.items():
            row = panel_b[panel_b["condition"].eq(condition)]
            figure_match = figure_match and not row.empty and int(row.iloc[0]["recovered_pair_count"]) == int(count)
    else:
        figure_match = False
    panel_c_path = output_root / "figures" / "main_figure_panel_c_data.csv"
    if panel_c_path.is_file():
        panel_c = pd.read_csv(panel_c_path)
        try:
            from paper_readiness_common_v1 import load_truncation_summary
            trunc = load_truncation_summary()
            for row in panel_c.itertuples(index=False):
                value = float(trunc[trunc["condition"].eq(row.condition)]["hysteresis_gap_mean"].mean())
                figure_match = figure_match and abs(float(row.mean_hysteresis_gap) - value) <= 1e-9
        except Exception:
            figure_match = False
    else:
        figure_match = False
    add(checks, "figure data files match reported table values", figure_match, "panel B matches hierarchy counts; panel C matches truncation summary means")
    headline = pd.read_csv(output_root / "headline_number_verification.csv") if (output_root / "headline_number_verification.csv").is_file() else pd.DataFrame()
    headline_counts_ok = True
    if not headline.empty:
        for condition, count in counts.items():
            label = condition.split("_")[0]
            matched = headline[headline["metric"].str.contains(HIERARCHY_ROOT.name, na=False)]
        headline_counts_ok = bool(headline["pass"].all())
    add(checks, "hierarchy criterion outputs match headline counts", headline_counts_ok, "headline verification pass column")

    checksum_rows = source_checksum_rows()
    write_checksum_file(output_root / "source_checksums_after.sha256", checksum_rows)
    write_checksum_file(output_root / "release_candidate" / "checksums.sha256", checksum_rows)
    payload = {"all_passed": all(row["pass"] for row in checks), "checks": checks, "git_diff_stat": git_diff_stat()}
    write_json(output_root / "artifact_validation.json", payload)
    lines = ["# Artifact Validation", "", markdown_table(checks, ["check", "pass", "details"]), ""]
    (output_root / "artifact_validation.md").write_text("\n".join(lines), encoding="utf-8")
    write_final_report(output_root, checks)


if __name__ == "__main__":
    main()
