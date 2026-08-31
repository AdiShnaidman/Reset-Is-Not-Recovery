"""Regenerate public table CSVs from included aggregate artifacts."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TABLES = {
    "table2": ("results/raw_aggregates/bootstrap_ci/pressure_ramp_with_ci.csv", "results/main_tables/table2_pressure_ramp.csv"),
    "table3": ("results/raw_aggregates/behavioral_controls/full14_behavioral_controls_aggregate_summary.csv", "results/main_tables/table3_controls.csv"),
    "table4": ("results/raw_aggregates/recovery_hierarchy/expanded_14pair_recovery_counts.csv", "results/main_tables/table4_recovery_counts.csv"),
    "table5": ("results/raw_aggregates/truncation_boundary/expanded_truncation_summary.csv", "results/main_tables/table5_truncation_boundary.csv"),
    "table6": ("results/raw_aggregates/r7_evidence/full14_aggregate_summary.csv", "results/main_tables/table6_r7_evidence.csv"),
    "appendix": ("results/raw_aggregates/pressure_false_context/pressure_vs_false_context_aggregate_summary.csv", "results/appendix_tables/pressure_vs_false_context_aggregate_summary.csv"),
    "appendix_plausibility": ("results/raw_aggregates/wrong_answer_plausibility/wrong_answer_plausibility_aggregate_tertile.csv", "results/appendix_tables/wrong_answer_plausibility_aggregate_tertile.csv"),
    "appendix_relabel_restoration": ("results/raw_aggregates/non_history_clearing_relabel/restoration_metrics.csv", "results/appendix_tables/non_history_clearing_relabel_restoration_metrics.csv"),
    "appendix_relabel_outcome": ("results/raw_aggregates/non_history_clearing_relabel/outcome_metrics.csv", "results/appendix_tables/non_history_clearing_relabel_outcome_metrics.csv"),
    "appendix_relabel_semantic": ("results/raw_aggregates/non_history_clearing_relabel/semantic_letter_diagnostic.csv", "results/appendix_tables/non_history_clearing_relabel_semantic_letter.csv"),
}


def regenerate(table: str) -> Path:
    source, target = TABLES[table]
    source_path = ROOT / source
    target_path = ROOT / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    original = target_path.read_bytes() if target_path.exists() else None
    shutil.copyfile(source_path, target_path)
    if original is not None and original != target_path.read_bytes():
        raise RuntimeError(f"Regenerated {target} differs from the existing release artifact")
    return target_path


def verify(table: str) -> bool:
    source, target = TABLES[table]
    return filecmp.cmp(ROOT / source, ROOT / target, shallow=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=sorted(TABLES))
    parser.add_argument("--verify", action="store_true", help="Verify existing output instead of rewriting it.")
    args = parser.parse_args()
    if args.verify:
        if not verify(args.table):
            raise SystemExit(f"{args.table} does not match its source aggregate")
        print(f"{args.table}: verified")
        return
    path = regenerate(args.table)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
