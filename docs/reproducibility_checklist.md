# Reproducibility Checklist

Date: 2026-07-06

## Commands

| target | command | expected output | status |
| --- | --- | --- | --- |
| Table 2 | `scripts/reproduce_table2_pressure_ramp.sh` | `results/main_tables/table2_pressure_ramp.csv` | matched |
| Table 3 | `scripts/reproduce_table3_controls.sh` | `results/main_tables/table3_controls.csv` | matched |
| Table 4 | `scripts/reproduce_table4_recovery_counts.sh` | `results/main_tables/table4_recovery_counts.csv` | matched |
| Table 5 | `scripts/reproduce_table5_truncation_boundary.sh` | `results/main_tables/table5_truncation_boundary.csv` | matched |
| Table 6 | `scripts/reproduce_table6_r7_evidence.sh` | `results/main_tables/table6_r7_evidence.csv` | matched |
| Appendix | `scripts/reproduce_appendix_tables.sh` | pressure-vs-false-context, wrong-answer plausibility, and non-history-clearing relabel diagnostic CSVs in `results/appendix_tables/` | matched |

## Known Limitations

- The default scripts reproduce tables from saved aggregate artifacts and do not rerun model inference.
- The R7 evidence-grounded release includes aggregate and pair-completion metadata, not private model caches or local run logs.
- The pressure-vs-false-context `user_pressure` baseline is imported from saved recovery-hierarchy ordinary-reset artifacts; the false-context controls are newly scored. See `docs/internal_audit/protocol_uniformity_audit.md`.
- Oversized item-level CCTP diagnostics and pressure-vs-false-context item rows are intentionally excluded from this GitHub release; aggregate CSVs needed by the table reproduction scripts are included.
