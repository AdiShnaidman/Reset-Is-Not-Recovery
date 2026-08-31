#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/reproduce_table2_pressure_ramp.sh
scripts/reproduce_table3_controls.sh
scripts/reproduce_table4_recovery_counts.sh
scripts/reproduce_table5_truncation_boundary.sh
scripts/reproduce_table6_r7_evidence.sh
scripts/reproduce_appendix_tables.sh
