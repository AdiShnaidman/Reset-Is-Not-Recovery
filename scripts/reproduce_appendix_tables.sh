#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.table_generation appendix
python3 -m src.table_generation appendix --verify
python3 -m src.table_generation appendix_plausibility
python3 -m src.table_generation appendix_plausibility --verify
python3 -m src.table_generation appendix_relabel_restoration
python3 -m src.table_generation appendix_relabel_restoration --verify
python3 -m src.table_generation appendix_relabel_outcome
python3 -m src.table_generation appendix_relabel_outcome --verify
python3 -m src.table_generation appendix_relabel_semantic
python3 -m src.table_generation appendix_relabel_semantic --verify
