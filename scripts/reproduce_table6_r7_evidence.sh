#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.table_generation table6
python3 -m src.table_generation table6 --verify
