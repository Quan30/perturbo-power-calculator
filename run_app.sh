#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it works from any current directory.
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$app_dir"

# Keep user-provided paths if these variables have already been set.
export CORTADO_DATA_ROOT="${CORTADO_DATA_ROOT:-$app_dir/baselines}"
export CORTADO_GASPERINI_BASELINE="${CORTADO_GASPERINI_BASELINE:-$app_dir/baselines/gasperini_baseline_params.npz}"
export CORTADO_REPLOGLE_BASELINE="${CORTADO_REPLOGLE_BASELINE:-$app_dir/baselines/replogle_baseline_params.npz}"
export CORTADO_GASPERINI_PAIR_TABLE="${CORTADO_GASPERINI_PAIR_TABLE:-$app_dir/artifacts/GSE120861_gene_gRNAgroup_pair_table.at_scale.txt.gz}"
export CORTADO_GASPERINI_DETECTION_SUMMARY="${CORTADO_GASPERINI_DETECTION_SUMMARY:-$app_dir/artifacts/gene_detection_summary.parquet}"

exec python -m streamlit run "$app_dir/app.py" "$@"
