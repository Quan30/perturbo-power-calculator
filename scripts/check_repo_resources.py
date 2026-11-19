#!/usr/bin/env python3
"""Check the files that must be distributed with the app's Git repository."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = (
    'artifacts/GSE120861_gene_gRNAgroup_pair_table.at_scale.txt.gz',
    'artifacts/gene_detection_summary.parquet',
    'artifacts/analysis3_pair_power.parquet',
    'artifacts/gasperini_cis_power_calibration.json',
    'artifacts/gasperini_cis_increase_power_calibration_10rep.json',
    'artifacts/replogle_trans_decrease_power_calibration_10rep.json',
    'artifacts/calibration_pilot_summary_10rep.csv',
    'baselines/gasperini_baseline_params.npz',
    'baselines/replogle_baseline_params.npz',
    'data/K562_M2Kb_replogle_programs.parquet',
    'data/collectri_programs.tsv',
    'data/collectri_provenance.json',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-tracked', action='store_true', help='Also require each resource to be tracked or staged in Git.')
    args = parser.parse_args()
    failures = []
    for relative in RESOURCES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f'MISSING OR EMPTY: {relative}')
    in_git = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, cwd=ROOT).returncode == 0
    if not in_git and not args.check_tracked:
        if failures:
            print('\n'.join(failures), file=sys.stderr)
            return 1
        print(f'All {len(RESOURCES)} bundled resources exist. Initialize a Git checkout, stage files, then rerun with --check-tracked to audit Git inclusion.')
        return 0
    ignored = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], input='\n'.join(RESOURCES) + '\n', text=True, capture_output=True, cwd=ROOT)
    if ignored.returncode not in (0, 1):
        failures.append('Git ignore check failed: ' + ignored.stderr.strip())
    failures.extend('IGNORED: ' + path for path in ignored.stdout.splitlines())
    if args.check_tracked:
        for relative in RESOURCES:
            tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', relative], capture_output=True, text=True, cwd=ROOT)
            if tracked.returncode:
                failures.append('NOT TRACKED OR STAGED: ' + relative)
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'All {len(RESOURCES)} bundled resources exist and are not ignored' + ('; all are tracked or staged.' if args.check_tracked else '.'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
