"""A Git checkout must contain every bundled runtime resource."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_calibration_resources_are_not_ignored(tmp_path):
    paths = [
        'artifacts/gasperini_cis_power_calibration.json',
        'artifacts/gasperini_cis_increase_power_calibration_10rep.json',
        'artifacts/replogle_trans_decrease_power_calibration_10rep.json',
        'artifacts/calibration_pilot_summary_10rep.csv',
    ]
    assert all((ROOT / path).is_file() for path in paths)
    # Also run from an extracted ZIP, which does not contain .git metadata.
    (tmp_path / '.gitignore').write_text((ROOT / '.gitignore').read_text())
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], input='\n'.join(paths) + '\n', text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode == 1, f'Required resources ignored by Git:\n{result.stdout}\n{result.stderr}'
