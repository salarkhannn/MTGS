import json
import subprocess
import sys


def test_run_experiment_dry_run_writes_metadata(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            "experiments/configs/full_matrix.yaml",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--limit",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "env_fingerprint.json").exists()


def test_churn_simulation_dry_run_writes_summary(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/churn_simulation.py",
            "--output-dir",
            str(tmp_path),
            "--level",
            "low",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "churn_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["level"] == "low"
    assert summary[0]["kill_interval_s"] == 600
