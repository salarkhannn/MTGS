import subprocess
import sys


def test_ddp_mtgs_smoke_script_runs_two_cpu_processes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ddp_mtgs_smoke.py",
            "--world-size",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "MTGS DDP smoke passed" in result.stdout
        or "MTGS DDP smoke skipped" in result.stdout
    )
