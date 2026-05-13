import csv

from mtgs.trainer import main


def test_baseline_trainer_smoke_run(tmp_path) -> None:
    output_dir = tmp_path / "run"
    checkpoint = tmp_path / "baseline.pt"

    exit_code = main(
        [
            "--mode",
            "baseline",
            "--steps",
            "2",
            "--dataset-size",
            "16",
            "--batch-size",
            "4",
            "--seq-length",
            "8",
            "--vocab-size",
            "32",
            "--hidden-size",
            "16",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--checkpoint-path",
            str(checkpoint),
        ]
    )

    assert exit_code == 0
    assert checkpoint.exists()
    metrics = output_dir / "throughput_rank0.csv"
    assert metrics.exists()
    with metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[-1]["status"] == "ok"
