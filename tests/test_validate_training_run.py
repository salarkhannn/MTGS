import csv

from scripts.validate_training_run import validate_training_csv


def test_validate_training_csv_accepts_decreasing_loss(tmp_path) -> None:
    path = tmp_path / "throughput.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "loss", "tokens_per_second"],
        )
        writer.writeheader()
        writer.writerow({"step": 1, "loss": 2.0, "tokens_per_second": 0.0})
        writer.writerow({"step": 2, "loss": 1.5, "tokens_per_second": 100.0})

    result = validate_training_csv(path, min_steps=2, require_loss_decrease=True)

    assert result["passed"], result["errors"]
