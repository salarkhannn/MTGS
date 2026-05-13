import csv

from mtgs.profiling.throughput import ThroughputLogger


def test_throughput_logger_writes_warmup_and_measured_rows(tmp_path) -> None:
    path = tmp_path / "throughput.csv"
    logger = ThroughputLogger(path, rank=1, warmup_steps=1)

    start = logger.now_ns() - 1_000_000
    warmup = logger.record_step(epoch=0, step=1, tokens=100, start_ns=start, loss=2.0)
    measured = logger.record_step(epoch=0, step=2, tokens=100, start_ns=start, loss=1.5)

    assert warmup.warmup_excluded is True
    assert warmup.tokens_per_second == 0.0
    assert measured.warmup_excluded is False
    assert measured.tokens_per_second > 0.0

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"
