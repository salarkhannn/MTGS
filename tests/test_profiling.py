from mtgs.profiling.ettr_timer import ETTRTimer
from mtgs.profiling.tracer import TraceRecorder


def test_ettr_timer_records_summary_and_csv(tmp_path) -> None:
    timer = ETTRTimer()
    timer.mark_detected("fault-1", rank=0, step=3, reason="timeout", timestamp=10.0)
    event = timer.mark_resumed("fault-1", timestamp=10.250)
    path = tmp_path / "ettr.csv"
    timer.write_events(path)

    assert event.ettr_ms == 250.0
    assert timer.summary()["count"] == 1.0
    assert "fault-1" in path.read_text(encoding="utf-8")


def test_trace_recorder_exports_json(tmp_path) -> None:
    recorder = TraceRecorder()
    with recorder.record("vote", rank=1, transaction_id="tx1", phase="2pc"):
        pass

    path = tmp_path / "trace.json"
    recorder.write_json(path)

    content = path.read_text(encoding="utf-8")
    assert "vote" in content
    assert "tx1" in content
