from mtgs.trainer import main


def test_mtgs_trainer_smoke_run_with_local_shadow(tmp_path) -> None:
    output_dir = tmp_path / "mtgs"

    exit_code = main(
        [
            "--mode",
            "mtgs",
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
        ]
    )

    assert exit_code == 0
    assert (output_dir / "throughput_rank0.csv").exists()
    events = (output_dir / "train_rank0.jsonl").read_text(encoding="utf-8")
    assert "shadow_copied" in events


def test_mtgs_trainer_forced_abort_logs_rollback(tmp_path) -> None:
    output_dir = tmp_path / "mtgs_abort"

    exit_code = main(
        [
            "--mode",
            "mtgs",
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
            "--mtgs-force-abort-step",
            "1",
        ]
    )

    assert exit_code == 0
    events = (output_dir / "train_rank0.jsonl").read_text(encoding="utf-8")
    assert "rollback_complete" in events
    assert "step_aborted" in events
