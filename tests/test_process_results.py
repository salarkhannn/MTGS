import csv

from scripts.process_results import compute_improvements, summarize_run, write_markdown_table


def test_process_results_summarizes_run_and_writes_markdown(tmp_path) -> None:
    run_dir = tmp_path / "B1_baseline_1n_rep1"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("mode: baseline\nfault_profile: none\n", encoding="utf-8")
    with (run_dir / "throughput_rank0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tokens_per_second", "loss"])
        writer.writeheader()
        writer.writerow({"tokens_per_second": "0", "loss": "2.0"})
        writer.writerow({"tokens_per_second": "100", "loss": "1.5"})

    rows = compute_improvements([summarize_run(run_dir)])
    table = tmp_path / "summary.md"
    write_markdown_table(rows, table)

    assert rows[0]["mean_tokens_per_second"] == 100.0
    assert "Local Smoke Results" in table.read_text(encoding="utf-8")
