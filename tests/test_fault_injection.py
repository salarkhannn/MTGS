from mtgs.fault.detector import FailureDetector
from mtgs.fault.injector import RankProcess, inject_fault, select_target


def test_select_target_respects_round_robin_and_protected_rank() -> None:
    processes = [
        RankProcess(rank=0, pid=100),
        RankProcess(rank=1, pid=101),
        RankProcess(rank=2, pid=102),
    ]

    target = select_target(
        processes,
        policy="round_robin",
        iteration=1,
        protected_ranks={0},
    )

    assert target.rank == 2


def test_inject_fault_dry_run_does_not_kill() -> None:
    event = inject_fault(
        RankProcess(rank=1, pid=99999),
        policy="specific",
        dry_run=True,
        protected_pids=set(),
    )

    assert event.action == "dry_run_logged"
    assert event.target_rank == 1


def test_failure_detector_classifies_timeout() -> None:
    detector = FailureDetector(rank=0, timeout_s=0.001)

    signal = detector.detect_exception(TimeoutError("collective timeout"), step=7)

    assert signal.error_code == "collective_timeout"
    assert signal.step == 7
