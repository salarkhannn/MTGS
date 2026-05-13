from mtgs.hooks.transaction import TransactionDecision, TransactionManager


def test_transaction_manager_commits_single_process_yes_vote() -> None:
    manager = TransactionManager()

    result = manager.vote_and_decide(
        transaction_id="step1:bucket0",
        local_healthy=True,
    )

    assert result.decision == TransactionDecision.COMMIT
    assert result.votes == [1]
    assert result.committed is True


def test_transaction_manager_aborts_single_process_no_vote() -> None:
    manager = TransactionManager()

    result = manager.vote_and_decide(
        transaction_id="step1:bucket0",
        local_healthy=False,
        reason="non_finite_gradient",
    )

    assert result.decision == TransactionDecision.ABORT
    assert result.votes == [0]
    assert result.committed is False
    assert result.reason == "non_finite_gradient"
