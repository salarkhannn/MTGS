from mtgs.baseline import BaselineModelConfig, build_model, build_optimizer
from mtgs.hooks.comm_hook import MTGSState, mtgs_comm_hook
from mtgs.hooks.transaction import TransactionDecision
from mtgs.shadow.copy_stream import ShadowCopyManager


class FakeBucket:
    def __init__(self, tensor, index: int = 0) -> None:
        self._tensor = tensor
        self._index = index

    def buffer(self):
        return self._tensor

    def index(self) -> int:
        return self._index


def test_mtgs_comm_hook_commits_finite_bucket() -> None:
    import torch

    model = build_model(BaselineModelConfig(vocab_size=16, hidden_size=8))
    optimizer = build_optimizer(model, learning_rate=1e-3)
    state = MTGSState(
        model=model,
        optimizer=optimizer,
        shadow_manager=ShadowCopyManager(pin_memory=False),
    )
    state.start_step(1)
    bucket = FakeBucket(torch.ones(4))

    result = mtgs_comm_hook(state, bucket).wait()

    torch.testing.assert_close(result, torch.ones(4))
    assert state.last_transaction is not None
    assert state.last_transaction.decision == TransactionDecision.COMMIT


def test_mtgs_comm_hook_aborts_non_finite_bucket_and_rolls_back() -> None:
    import torch

    model = build_model(BaselineModelConfig(vocab_size=16, hidden_size=8))
    optimizer = build_optimizer(model, learning_rate=1e-3)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    state = MTGSState(
        model=model,
        optimizer=optimizer,
        shadow_manager=ShadowCopyManager(pin_memory=False),
    )
    state.start_step(1)
    bucket = FakeBucket(torch.tensor([float("nan"), 1.0]))

    result = mtgs_comm_hook(state, bucket).wait()

    torch.testing.assert_close(result, torch.zeros(2))
    assert state.last_transaction is not None
    assert state.last_transaction.decision == TransactionDecision.ABORT
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
