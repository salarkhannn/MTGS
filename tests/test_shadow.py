from mtgs.baseline import BaselineModelConfig, build_model, build_optimizer
from mtgs.shadow.copy_stream import ShadowCopyManager
from mtgs.shadow.rollback import rollback_model_state


def test_shadow_capture_and_rollback_restores_model_and_optimizer() -> None:
    import torch

    model = build_model(BaselineModelConfig(vocab_size=16, hidden_size=8))
    optimizer = build_optimizer(model, learning_rate=1e-3)
    manager = ShadowCopyManager(pin_memory=True)

    result = manager.capture(model, optimizer=optimizer)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    with torch.no_grad():
        for param in model.parameters():
            param.add_(3.0)

    rollback = rollback_model_state(model, result.snapshot, optimizer=optimizer, reason="test")

    assert rollback.restored_tensors == len(before)
    assert result.snapshot.total_bytes > 0
    assert result.snapshot.pinned is False
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
