from mtgs.baseline import (
    BaselineModelConfig,
    build_model,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_restores_model_parameters(tmp_path) -> None:
    import torch

    config = BaselineModelConfig(vocab_size=16, hidden_size=8, max_seq_length=4)
    model = build_model(config)
    optimizer = build_optimizer(model, learning_rate=1e-3)
    checkpoint = tmp_path / "baseline.pt"

    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        step=3,
    )
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    state = load_checkpoint(checkpoint, model=model, optimizer=optimizer)

    assert state["step"] == 3
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
