import torch

from k3 import K3Config, K3ForCausalLM
from k3.monitoring import TrainingMonitor


def tiny_config() -> K3Config:
    return K3Config(
        vocab_size=128,
        hidden_size=32,
        num_layers=5,
        num_heads=4,
        mla_latent_size=16,
        decay_rank=8,
        short_conv_kernel=3,
        attnres_block_size=2,
        latent_moe_size=16,
        routed_expert_hidden_size=16,
        shared_expert_hidden_size=20,
        num_routed_experts=4,
        num_shared_experts=1,
        experts_per_token=2,
    )


def test_attention_composition_matches_report_pattern() -> None:
    config = K3Config(num_layers=13)
    types = [config.attention_type(index) for index in range(config.num_layers)]
    assert types == [
        "kda",
        "kda",
        "kda",
        "mla",
        "kda",
        "kda",
        "kda",
        "mla",
        "kda",
        "kda",
        "kda",
        "mla",
        "mla",
    ]


def test_forward_backward_and_quantile_bias() -> None:
    torch.manual_seed(7)
    model = K3ForCausalLM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 8))
    labels = torch.randint(0, 128, (2, 8))
    output = model(input_ids, labels)
    assert output.logits.shape == (2, 8, 128)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None
    for layer in model.layers:
        assert torch.isclose(layer.moe.router_bias.mean(), torch.tensor(0.0), atol=1e-6)


def test_model_is_causal() -> None:
    torch.manual_seed(11)
    model = K3ForCausalLM(tiny_config()).eval()
    first = torch.randint(0, 128, (1, 8))
    second = first.clone()
    second[:, 5:] = torch.randint(0, 128, (1, 3))
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    torch.testing.assert_close(first_logits[:, :5], second_logits[:, :5], atol=2e-5, rtol=2e-5)


def test_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(13)
    model = K3ForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        full_logits = model(input_ids).logits
        cache = None
        step_logits = []
        for position in range(input_ids.size(1)):
            logits, cache = model.forward_step(
                input_ids[:, position : position + 1],
                cache,
            )
            step_logits.append(logits)
    cached_logits = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(full_logits, cached_logits, atol=3e-5, rtol=3e-5)


def test_50m_configuration_parameter_budget() -> None:
    config = K3Config.from_json("configs/k3_50m.json")
    model = K3ForCausalLM(config)
    parameters = model.num_parameters()
    assert 48_000_000 <= parameters <= 52_000_000, f"got {parameters:,}"


def test_training_monitor_writes_dashboard_artifacts(tmp_path) -> None:
    monitor = TrainingMonitor(tmp_path, model_parameters=49_019_770, total_steps=100)
    monitor.record_train(0, 9.1, 0.1, 123.0, 1.2)
    monitor.record_validation(0, 9.0)
    monitor.record_sample(0, "Once upon a time", "Once upon a time there was a fox.")
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "loss.png").stat().st_size > 1_000
    state = (tmp_path / "dashboard_state.json").read_text(encoding="utf-8")
    assert '"train_loss": 9.1' in state
    assert "there was a fox" in state
