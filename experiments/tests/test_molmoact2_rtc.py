"""Training-time real-time chunking: per-step flow timesteps and the pinned action prefix."""

import pytest
import torch

from olmo.models.molmoact2.molmoact2 import (
    _per_step_timesteps,
    _pin_action_prefix,
    _rtc_prefix_mask,
    _sample_rtc_delays,
    _validate_rtc_delay_probs,
)
from olmo.nn.action_expert import ActionExpertConfig, SinusoidalTimeEmbedding


def _tiny_expert():
    torch.manual_seed(0)
    cfg = ActionExpertConfig(
        max_horizon=4,
        max_action_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        timestep_embed_dim=4,
        ffn_multiple_of=8,
    )
    expert = cfg.build(llm_dim=8, llm_kv_dim=4).eval()
    # The modulation layers are zero-initialised (AdaLN-zero), which makes the
    # timestep inert at init; random weights make its effect observable.
    with torch.no_grad():
        for parameter in expert.parameters():
            parameter.normal_(0.0, 0.1)
    return expert


def _expert_inputs(batch: int = 2, horizon: int = 4):
    torch.manual_seed(1)
    actions = torch.randn(batch, horizon, 3)
    kv_states = [(torch.randn(batch, 5, 4), torch.randn(batch, 5, 4))]
    return actions, kv_states


# -- per-step timesteps ---------------------------------------------------------


def test_time_embedding_embeds_each_step_when_given_one_timestep_per_step():
    embed = SinusoidalTimeEmbedding(4)
    per_sample = torch.tensor([0.2, 0.7])
    per_step = torch.tensor([[0.2, 1.0], [0.7, 1.0]])

    sample_embed = embed(per_sample)
    step_embed = embed(per_step)

    assert step_embed.shape == (2, 2, 4)
    assert torch.equal(step_embed[:, 0], sample_embed)
    assert torch.equal(step_embed[:, 1], embed(torch.ones(2)))


def test_time_embedding_refuses_shapes_it_would_otherwise_collapse():
    with pytest.raises(ValueError, match="Flow timesteps must be"):
        SinusoidalTimeEmbedding(4)(torch.zeros(2, 3, 1))


def test_expert_with_broadcast_per_step_timesteps_matches_per_sample_timesteps():
    expert = _tiny_expert()
    actions, kv_states = _expert_inputs()
    t = torch.tensor([0.3, 0.6])

    per_sample = expert(actions, t, encoder_kv_states=kv_states)
    per_step = expert(
        actions, t[:, None].expand(-1, actions.shape[1]), encoder_kv_states=kv_states
    )

    torch.testing.assert_close(per_step, per_sample)


def test_expert_conditions_each_step_on_its_own_timestep():
    expert = _tiny_expert()
    actions, kv_states = _expert_inputs()
    t = torch.tensor([0.3, 0.6])

    per_sample = expert(actions, t, encoder_kv_states=kv_states)
    pinned = expert(
        actions,
        _per_step_timesteps(t, action_horizon=actions.shape[1], delay=1),
        encoder_kv_states=kv_states,
    )

    # The clean prefix step is conditioned differently; the rest sees the same
    # timestep but attends to a differently-modulated neighbour, so it may move too.
    assert not torch.allclose(pinned[:, 0], per_sample[:, 0])


def test_expert_refuses_timesteps_that_match_neither_samples_nor_steps():
    expert = _tiny_expert()
    actions, kv_states = _expert_inputs()

    with pytest.raises(ValueError, match="Flow timesteps must be"):
        expert(actions, torch.zeros(2, 3), encoder_kv_states=kv_states)


# -- the pinned prefix ----------------------------------------------------------


def test_per_step_timesteps_are_clean_on_the_prefix_and_the_flow_time_after():
    t = torch.tensor([0.25, 0.5])

    per_step = _per_step_timesteps(t, action_horizon=4, delay=2)

    assert torch.equal(
        per_step, torch.tensor([[1.0, 1.0, 0.25, 0.25], [1.0, 1.0, 0.5, 0.5]])
    )


def test_pin_action_prefix_replaces_exactly_the_prefix_rows():
    trajectory = torch.zeros(1, 4, 3)
    prefix = torch.ones(1, 2, 3)

    pinned = _pin_action_prefix(trajectory, prefix)

    assert torch.equal(pinned[:, :2], prefix)
    assert torch.equal(pinned[:, 2:], trajectory[:, 2:])


def test_prefix_mask_marks_the_first_delay_steps_per_example():
    mask = _rtc_prefix_mask(torch.tensor([0, 2]), action_horizon=3)

    assert torch.equal(
        mask, torch.tensor([[False, False, False], [True, True, False]])
    )


def test_sampled_delays_only_take_values_with_mass():
    torch.manual_seed(0)
    delays = _sample_rtc_delays((0.75, 0.0, 0.0, 0.25), batch_size=512, device="cpu")

    assert set(delays.tolist()) == {0, 3}
    assert 0.15 < (delays == 3).float().mean().item() < 0.35


@pytest.mark.parametrize(
    ("probs", "message"),
    [
        ((1.0,), "2..4 entries"),
        ((0.5, 0.0, 0.0, 0.0, 0.5), "2..4 entries"),
        ((0.75, -0.25, 0.5), "non-negative"),
        ((0.5, 0.5, 0.0), "last entry"),
        ((0.5, 0.4), "sum to 1"),
    ],
)
def test_delay_probs_that_are_not_a_distribution_below_the_horizon_are_refused(
    probs, message
):
    with pytest.raises(ValueError, match=message):
        _validate_rtc_delay_probs(probs, action_horizon=4)
