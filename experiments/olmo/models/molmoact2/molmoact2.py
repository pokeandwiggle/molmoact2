import dataclasses
import logging
import math
import re
from dataclasses import field
from typing import (
    ClassVar,
    List,
    Optional,
    Sequence,
    Tuple,
    Iterator,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from olmo import tokenizer as tok
from olmo.config import D
from olmo.extra_tokens import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEPTH_OUTPUT_TOKEN,
    DEPTH_TOKENS,
)
from olmo.models.model import OLMoOutput
from olmo.data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from olmo.models.molmo2.molmo2 import Molmo2, Molmo2Config
from olmo.nn.action_expert import ActionExpert, ActionExpertConfig
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.preprocessing.multimodal_collator import MMCollator
from olmo.tokenizer import get_special_token_ids


log = logging.getLogger(__name__)


def _sample_beta_timesteps(
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Sample timesteps using a beta distribution with bounded support."""
    if cutoff < time_offset:
        raise ValueError(f"flow-matching cutoff must be >= time_offset, got {cutoff} < {time_offset}")
    if time_scale <= 0:
        raise ValueError(f"flow-matching time_scale must be > 0, got {time_scale}")
    upper = min(cutoff, time_offset + time_scale)
    dist = Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    scale = upper - time_offset
    samples = dist.sample((batch_size,))
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _validate_rtc_delay_probs(probs: Sequence[float], *, action_horizon: int) -> None:
    """Refuse an RTC delay distribution that is not a distribution over ``0..len-1 < horizon``."""
    if not 2 <= len(probs) <= action_horizon:
        raise ValueError(
            f"rtc_delay_probs covers delays 0..len-1 below action_horizon={action_horizon}, "
            f"so it needs 2..{action_horizon} entries, got {len(probs)}."
        )
    if min(probs) < 0:
        raise ValueError(f"rtc_delay_probs must be non-negative, got {tuple(probs)}.")
    if probs[-1] <= 0:
        raise ValueError(
            "the last entry of rtc_delay_probs must be positive: trailing zeros would hide "
            f"the largest trained delay, got {tuple(probs)}."
        )
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"rtc_delay_probs must sum to 1, got {sum(probs)}.")


def _sample_rtc_delays(
    probs: Sequence[float], *, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Draw one RTC delay per example: the number of leading actions pinned as a clean prefix."""
    weights = torch.as_tensor(probs, device=device, dtype=torch.float32)
    return torch.multinomial(weights, batch_size, replacement=True)


def _rtc_prefix_mask(delays: torch.Tensor, *, action_horizon: int) -> torch.Tensor:
    """``(batch, action_horizon)`` bool: True on the action steps pinned as the prefix."""
    steps = torch.arange(action_horizon, device=delays.device)
    return steps[None, :] < delays[:, None]


def _pin_action_prefix(trajectory: torch.Tensor, action_prefix: torch.Tensor) -> torch.Tensor:
    """``trajectory`` with its first ``action_prefix.shape[1]`` steps replaced by the prefix."""
    return torch.cat((action_prefix, trajectory[:, action_prefix.shape[1] :]), dim=1)


def _per_step_timesteps(timesteps: torch.Tensor, *, action_horizon: int, delay: int) -> torch.Tensor:
    """``(batch, action_horizon)`` flow timesteps: the prefix is clean (t=1), the rest is ``timesteps``."""
    per_step = timesteps[:, None].expand(-1, action_horizon)
    prefix = torch.arange(action_horizon, device=timesteps.device) < delay
    return torch.where(prefix[None, :], torch.ones_like(per_step), per_step)


@dataclasses.dataclass
class MolmoAct2Config(Molmo2Config):
    """Configuration for the MolmoAct2 model."""

    _model_name: ClassVar[str] = "molmoact2"

    max_action_dim: int = 32
    """Maximum dimensionality of each action vector after right-padding."""

    action_horizon: int = 30
    """Number of action steps predicted by the policy."""

    n_action_steps: Optional[int] = None
    """Deprecated checkpoint field. Inference defaults now come from per-tag metadata."""

    n_obs_steps: int = 1
    """Number of observation steps provided to the policy."""

    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    """Configuration for the diffusion-style action head."""

    add_action_expert: bool = True
    """If True, build the action expert branch. Disable for pure autoregressive pretraining."""

    action_expert_detach_vlm: bool = False
    """If True, stop gradients from the action expert branch from flowing back into VLM conditioning features."""

    action_expert_depth_gate: bool = False
    """If True, learn a per-example scalar gate that scales depth-token conditioning for the action expert."""

    action_expert_depth_gate_per_layer: bool = False
    """If True, learn one depth gate per selected action-expert conditioning layer."""

    action_expert_depth_gate_init_bias: float = -4.0
    """Initial depth gate logit. Negative values make the initial policy close to the no-depth path."""

    action_format: str = "continuous"
    """Action supervision mode: "continuous", "discrete", or "both"."""

    state_format: str = "discrete"
    """State conditioning mode: "continuous", "discrete", or "both"."""

    flow_matching_num_steps: int = 10
    """Number of integration steps during flow-matching inference."""

    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_flow_timesteps: int = 1
    """Number of timesteps/noise vectors to use per batch item during training."""

    rtc_delay_probs: Optional[Tuple[float, ...]] = None
    """Training-time real-time chunking (arXiv 2512.05964): condition each chunk on the
    actions the robot is already committed to. Entry ``i`` is the probability that a
    training example pins its first ``i`` actions as a clean prefix (held at t=1 and
    excluded from the loss), so the served model can continue a chunk mid-execution;
    ``len - 1`` is the largest trainable delay. ``None`` trains without RTC, bit-identical
    to before the field existed. At inference ``generate_actions(action_prefix=...)`` pins
    a prefix whose length must be a delay this distribution gave non-zero mass."""

    mask_action_chunk_padding: bool = True
    """Deprecated knob. Time padding from tag horizon to max horizon is always masked in training."""

    mask_action_dim_padding: bool = True
    """If True, exclude right-padded action dimensions from flow-matching dynamics and loss."""

    enable_depth_reasoning: bool = False
    """If True, activate depth-reasoning training/inference paths when the caller requests them."""

    num_depth_codes: int = 100
    """Number of depth code positions emitted per image (for example a 10x10 grid)."""

    depth_code_input_noise_rate: float = 0.0
    """Training-only fraction of teacher-forced depth code tokens replaced with random depth codes."""

    robot_processor: Optional[RobotProcessorConfig] = None
    """Shared normalization pipeline used to normalize/unnormalize actions and states."""

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = super().update_legacy_settings(config)
        if "action_dim" in config:
            legacy_dim = int(config["action_dim"])
            if "max_action_dim" in config and int(config["max_action_dim"]) != legacy_dim:
                raise ValueError(
                    "Found conflicting MolmoAct2 action dimensions in config: "
                    f"action_dim={legacy_dim} vs max_action_dim={int(config['max_action_dim'])}."
                )
            config["max_action_dim"] = legacy_dim
            del config["action_dim"]
        if "action_expert" in config and config.action_expert is not None:
            config.action_expert = ActionExpertConfig.update_legacy_settings(config.action_expert)
        if "action_expert_layer_mode" in config:
            value = str(config["action_expert_layer_mode"])
            if value != "per_layer":
                raise ValueError(
                    "MolmoAct2 action expert only supports per-layer conditioning; "
                    f"found legacy action_expert_layer_mode={value!r}."
                )
            del config["action_expert_layer_mode"]
        if "action_expert_condition_source" in config:
            value = str(config["action_expert_condition_source"])
            if value != "kv_cache":
                raise ValueError(
                    "MolmoAct2 action expert only supports KV-cache conditioning; "
                    f"found legacy action_expert_condition_source={value!r}."
                )
            del config["action_expert_condition_source"]
        if "robot_processor" not in config:
            if "robot_preprocessor" in config and config.robot_preprocessor is not None:
                config.robot_processor = config.robot_preprocessor
            elif "robot_postprocessor" in config and config.robot_postprocessor is not None:
                config.robot_processor = config.robot_postprocessor
        if "robot_preprocessor" in config:
            del config["robot_preprocessor"]
        if "robot_postprocessor" in config:
            del config["robot_postprocessor"]
        if "robot_processor" in config and config.robot_processor is not None:
            config.robot_processor = RobotProcessorConfig.update_legacy_settings(config.robot_processor)
        if "progress_token_value_encoding" in config:
            del config["progress_token_value_encoding"]
        if "progress_token_value_encoding_scale" in config:
            del config["progress_token_value_encoding_scale"]
        return config

    def build_model(self, device=None):
        return MolmoAct2(self, device)

    def build_collator(self, output_shapes, pad_mode: str, include_metadata=True) -> MMCollator:
        return MMCollator(
            get_special_token_ids(self.build_tokenizer()),
            output_shapes,
            include_metadata=include_metadata,
            pad=pad_mode,
            cp_enabled=self.cp_enabled,
            packed_action_shape=(self.action_horizon, self.max_action_dim),
        )


class MolmoAct2(Molmo2):
    """MolmoAct2 extends Molmo2 with an action diffusion head."""

    def __init__(self, config: MolmoAct2Config, device=None):
        super().__init__(config, device)
        valid_action_formats = {"continuous", "discrete", "both"}
        if config.action_format not in valid_action_formats:
            raise ValueError(
                f"Unknown action_format '{config.action_format}'. "
                f"Expected one of {sorted(valid_action_formats)}."
            )
        valid_state_formats = {"continuous", "discrete", "both"}
        if config.state_format not in valid_state_formats:
            raise ValueError(
                f"Unknown state_format '{config.state_format}'. "
                f"Expected one of {sorted(valid_state_formats)}."
            )
        if int(config.num_depth_codes) <= 0:
            raise ValueError(
                f"num_depth_codes must be > 0, got {config.num_depth_codes}."
            )
        if not 0.0 <= float(config.depth_code_input_noise_rate) <= 1.0:
            raise ValueError(
                "depth_code_input_noise_rate must be in [0, 1], "
                f"got {config.depth_code_input_noise_rate}."
            )
        if config.action_expert_depth_gate and not config.add_action_expert:
            raise ValueError("action_expert_depth_gate requires add_action_expert=True.")
        if config.action_expert_depth_gate_per_layer and not config.action_expert_depth_gate:
            raise ValueError("action_expert_depth_gate_per_layer requires action_expert_depth_gate=True.")
        if config.flow_matching_time_offset > config.flow_matching_cutoff:
            raise ValueError(
                "flow_matching_time_offset must be <= flow_matching_cutoff "
                f"(got {config.flow_matching_time_offset} > {config.flow_matching_cutoff})."
            )
        if config.flow_matching_time_scale <= 0:
            raise ValueError(
                "flow_matching_time_scale must be > 0 "
                f"(got {config.flow_matching_time_scale})."
            )
        if config.rtc_delay_probs is not None:
            _validate_rtc_delay_probs(config.rtc_delay_probs, action_horizon=config.action_horizon)
        self._action_start_token_id: Optional[int] = None
        self._action_end_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None
        self._depth_gate_token_ids: Tuple[int, ...] = ()
        if config.action_format == "both":
            tokenizer = self.config.build_tokenizer()
            action_start_ids = tokenizer.encode(ACTION_START_TOKEN)
            action_end_ids = tokenizer.encode(ACTION_END_TOKEN)
            if len(action_start_ids) != 1 or len(action_end_ids) != 1:
                raise ValueError(
                    "action_format='both' requires single-token <action_start>/<action_end>. "
                    "Enable action tokens in the tokenizer before using both-mode supervision."
                )
            self._action_start_token_id = int(action_start_ids[0])
            self._action_end_token_id = int(action_end_ids[0])
            eos_id = getattr(tokenizer, "eos_token_id", None)
            self._eos_token_id = None if eos_id is None else int(eos_id)
            log.info(
                "Masking discrete answer spans for action expert conditioning "
                "(action_start=%d, action_end=%d, eos=%s).",
                self._action_start_token_id,
                self._action_end_token_id,
                str(self._eos_token_id),
            )
        if config.action_expert_depth_gate:
            self._depth_gate_token_ids = self._resolve_depth_gate_token_ids()
        if config.action_expert.max_action_dim != config.max_action_dim:
            config.action_expert.max_action_dim = config.max_action_dim
        if config.action_expert.max_horizon < config.action_horizon:
            config.action_expert.max_horizon = config.action_horizon
        self.action_expert: Optional[ActionExpert]
        if config.add_action_expert:
            if config.action_expert.num_layers != self.config.llm.n_layers:
                raise ValueError(
                    f"Action expert depth ({config.action_expert.num_layers}) must match LLM layers "
                    f"({self.config.llm.n_layers}) when using per_layer conditioning."
                )
            llm_head_dim = (
                self.config.llm.head_dim
                if self.config.llm.head_dim is not None
                else self.config.llm.d_model // self.config.llm.n_heads
            )
            llm_kv_dim = self.config.llm.effective_n_kv_heads * llm_head_dim
            self.action_expert = config.action_expert.build(
                llm_dim=self.config.llm.d_model,
                llm_kv_dim=llm_kv_dim,
                llm_num_kv_heads=self.config.llm.effective_n_kv_heads,
                llm_num_layers=self.config.llm.n_layers,
                device=device,
            )
        else:
            self.action_expert = None
        self.action_expert_depth_gate: Optional[nn.Module]
        if config.action_expert_depth_gate:
            gate_input_dim = llm_kv_dim
            if config.action_expert_depth_gate_per_layer:
                num_gate_layers = len(self._require_action_expert().blocks)
                self.action_expert_depth_gate = nn.ModuleList(
                    nn.Linear(gate_input_dim, 1).to(device=device)
                    for _ in range(num_gate_layers)
                )
            else:
                self.action_expert_depth_gate = nn.Linear(gate_input_dim, 1).to(device=device)
            self.reset_action_expert_depth_gate_parameters()
        else:
            self.action_expert_depth_gate = None

    def reset_action_expert_depth_gate_parameters(self) -> None:
        if self.action_expert_depth_gate is None:
            return
        gates = (
            self.action_expert_depth_gate
            if isinstance(self.action_expert_depth_gate, nn.ModuleList)
            else [self.action_expert_depth_gate]
        )
        for gate in gates:
            if not isinstance(gate, nn.Linear):
                raise TypeError(f"Expected depth gate to be nn.Linear, got {type(gate).__name__}.")
            nn.init.zeros_(gate.weight)
            nn.init.constant_(
                gate.bias,
                float(self.config.action_expert_depth_gate_init_bias),
            )

    def _resolve_depth_gate_token_ids(self) -> Tuple[int, ...]:
        added_tokens = list(self.config.llm.tokenizer.resolve_new_tokens_for_both_input_and_output())
        depth_tokens = [DEPTH_OUTPUT_TOKEN]
        depth_tokens.extend(
            token
            for token in added_tokens
            if (
                token == DEPTH_TOKENS.start_token
                or token == DEPTH_TOKENS.end_token
                or DEPTH_TOKENS.parse_index(token) is not None
            )
        )
        tokenizer = self.config.build_tokenizer()
        token_ids = []
        for token in dict.fromkeys(depth_tokens):
            encoded = tokenizer.encode(token)
            if len(encoded) == 1:
                token_ids.append(int(encoded[0]))
        if not token_ids:
            raise ValueError(
                "action_expert_depth_gate=True requires depth tokens in the tokenizer."
            )
        return tuple(token_ids)

    def apply_activation_checkpointing(self):
        """Enable activation checkpointing on both the VLM backbone and action expert."""
        super().apply_activation_checkpointing()

    def _require_action_expert(self) -> ActionExpert:
        if self.action_expert is None:
            raise RuntimeError(
                "This MolmoAct2 instance was built with add_action_expert=False, so action expert "
                "training/generation is unavailable."
            )
        return self.action_expert

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,
        num_images: Optional[torch.Tensor] = None,
        multimodal_type: Optional[torch.Tensor] = None,
        num_image_starts: Optional[torch.Tensor] = None,
        response_logits_only: bool = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        collect_layer_hidden_states: bool = False,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_horizon_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        packed_batch_idx: Optional[torch.Tensor] = None,
        packed_example_ids: Optional[torch.Tensor] = None,
        packed_action_chunk_is_valid: Optional[torch.Tensor] = None,
        packed_num_chunks: Optional[torch.Tensor] = None,
        packed_action_chunk_cap: Optional[torch.Tensor] = None,
        packed_action_chunk_overflow: Optional[torch.Tensor] = None,
    ) -> OLMoOutput:
        """Run the base VLM and (optionally) compute the action loss."""
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        if action_horizon_is_pad is not None and action_is_pad is not None:
            raise ValueError("Provide only one of action_horizon_is_pad or legacy action_is_pad.")
        resolved_action_horizon_is_pad = (
            action_horizon_is_pad if action_horizon_is_pad is not None else action_is_pad
        )
        if actions is not None:
            self._require_action_expert()
            if actions.shape[1] != self.config.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.config.action_horizon}, got {actions.shape[1]}"
                )
            if actions.shape[-1] != self.config.max_action_dim:
                raise ValueError(
                    f"Expected max_action_dim {self.config.max_action_dim}, got {actions.shape[-1]}"
                )
            if states is None and self.config.state_format in {"continuous", "both"}:
                raise ValueError(
                    f"States must be provided when computing action losses with "
                    f"state_format='{self.config.state_format}'."
                )
        collect_layer_states = collect_layer_hidden_states
        collect_layer_input_states = False
        collect_block_output_states = collect_layer_states
        collect_layer_kv = actions is not None
        forward_kwargs = dict(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            labels=labels,
            loss_masks=loss_masks,
            images=images,
            image_masks=image_masks,
            token_pooling=token_pooling,
            low_res_token_pooling=low_res_token_pooling,
            num_images=num_images,
            multimodal_type=multimodal_type,
            num_image_starts=num_image_starts,
            response_logits_only=response_logits_only,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            append_last_valid_logits=append_last_valid_logits,
        )

        base_output, layer_states, layer_kv_states = self._run_backbone(
            collect_layer_hidden_states=collect_block_output_states,
            collect_layer_kv_states=collect_layer_kv,
            collect_layer_input_states=collect_layer_input_states,
            output_hidden_states=output_hidden_states,
            **forward_kwargs,
        )

        metrics = dict(base_output.metrics or {})
        internal = dict(base_output.internal or {})

        if actions is not None:
            encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
            if layer_kv_states is None:
                raise RuntimeError("Layer KV states are required for action training.")
            selected_layer_states = None
            selected_layer_kv_states = self._select_layer_kv_states(layer_kv_states)
            depth_gate, depth_mask = self._depth_gate_from_condition(
                input_ids=input_ids,
                encoder_attention_mask=encoder_attention_mask,
                layer_states=selected_layer_states,
                layer_kv_states=selected_layer_kv_states,
            )
            selected_layer_states = self._apply_depth_gate_to_layer_states(
                selected_layer_states,
                depth_mask,
                depth_gate,
            )
            selected_layer_kv_states = self._apply_depth_gate_to_layer_kv_states(
                selected_layer_kv_states,
                depth_mask,
                depth_gate,
            )
            flow_loss, velocity = self._compute_flow_matching_loss(
                actions=actions,
                layer_states=selected_layer_states,
                layer_kv_states=selected_layer_kv_states,
                encoder_attention_mask=encoder_attention_mask,
                action_horizon_is_pad=resolved_action_horizon_is_pad,
                action_dim_is_pad=action_dim_is_pad,
                states=states,
                packed_batch_idx=packed_batch_idx,
                packed_example_ids=packed_example_ids,
                packed_action_chunk_is_valid=packed_action_chunk_is_valid,
                subsegment_ids=subsegment_ids,
                num_flow_timesteps=self.config.num_flow_timesteps,
            )
            metrics["action_flow_loss"] = flow_loss.detach()
            internal["action_flow_loss"] = flow_loss
            internal["action_velocity"] = velocity
            if depth_gate is not None:
                metrics["action_expert_depth_gate"] = self._mean_depth_gate(depth_gate).detach()
                internal["action_expert_depth_gate"] = depth_gate

        return base_output._replace(metrics=metrics, internal=internal)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: Optional[torch.LongTensor],
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,
        num_images: Optional[torch.Tensor] = None,
        multimodal_type: Optional[torch.Tensor] = None,
        num_image_starts: Optional[torch.Tensor] = None,
        response_logits_only: bool = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        encoder_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        action_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate an action trajectory via flow-matching integration.

        ``action_prefix`` — ``(batch, delay, max_action_dim)`` in the model's normalized
        action space — pins the first ``delay`` steps of the chunk to actions the robot is
        already committed to (real-time chunking; requires a checkpoint trained with
        ``rtc_delay_probs`` that gave ``delay`` non-zero mass). ``None`` samples the chunk
        unconditioned, which is also the mode such a checkpoint trained at delay 0.
        """
        action_expert = self._require_action_expert()
        if states is None and self.config.state_format in {"continuous", "both"}:
            raise ValueError(
                f"States must be provided for action generation with "
                f"state_format='{self.config.state_format}'."
            )
        if encoder_kv_states is not None:
            layer_kv_states = self._select_layer_kv_states(encoder_kv_states)
            layer_states = None
            if encoder_attention_mask is None and input_ids is not None:
                encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        else:
            if input_ids is None:
                raise ValueError("input_ids must be provided when encoder conditioning is not precomputed.")
            encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
            forward_kwargs = dict(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                response_mask=response_mask,
                subsegment_ids=subsegment_ids,
                position_ids=position_ids,
                labels=labels,
                loss_masks=loss_masks,
                images=images,
                image_masks=image_masks,
                token_pooling=token_pooling,
                low_res_token_pooling=low_res_token_pooling,
                num_images=num_images,
                multimodal_type=multimodal_type,
                num_image_starts=num_image_starts,
                response_logits_only=response_logits_only,
                past_key_values=past_key_values,
                use_cache=use_cache,
                last_logits_only=last_logits_only,
                append_last_valid_logits=append_last_valid_logits,
            )
            collect_layer_input_states = False
            _, layer_states, layer_kv_states = self._run_backbone(
                collect_layer_hidden_states=False,
                collect_layer_kv_states=True,
                collect_layer_input_states=collect_layer_input_states,
                output_hidden_states=False,
                **forward_kwargs,
            )
            if layer_kv_states is None:
                raise RuntimeError("Failed to capture KV states for action generation.")
            layer_kv_states = self._select_layer_kv_states(layer_kv_states)
            layer_states = None
        depth_gate, depth_mask = self._depth_gate_from_condition(
            input_ids=input_ids,
            encoder_attention_mask=encoder_attention_mask,
            layer_states=layer_states,
            layer_kv_states=layer_kv_states,
        )
        layer_states = self._apply_depth_gate_to_layer_states(
            layer_states,
            depth_mask,
            depth_gate,
        )
        layer_kv_states = self._apply_depth_gate_to_layer_kv_states(
            layer_kv_states,
            depth_mask,
            depth_gate,
        )
        steps = num_steps or self.config.flow_matching_num_steps
        sample_source = layer_states[0] if layer_states is not None else layer_kv_states[0][0]
        batch_size = sample_source.shape[0]
        device = sample_source.device
        trajectory = torch.randn(
            (batch_size, self.config.action_horizon, self.config.max_action_dim),
            device=device,
            generator=generator,
        )
        trajectory = self._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        dt = 1.0 / steps
        layer_states, layer_kv_states = self._maybe_detach_action_expert_condition(
            layer_states,
            layer_kv_states,
        )
        if action_prefix is not None:
            action_prefix = self.validated_action_prefix(
                action_prefix,
                batch_size=batch_size,
                action_dim_is_pad=action_dim_is_pad,
            )
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            trajectory = self._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=self.config.mask_action_dim_padding,
            )
            if action_prefix is not None:
                # Hold the committed actions fixed and mark them as already clean (t=1).
                trajectory = _pin_action_prefix(trajectory, action_prefix)
                t = _per_step_timesteps(
                    t, action_horizon=trajectory.shape[1], delay=action_prefix.shape[1]
                )
            action_expert_kwargs = dict(
                encoder_kv_states=layer_kv_states,
                encoder_attention_mask=encoder_attention_mask,
                state_embeddings=states,
            )
            velocity = action_expert(
                trajectory,
                t,
                **action_expert_kwargs,
            )
            velocity = self._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=self.config.mask_action_dim_padding,
            )
            trajectory = trajectory + dt * velocity
            trajectory = self._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=self.config.mask_action_dim_padding,
            )
        if action_prefix is not None:
            # Return the committed actions exactly as given, not the integrator's copy.
            trajectory = _pin_action_prefix(trajectory, action_prefix)
        return trajectory

    def validated_action_prefix(
        self,
        action_prefix: torch.Tensor,
        *,
        batch_size: int,
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The committed prefix this checkpoint can condition on, padded dims zeroed.

        Refuses a prefix the checkpoint never trained for: no ``rtc_delay_probs`` at
        all, or a prefix length that distribution gave zero mass. Either would serve
        confidently wrong actions rather than fail.
        """
        probs = self.config.rtc_delay_probs
        if probs is None:
            raise ValueError(
                "action_prefix conditioning requires a checkpoint trained with "
                "rtc_delay_probs; this one was trained without real-time chunking."
            )
        expected_shape = (batch_size, None, self.config.max_action_dim)
        if action_prefix.dim() != 3 or action_prefix.shape[0] != batch_size or (
            action_prefix.shape[2] != self.config.max_action_dim
        ):
            raise ValueError(
                f"action_prefix must be (batch={batch_size}, delay, "
                f"max_action_dim={self.config.max_action_dim}), got shape "
                f"{tuple(action_prefix.shape)} (expected {expected_shape})."
            )
        delay = action_prefix.shape[1]
        if not 1 <= delay <= len(probs) - 1 or probs[delay] <= 0:
            raise ValueError(
                f"an action prefix of {delay} steps was never trained: "
                f"rtc_delay_probs={tuple(probs)} gives delay {delay} no mass."
            )
        return self._mask_action_dim_tensor(
            action_prefix,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )

    def _run_backbone(
        self,
        output_hidden_states: bool,
        collect_layer_hidden_states: bool,
        collect_layer_kv_states: bool,
        collect_layer_input_states: bool = False,
        **forward_kwargs,
    ) -> Tuple[
        OLMoOutput,
        Optional[Sequence[torch.Tensor]],
        Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        kwargs = dict(forward_kwargs)
        kwargs["collect_layer_hidden_states"] = collect_layer_hidden_states
        kwargs["collect_layer_kv_states"] = collect_layer_kv_states
        kwargs["output_hidden_states"] = output_hidden_states or collect_layer_input_states
        original_use_cache = bool(kwargs.get("use_cache", False))
        base_output = super().forward(**kwargs)
        internal = dict(base_output.internal or {})
        layer_states = internal.pop("layer_hidden_states", None)
        if collect_layer_input_states:
            hidden_states = base_output.hidden_states
            if hidden_states is None:
                raise RuntimeError("Backbone did not return hidden states for input-layer action conditioning.")
            num_layers = len(self.transformer.blocks)
            if len(hidden_states) < num_layers:
                raise RuntimeError(
                    "Backbone returned too few hidden-state tensors for input-layer action conditioning: "
                    f"got {len(hidden_states)}, expected at least {num_layers}."
                )
            layer_states = tuple(hidden_states[: num_layers + 1])
        layer_kv_states = base_output.attn_key_values if collect_layer_kv_states else None
        attn_key_values = base_output.attn_key_values if original_use_cache else None
        if not output_hidden_states:
            base_output = base_output._replace(hidden_states=None)
        base_output = base_output._replace(internal=internal, attn_key_values=attn_key_values)
        return base_output, layer_states, layer_kv_states

    def _get_encoder_attention_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool).clone()
        elif input_ids is not None:
            mask = (input_ids != -1)
        else:
            return None

        if self.config.action_format != "both" or input_ids is None:
            return mask

        eos_id = self._eos_token_id
        if eos_id is not None:
            mask &= (input_ids != eos_id)

        for batch_idx in range(input_ids.shape[0]):
            row_ids = input_ids[batch_idx]
            row_mask = mask[batch_idx]
            self._mask_discrete_output_span(
                row_ids,
                row_mask,
                self._action_start_token_id,
                self._action_end_token_id,
            )

        return mask

    def _get_depth_token_mask(
        self,
        input_ids: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.config.action_expert_depth_gate or input_ids is None:
            return None
        if not self._depth_gate_token_ids:
            return None
        depth_token_ids = torch.as_tensor(
            self._depth_gate_token_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        depth_mask = (input_ids.unsqueeze(-1) == depth_token_ids).any(dim=-1)
        if encoder_attention_mask is not None:
            depth_mask = depth_mask & encoder_attention_mask.to(device=input_ids.device, dtype=torch.bool)
        return depth_mask

    def _depth_gate_from_source(
        self,
        gate_head: nn.Linear,
        *,
        source: torch.Tensor,
        depth_mask: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if source.ndim == 4:
            source = source.reshape(source.shape[0], source.shape[1], -1)
        if source.ndim != 3:
            raise ValueError(
                f"Depth gate expected a sequence tensor with 3 dims after flattening, got {tuple(source.shape)}."
            )
        if source.shape[:2] != depth_mask.shape:
            raise ValueError(
                "Depth gate conditioning shape mismatch: "
                f"condition={tuple(source.shape)}, depth_mask={tuple(depth_mask.shape)}."
            )
        if encoder_attention_mask is not None:
            valid_mask = encoder_attention_mask.to(device=source.device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(depth_mask.shape, device=source.device, dtype=torch.bool)
        depth_mask = depth_mask.to(device=source.device, dtype=torch.bool)
        pool_mask = valid_mask & ~depth_mask
        # If a packed/corner-case row contains only depth tokens, fall back to all valid tokens.
        has_pool = pool_mask.any(dim=-1, keepdim=True)
        pool_mask = torch.where(has_pool, pool_mask, valid_mask)
        weights = pool_mask.to(dtype=source.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (source * weights).sum(dim=1) / denom
        gate_logits = gate_head(pooled.to(dtype=gate_head.weight.dtype))
        return torch.sigmoid(gate_logits).to(dtype=source.dtype)

    def _depth_gate_from_condition(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        layer_states: Optional[Sequence[torch.Tensor]],
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        gate_head = self.action_expert_depth_gate
        if gate_head is None:
            return None, None
        depth_mask = self._get_depth_token_mask(input_ids, encoder_attention_mask)
        if depth_mask is None:
            return None, depth_mask
        if layer_states is not None:
            sources = list(layer_states)
        elif layer_kv_states is not None:
            sources = [value for _, value in layer_kv_states]
        else:
            return None, depth_mask
        if isinstance(gate_head, nn.ModuleList):
            if len(gate_head) != len(sources):
                raise ValueError(
                    "Per-layer depth gate count mismatch: "
                    f"gates={len(gate_head)}, condition_layers={len(sources)}."
                )
            gates = [
                self._depth_gate_from_source(
                    gate,
                    source=source,
                    depth_mask=depth_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                for gate, source in zip(gate_head, sources)
            ]
            return gates, depth_mask
        if not isinstance(gate_head, nn.Linear):
            raise TypeError(
                f"Expected depth gate to be nn.Linear or nn.ModuleList, got {type(gate_head).__name__}."
            )
        gate = self._depth_gate_from_source(
            gate_head,
            source=sources[-1],
            depth_mask=depth_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return gate, depth_mask

    @staticmethod
    def _depth_gate_for_layer(
        gate: torch.Tensor | Sequence[torch.Tensor],
        layer_idx: int,
        *,
        num_layers: int,
    ) -> torch.Tensor:
        if isinstance(gate, torch.Tensor):
            return gate
        if len(gate) != num_layers:
            raise ValueError(f"Depth gate layer count mismatch: gates={len(gate)}, layers={num_layers}.")
        return gate[layer_idx]

    @staticmethod
    def _mean_depth_gate(gate: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(gate, torch.Tensor):
            return gate.mean()
        if not gate:
            raise ValueError("Cannot compute mean for an empty depth gate sequence.")
        return torch.stack([layer_gate.mean() for layer_gate in gate]).mean()

    def _apply_depth_gate_to_layer_states(
        self,
        layer_states: Optional[Sequence[torch.Tensor]],
        depth_mask: Optional[torch.Tensor],
        gate: Optional[torch.Tensor | Sequence[torch.Tensor]],
    ) -> Optional[Sequence[torch.Tensor]]:
        if layer_states is None or depth_mask is None or gate is None:
            return layer_states
        gated_states = []
        for layer_idx, hidden in enumerate(layer_states):
            layer_gate = self._depth_gate_for_layer(gate, layer_idx, num_layers=len(layer_states))
            mask = depth_mask.to(device=hidden.device, dtype=torch.bool)
            scale = torch.ones((*mask.shape, 1), device=hidden.device, dtype=hidden.dtype)
            scale = torch.where(
                mask.unsqueeze(-1),
                layer_gate.to(device=hidden.device, dtype=hidden.dtype).view(-1, 1, 1),
                scale,
            )
            gated_states.append(hidden * scale)
        return gated_states

    def _apply_depth_gate_to_layer_kv_states(
        self,
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
        depth_mask: Optional[torch.Tensor],
        gate: Optional[torch.Tensor | Sequence[torch.Tensor]],
    ) -> Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        if layer_kv_states is None or depth_mask is None or gate is None:
            return layer_kv_states
        gated_kv = []
        for layer_idx, (key, value) in enumerate(layer_kv_states):
            layer_gate = self._depth_gate_for_layer(gate, layer_idx, num_layers=len(layer_kv_states))
            mask = depth_mask.to(device=key.device, dtype=torch.bool)
            view_shape = [mask.shape[0], mask.shape[1]] + [1] * (key.ndim - 2)
            scale = torch.ones(view_shape, device=key.device, dtype=key.dtype)
            gate_view = layer_gate.to(device=key.device, dtype=key.dtype).view(
                layer_gate.shape[0],
                *([1] * (key.ndim - 1)),
            )
            scale = torch.where(mask.view(view_shape), gate_view, scale)
            gated_kv.append((key * scale, value * scale))
        return gated_kv

    @staticmethod
    def _mask_discrete_output_span(
        row_ids: torch.Tensor,
        row_mask: torch.Tensor,
        start_id: Optional[int],
        end_id: Optional[int],
    ) -> None:
        if start_id is None or end_id is None:
            return
        start_positions = (row_ids == start_id).nonzero(as_tuple=False).flatten().tolist()
        if not start_positions:
            return
        end_positions = (row_ids == end_id).nonzero(as_tuple=False).flatten().tolist()
        end_ptr = 0
        for start_pos in start_positions:
            while end_ptr < len(end_positions) and end_positions[end_ptr] < start_pos:
                end_ptr += 1
            if end_ptr >= len(end_positions):
                row_mask[start_pos:] = False
                break
            end_pos = end_positions[end_ptr]
            row_mask[start_pos:end_pos + 1] = False
            end_ptr += 1

    def _cache_to_sequence(self, cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() != 4:
            raise ValueError(f"Expected KV cache tensor with 4 dims, got shape {tuple(cache.shape)}")
        head_candidates = {self.config.llm.effective_n_kv_heads, self.config.llm.n_heads}
        # Standard cache shape: (B, n_heads, T, head_dim)
        if cache.shape[1] in head_candidates:
            bsz, n_heads, seq_len, head_dim = cache.shape
            return cache.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
        # CP variants may use (B, T, n_heads, head_dim)
        if cache.shape[2] in head_candidates:
            bsz, seq_len, n_heads, head_dim = cache.shape
            return cache.reshape(bsz, seq_len, n_heads * head_dim)
        # Fallback: assume the smaller of dims 1/2 is head count.
        if cache.shape[1] <= cache.shape[2]:
            bsz, n_heads, seq_len, head_dim = cache.shape
            return cache.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
        bsz, seq_len, n_heads, head_dim = cache.shape
        return cache.reshape(bsz, seq_len, n_heads * head_dim)

    def _select_layer_kv_states(
        self,
        layer_kv_states: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if not layer_kv_states:
            raise ValueError("No layer KV states provided for action expert conditioning.")
        action_expert = self._require_action_expert()
        kv_seq = [
            (self._cache_to_sequence(k), self._cache_to_sequence(v))
            for k, v in layer_kv_states
        ]
        num_target = len(action_expert.blocks)
        num_available = len(kv_seq)
        if num_available != num_target:
            raise ValueError(
                f"Expected {num_target} KV states, received {num_available} with per_layer mode."
            )
        return kv_seq

    def _chunk_attention_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        subsegment_ids: Optional[torch.Tensor],
        packed_batch_idx: torch.Tensor,
        packed_example_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.index_select(0, packed_batch_idx)
        else:
            mask = None
        if subsegment_ids is None:
            return mask
        example_assignments = subsegment_ids.index_select(
            0, packed_batch_idx
        ) // EXAMPLE_SUBSEGMENT_INCREMENT
        chunk_examples = packed_example_ids.view(-1, 1)
        chunk_mask = example_assignments == chunk_examples
        if mask is None:
            return chunk_mask
        return chunk_mask & mask

    def _maybe_detach_action_expert_condition(
        self,
        layer_states: Optional[Sequence[torch.Tensor]],
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[
        Optional[Sequence[torch.Tensor]],
        Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        if not self.config.action_expert_detach_vlm:
            return layer_states, layer_kv_states
        detached_layer_states = None
        detached_layer_kv_states = None
        if layer_states is not None:
            detached_layer_states = [hidden.detach() for hidden in layer_states]
        if layer_kv_states is not None:
            detached_layer_kv_states = [
                (key.detach(), value.detach()) for key, value in layer_kv_states
            ]
        return detached_layer_states, detached_layer_kv_states

    def _compute_flow_matching_loss(
        self,
        actions: torch.Tensor,
        layer_states: Optional[Sequence[torch.Tensor]],
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
        encoder_attention_mask: Optional[torch.Tensor],
        action_horizon_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        packed_batch_idx: Optional[torch.Tensor],
        packed_example_ids: Optional[torch.Tensor],
        packed_action_chunk_is_valid: Optional[torch.Tensor],
        subsegment_ids: Optional[torch.Tensor],
        num_flow_timesteps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_expert = self._require_action_expert()
        if (layer_states is None) == (layer_kv_states is None):
            raise ValueError("Provide exactly one of layer_states or layer_kv_states.")
        if layer_states is not None and len(layer_states) == 0:
            raise ValueError("Expected at least one layer state for action conditioning.")
        if layer_kv_states is not None and len(layer_kv_states) == 0:
            raise ValueError("Expected at least one layer KV state for action conditioning.")
        batch_size = actions.shape[0]
        device = actions.device
        if packed_batch_idx is None:
            packed_batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
        else:
            packed_batch_idx = packed_batch_idx.to(device=device, dtype=torch.long)
        if packed_example_ids is None:
            packed_example_ids = torch.zeros_like(packed_batch_idx)
        else:
            packed_example_ids = packed_example_ids.to(device=device, dtype=torch.long)
        if packed_action_chunk_is_valid is not None:
            packed_action_chunk_is_valid = packed_action_chunk_is_valid.to(device=device, dtype=torch.bool)
        if packed_batch_idx.numel() == 0:
            raise RuntimeError("Received empty action chunks for flow matching")
        source_tensor = (
            layer_states[0]
            if layer_states is not None
            else layer_kv_states[0][0]
        )
        batch_dim = source_tensor.shape[0]
        max_idx = int(packed_batch_idx.max().item())
        if max_idx >= batch_dim:
            raise RuntimeError(
                f"Action chunk batch index {max_idx} exceeds available layer states {batch_dim}"
            )

        chunk_layer_states = None
        chunk_layer_kv_states = None
        if layer_states is not None:
            chunk_layer_states = [
                hidden.index_select(0, packed_batch_idx) for hidden in layer_states
            ]
        else:
            assert layer_kv_states is not None
            chunk_layer_kv_states = [
                (
                    key.index_select(0, packed_batch_idx),
                    value.index_select(0, packed_batch_idx),
                )
                for key, value in layer_kv_states
            ]
        chunk_layer_states, chunk_layer_kv_states = self._maybe_detach_action_expert_condition(
            chunk_layer_states,
            chunk_layer_kv_states,
        )
        chunk_attention_mask = self._chunk_attention_mask(
            encoder_attention_mask,
            subsegment_ids,
            packed_batch_idx,
            packed_example_ids,
        )

        num_flow_timesteps = max(1, int(num_flow_timesteps))
        timesteps = _sample_beta_timesteps(
            batch_size=batch_size * num_flow_timesteps,
            device=device,
            cutoff=self.config.flow_matching_cutoff,
            time_offset=self.config.flow_matching_time_offset,
            time_scale=self.config.flow_matching_time_scale,
            alpha=self.config.flow_matching_beta_alpha,
            beta=self.config.flow_matching_beta_beta,
        )
        timesteps = timesteps.view(batch_size, num_flow_timesteps)
        t_broadcast = timesteps.view(batch_size, num_flow_timesteps, 1, 1)
        # Training-time RTC: each example pins its first `delay` actions as a clean
        # prefix — held at t=1 with the ground-truth actions in place of x_t, and
        # excluded from the loss — so the model learns to continue a committed chunk.
        rtc_delays = None
        rtc_prefix = None
        if self.config.rtc_delay_probs is not None:
            rtc_delays = _sample_rtc_delays(
                self.config.rtc_delay_probs, batch_size=batch_size, device=device
            )
            rtc_prefix = _rtc_prefix_mask(rtc_delays, action_horizon=actions.shape[1])
            rtc_prefix = rtc_prefix.view(batch_size, 1, actions.shape[1], 1)

        actions = self._mask_action_dim_tensor(
            actions,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        noise = torch.randn(
            batch_size,
            num_flow_timesteps,
            actions.shape[1],
            actions.shape[2],
            device=device,
            dtype=actions.dtype,
        )
        noise = self._mask_action_dim_tensor(
            noise,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        actions_expanded = actions.unsqueeze(1).expand(-1, num_flow_timesteps, -1, -1)
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions_expanded
        if rtc_prefix is not None:
            xt = torch.where(rtc_prefix, actions_expanded, xt)
        xt = self._mask_action_dim_tensor(
            xt,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        target_velocity = actions_expanded - noise
        target_velocity = self._mask_action_dim_tensor(
            target_velocity,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )

        xt_flat = xt.reshape(
            batch_size * num_flow_timesteps,
            actions.shape[1],
            actions.shape[2],
        )
        if rtc_prefix is None:
            timesteps_flat = timesteps.reshape(batch_size * num_flow_timesteps)
        else:
            # One flow timestep per action step: the prefix is clean (t=1).
            per_step = timesteps[:, :, None].expand(-1, -1, actions.shape[1])
            per_step = torch.where(rtc_prefix[..., 0], torch.ones_like(per_step), per_step)
            timesteps_flat = per_step.reshape(batch_size * num_flow_timesteps, actions.shape[1])
        chunk_layer_states_expanded = None
        chunk_layer_kv_states_expanded = None
        if chunk_layer_states is not None:
            chunk_layer_states_expanded = [
                hidden.unsqueeze(1)
                .expand(-1, num_flow_timesteps, -1, -1)
                .reshape(batch_size * num_flow_timesteps, hidden.shape[1], hidden.shape[2])
                for hidden in chunk_layer_states
            ]
        else:
            assert chunk_layer_kv_states is not None
            chunk_layer_kv_states_expanded = [
                (
                    key.unsqueeze(1)
                    .expand(-1, num_flow_timesteps, *([-1] * (key.ndim - 1)))
                    .reshape(batch_size * num_flow_timesteps, *key.shape[1:]),
                    value.unsqueeze(1)
                    .expand(-1, num_flow_timesteps, *([-1] * (value.ndim - 1)))
                    .reshape(batch_size * num_flow_timesteps, *value.shape[1:]),
                )
                for key, value in chunk_layer_kv_states
            ]
        if chunk_attention_mask is not None:
            chunk_attention_mask_expanded = (
                chunk_attention_mask.unsqueeze(1)
                .expand(-1, num_flow_timesteps, -1)
                .reshape(batch_size * num_flow_timesteps, chunk_attention_mask.shape[1])
            )
        else:
            chunk_attention_mask_expanded = None
        if states is not None:
            expand_shape = (-1, num_flow_timesteps, *([-1] * (states.ndim - 1)))
            states_expanded = (
                states.unsqueeze(1)
                .expand(*expand_shape)
                .reshape(batch_size * num_flow_timesteps, *states.shape[1:])
            )
        else:
            states_expanded = None
        if action_horizon_is_pad is not None:
            action_attention_mask = (~action_horizon_is_pad.to(device=device, dtype=torch.bool))
            action_attention_mask_expanded = (
                action_attention_mask.unsqueeze(1)
                .expand(-1, num_flow_timesteps, -1)
                .reshape(batch_size * num_flow_timesteps, action_attention_mask.shape[1])
            )
        else:
            action_attention_mask_expanded = None

        action_expert_kwargs = dict(
            encoder_kv_states=chunk_layer_kv_states_expanded,
            encoder_attention_mask=chunk_attention_mask_expanded,
            action_attention_mask=action_attention_mask_expanded,
            state_embeddings=states_expanded,
        )
        try:
            pred_velocity = action_expert(
                xt_flat,
                timesteps_flat,
                **action_expert_kwargs,
            )
        except RuntimeError as exc:
            exc_message = str(exc)
            if "Non-finite" in exc_message:
                bad_rows_match = re.search(r"(?:^|, )k_bad_batch_rows=\[([^\]]*)\]", exc_message)
                suspect_base_rows = None
                if bad_rows_match is not None:
                    raw_rows = [
                        int(part.strip())
                        for part in bad_rows_match.group(1).split(",")
                        if part.strip()
                    ]
                    suspect_base_rows = sorted({row // num_flow_timesteps for row in raw_rows})
                raise RuntimeError(
                    f"{exc_message}, num_flow_timesteps={num_flow_timesteps}, "
                    f"suspect_base_batch_rows={suspect_base_rows}, "
                    f"actions_shape={tuple(actions.shape)}."
                ) from exc
            raise
        pred_velocity = pred_velocity.reshape(
            batch_size,
            num_flow_timesteps,
            actions.shape[1],
            actions.shape[2],
        )
        pred_velocity = self._mask_action_dim_tensor(
            pred_velocity,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        if not torch.isfinite(pred_velocity).all():
            encoder_all_masked = None
            if chunk_attention_mask_expanded is not None:
                encoder_all_masked = bool((~chunk_attention_mask_expanded.to(torch.bool)).all(dim=-1).any().item())
            action_all_masked = None
            if action_attention_mask_expanded is not None:
                action_all_masked = bool((~action_attention_mask_expanded.to(torch.bool)).all(dim=-1).any().item())
            raise RuntimeError(
                "Non-finite action expert prediction in flow matching: "
                f"pred_velocity_finite={bool(torch.isfinite(pred_velocity).all().item())}, "
                f"actions_finite={bool(torch.isfinite(actions).all().item())}, "
                f"xt_finite={bool(torch.isfinite(xt_flat).all().item())}, "
                f"timesteps_finite={bool(torch.isfinite(timesteps_flat).all().item())}, "
                f"states_present={states_expanded is not None}, "
                f"state_finite={None if states_expanded is None else bool(torch.isfinite(states_expanded).all().item())}, "
                f"encoder_mask_present={chunk_attention_mask_expanded is not None}, "
                f"encoder_any_all_masked={encoder_all_masked}, "
                f"action_mask_present={action_attention_mask_expanded is not None}, "
                f"action_any_all_masked={action_all_masked}, "
                f"num_flow_timesteps={num_flow_timesteps}, "
                f"packed_valid_present={packed_action_chunk_is_valid is not None}, "
                f"actions_shape={tuple(actions.shape)}, "
                f"xt_shape={tuple(xt_flat.shape)}."
            )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = self._apply_action_chunk_padding_mask(
            loss,
            action_horizon_is_pad=action_horizon_is_pad,
            enabled=True,
        )
        if rtc_prefix is not None:
            # Supervise the postfix only. Rescaling by horizon / (horizon - delay) keeps the
            # plain mean over the chunk equal to the postfix-normalized mean of the paper.
            horizon = actions.shape[1]
            rescale = horizon / (horizon - rtc_delays).to(loss.dtype)
            loss = loss.masked_fill(rtc_prefix, 0.0) * rescale.view(batch_size, 1, 1, 1)
        loss = self._apply_action_dim_padding_mask(
            loss,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                "Non-finite unreduced flow matching loss: "
                f"loss_finite={bool(torch.isfinite(loss).all().item())}, "
                f"pred_velocity_finite={bool(torch.isfinite(pred_velocity).all().item())}, "
                f"target_velocity_finite={bool(torch.isfinite(target_velocity).all().item())}, "
                f"actions_finite={bool(torch.isfinite(actions).all().item())}, "
                f"num_flow_timesteps={num_flow_timesteps}, "
                f"actions_shape={tuple(actions.shape)}."
            )

        return self._reduce_flow_matching_loss(loss, packed_action_chunk_is_valid), pred_velocity.mean(dim=1)

    @staticmethod
    def _reduce_flow_matching_loss(
        loss: torch.Tensor,
        packed_action_chunk_is_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if packed_action_chunk_is_valid is None:
            return loss.mean()
        if packed_action_chunk_is_valid.ndim != 1:
            raise ValueError(
                "packed_action_chunk_is_valid must be a 1D tensor with one entry per packed action chunk."
            )
        if packed_action_chunk_is_valid.shape[0] != loss.shape[0]:
            raise ValueError(
                "packed_action_chunk_is_valid length must match the packed action batch size: "
                f"got {packed_action_chunk_is_valid.shape[0]} for batch {loss.shape[0]}."
            )
        valid = packed_action_chunk_is_valid.to(device=loss.device, dtype=loss.dtype)
        valid_count = valid.sum()
        if valid_count.item() <= 0:
            return loss.sum() * 0.0
        valid_view = valid.view(loss.shape[0], *([1] * (loss.ndim - 1)))
        denom = valid_count * math.prod(loss.shape[1:])
        return (loss * valid_view).sum() / denom

    @staticmethod
    def _apply_action_chunk_padding_mask(
        loss: torch.Tensor,
        action_horizon_is_pad: Optional[torch.Tensor],
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled or action_horizon_is_pad is None:
            return loss
        valid_action = (~action_horizon_is_pad).unsqueeze(1).unsqueeze(-1)
        return loss * valid_action

    @staticmethod
    def _action_dim_valid_mask(
        target: torch.Tensor,
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if action_dim_is_pad is None:
            return None
        mask = (~action_dim_is_pad.to(device=target.device, dtype=torch.bool))
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[-1] != target.shape[-1]:
            raise ValueError(
                "action_dim_is_pad width does not match target width: "
                f"mask={mask.shape[-1]}, target={target.shape[-1]}."
            )
        if mask.shape[0] == 1 and target.shape[0] != 1:
            mask = mask.expand(target.shape[0], -1)
        if mask.shape[0] != target.shape[0]:
            raise ValueError(
                "action_dim_is_pad batch size does not match target batch size: "
                f"mask={mask.shape[0]}, target={target.shape[0]}."
            )
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(1)
        return mask

    @classmethod
    def _mask_action_dim_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        action_dim_is_pad: Optional[torch.Tensor],
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled:
            return tensor
        valid_mask = cls._action_dim_valid_mask(tensor, action_dim_is_pad)
        if valid_mask is None:
            return tensor
        return tensor.masked_fill(~valid_mask, 0)

    @classmethod
    def _apply_action_dim_padding_mask(
        cls,
        loss: torch.Tensor,
        *,
        action_dim_is_pad: Optional[torch.Tensor],
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled:
            return loss
        valid_mask = cls._action_dim_valid_mask(loss, action_dim_is_pad)
        if valid_mask is None:
            return loss
        valid = valid_mask.to(dtype=loss.dtype)
        denom = valid.sum(dim=-1).clamp_min(1.0)
        return (loss * valid).sum(dim=-1) / denom

    def get_action_expert_parameters(self) -> Iterator[torch.Tensor]:
        params: List[torch.Tensor] = []
        if self.action_expert is not None:
            params.extend(self.action_expert.parameters())
        if self.action_expert_depth_gate is not None:
            params.extend(self.action_expert_depth_gate.parameters())
        return iter(params)
