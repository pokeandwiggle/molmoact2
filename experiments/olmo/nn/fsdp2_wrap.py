"""Shared FSDP2 wrap helper, so ``apply_fsdp2`` call sites don't each re-derive
the FULL_SHARD vs. NO_SHARD branch.

``fully_shard()`` and ``replicate()`` aren't drop-in substitutes for each
other: ``replicate()`` (composable DDP) doesn't accept ``mp_policy`` --
compute precision under it comes from the trainer's own
``torch.autocast(...)`` around the forward pass, not from an FSDP2-specific
mixed-precision policy, so dropping ``mp_policy`` here is a deliberate
omission, not an oversight.

``replicate()`` also refuses a module with no trainable parameters --
DDP-style gradient sync has nothing to do for one, and it raises rather
than no-op. ``fully_shard()`` has no such restriction; it shards a frozen
module's parameters same as any other. Partial fine-tuning (LoRA, frozen
VLM backbone) routinely wraps frozen submodules this way, so under
NO_SHARD those are left unwrapped instead -- a frozen module contributes
no gradients to synchronize either way, so skipping replicate() on it is
a no-op change in behavior, not a gap.
"""

from __future__ import annotations

from typing import Any

from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import ShardingStrategy, fully_shard


def _has_trainable_params(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def apply_fsdp2_wrap(module: Any, **kwargs: Any) -> Any:
    """``fully_shard(module, **kwargs)``, or ``replicate(module, ...)`` under NO_SHARD.

    ``kwargs`` is exactly what ``FSDPConfig.get_fsd2_args()`` returns, plus the
    ``sharding_strategy`` field it now also carries -- popped here rather than
    forwarded, since neither composable API takes it directly.

    ``fully_shard`` accepts either one module or a list of modules (grouping
    them for reshard/prefetch scheduling); ``replicate`` only ever accepts one
    module -- it has no such grouping concept, so a list is applied
    element-by-element rather than passed through as-is.
    """
    strategy = kwargs.pop("sharding_strategy", ShardingStrategy.FULL_SHARD)
    if strategy == ShardingStrategy.NO_SHARD:
        kwargs.pop("mp_policy", None)
        if isinstance(module, nn.Module):
            if not _has_trainable_params(module):
                return module
            return replicate(module, **kwargs)
        return [
            replicate(m, **kwargs) if _has_trainable_params(m) else m for m in module
        ]
    return fully_shard(module, **kwargs)
