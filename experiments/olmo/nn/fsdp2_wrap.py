"""Shared FSDP2 wrap helper, so ``apply_fsdp2`` call sites don't each re-derive
the FULL_SHARD vs. NO_SHARD branch.

``fully_shard()`` and ``replicate()`` aren't drop-in substitutes for each
other: ``replicate()`` (composable DDP) doesn't accept ``mp_policy`` --
compute precision under it comes from the trainer's own
``torch.autocast(...)`` around the forward pass, not from an FSDP2-specific
mixed-precision policy, so dropping ``mp_policy`` here is a deliberate
omission, not an oversight.
"""

from __future__ import annotations

from typing import Any

from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import ShardingStrategy, fully_shard


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
            return replicate(module, **kwargs)
        return [replicate(m, **kwargs) for m in module]
    return fully_shard(module, **kwargs)
