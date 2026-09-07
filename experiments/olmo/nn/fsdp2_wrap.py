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

import sys
from typing import Any

from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import ShardingStrategy, fully_shard
from torch.nn.parallel.distributed import DistributedDataParallel

# DEBUG, temporary: print each parameter's live requires_grad + identity at the
# exact moment DDP's own constructor runs (both wte's own replicate() instance
# and the top-level model's), to pin down why a param that reads
# requires_grad=True at apply_fsdp2_wrap() call time reads False by the time
# replicate()'s deferred lazy_init() constructs DistributedDataParallel.
_orig_ddp_init = DistributedDataParallel.__init__


def _debug_ddp_init(self, module, *args, **kwargs):
    if isinstance(module, nn.ParameterList):
        print(
            f"[PAW-1809 DEBUG] DDP.__init__ called with ParameterList of "
            f"{len(module)} params:",
            file=sys.stderr,
            flush=True,
        )
        for i, p in enumerate(module):
            print(
                f"[PAW-1809 DEBUG]   [{i}] shape={tuple(p.shape)} "
                f"requires_grad={p.requires_grad} id={id(p)}",
                file=sys.stderr,
                flush=True,
            )
    return _orig_ddp_init(self, module, *args, **kwargs)


DistributedDataParallel.__init__ = _debug_ddp_init


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
            if type(module).__name__ == "Embedding":
                print(
                    f"[PAW-1809 DEBUG] apply_fsdp2_wrap(wrap-time) module=Embedding "
                    f"params={[(n, p.requires_grad, id(p)) for n, p in module.named_parameters(recurse=True)]}",
                    file=sys.stderr,
                    flush=True,
                )
            if not _has_trainable_params(module):
                return module
            return replicate(module, **kwargs)
        return [
            replicate(m, **kwargs) if _has_trainable_params(m) else m for m in module
        ]
    return fully_shard(module, **kwargs)
