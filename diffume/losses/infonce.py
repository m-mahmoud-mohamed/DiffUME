"""InfoNCE / CLIP-style symmetric contrastive loss.

Adapted from UME-R1's ``ClipLoss`` (sft-train/qwenvl/train/trainer.py
L449-L564) with the same gather-with-grad, local-loss semantics so the
loss matches UME-R1 exactly when ``world_size > 1``.  Falls back to a
simple in-batch contrastive loss when not in a distributed context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    import torch.distributed.nn  # noqa: F401  — for grad-aware all_gather
    _HAS_DIST = True
except ImportError:  # pragma: no cover
    _HAS_DIST = False


def _is_dist() -> bool:
    return _HAS_DIST and dist.is_available() and dist.is_initialized()


def _gather(query, target, local_loss=True, gather_with_grad=True):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if gather_with_grad:
        all_q = torch.cat(torch.distributed.nn.all_gather(query), dim=0)
        all_t = torch.cat(torch.distributed.nn.all_gather(target), dim=0)
    else:
        gq = [torch.zeros_like(query) for _ in range(world_size)]
        gt = [torch.zeros_like(target) for _ in range(world_size)]
        dist.all_gather(gq, query)
        dist.all_gather(gt, target)
        if not local_loss:
            gq[rank] = query
            gt[rank] = target
        all_q = torch.cat(gq, dim=0)
        all_t = torch.cat(gt, dim=0)
    return all_q, all_t


class ClipLoss(nn.Module):
    """Symmetric in-batch contrastive (InfoNCE) loss with optional all-gather."""

    def __init__(
        self,
        local_loss: bool = True,
        gather_with_grad: bool = True,
        cache_labels: bool = False,
    ) -> None:
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self._labels_cache: dict = {}
        self._prev_n: int = 0

    def _ground_truth(self, device, n):
        if self._prev_n != n or device not in self._labels_cache:
            labels = torch.arange(n, device=device, dtype=torch.long)
            if _is_dist() and self.local_loss:
                labels = labels + n * dist.get_rank()
            if self.cache_labels:
                self._labels_cache[device] = labels
                self._prev_n = n
            return labels
        return self._labels_cache[device]

    def forward(self, q: torch.Tensor, t: torch.Tensor, logit_scale: float = 50.0):
        if _is_dist():
            all_q, all_t = _gather(q, t, self.local_loss, self.gather_with_grad)
            if self.local_loss:
                logits_qt = logit_scale * q @ all_t.T
                logits_tq = logit_scale * t @ all_q.T
            else:
                logits_qt = logit_scale * all_q @ all_t.T
                logits_tq = logits_qt.T
        else:
            logits_qt = logit_scale * q @ t.T
            logits_tq = logits_qt.T

        labels = self._ground_truth(q.device, logits_qt.size(0))
        loss = (F.cross_entropy(logits_qt, labels) + F.cross_entropy(logits_tq, labels)) / 2.0
        return loss
