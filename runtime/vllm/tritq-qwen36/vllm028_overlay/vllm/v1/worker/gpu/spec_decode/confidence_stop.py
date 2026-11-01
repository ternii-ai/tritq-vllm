# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Wai Chung, 2026-10 (Apache-2.0 section 4(b) notice): adds a confidence-stop draft depth for Model Runner V2
# speculative decoding, ported from vLLM pull request #60068 by @stecasta (Apache-2.0) as carried by
# salient-data/qwen38-next-dgx-recipe @f309019, recipe/patches/0010 (Copyright 2026 Salient Data, Apache-2.0),
# onto vLLM 0.28.0. Enabled only when the environment variable TQ_CSTOP="threshold,fallback" is set.
# Port of vLLM pull request #60068 (confidence stop for autoregressive drafting, @stecasta, Apache-2.0) as carried by
# salient-data/qwen38-next-dgx-recipe patch 0010, onto vLLM 0.28.0's Model Runner V2. Class body as in the patch; config via
# env TQ_CSTOP="thr,fallback".
import math
import os

import numpy as np
import torch


def cstop_config():
    v = os.environ.get("TQ_CSTOP", "")
    if not v:
        return None
    thr, fb = v.split(",")
    return float(thr), int(fb)


class DraftConfidenceStop:
    """Ends a single request's draft round once a draft's top-1 probability
    falls below the threshold; the first draft is always kept."""

    def __init__(self, threshold, fallback_depth, max_num_reqs, num_steps, device):
        self.fallback_depth = fallback_depth
        self.num_steps = num_steps
        self.log_threshold = math.log(threshold)
        self.alive = torch.ones(1, dtype=torch.bool, device=device)
        self._alive_cpu = torch.ones(1, dtype=torch.bool, pin_memory=True)
        self._alive_ready = torch.cuda.Event()
        self._num_drafts_np = np.full(max_num_reqs, num_steps, dtype=np.int32)
        self._slot = None
        self._pending_slot = None
        self._steps_launched = 0
        self.n_rounds = 0; self.n_drafts = 0                      # confidence-stop port: counters for the log

    def update(self, logits, draft_step):
        row = logits[:1].float()
        confident = (row.amax(-1) - row.logsumexp(-1)) >= self.log_threshold
        self.alive.copy_(confident & (self.alive | (draft_step == 0)))

    def begin_round(self, slot):
        self._resolve_pending()
        self._slot = slot
        self._steps_launched = 0

    def fixed_round(self, slots, num_drafts):
        self._resolve_pending()
        self._num_drafts_np[slots] = num_drafts
        self._slot = None

    def step_launched(self):
        self._steps_launched += 1
        self._alive_cpu.copy_(self.alive, non_blocking=True)
        self._alive_ready.record()

    def _read_alive(self):
        self._alive_ready.synchronize()
        return bool(self._alive_cpu[0])

    def should_continue(self):
        if self._read_alive():
            return True
        assert self._slot is not None
        self._num_drafts_np[self._slot] = max(1, self._steps_launched - 1)
        self.n_rounds += 1; self.n_drafts += self._num_drafts_np[self._slot]
        self._slot = None
        return False

    def end_round(self):
        self._pending_slot, self._slot = self._slot, None

    def _resolve_pending(self):
        if self._pending_slot is not None:
            survived = self._read_alive()
            self._num_drafts_np[self._pending_slot] = self.num_steps - (not survived)
            self.n_rounds += 1; self.n_drafts += self._num_drafts_np[self._pending_slot]
            self._pending_slot = None

    def num_verifiable_drafts(self, slots):
        self._resolve_pending()
        return self._num_drafts_np[slots]
