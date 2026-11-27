"""Auxiliary KL on frozen, successful C749 training histories."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from submission.policy import HEADS


def stack_logits(policy, histories, valid):
    """Encode intact packed frames and apply the episode-start mask after encoding."""
    encoded = policy.encode(histories.flatten(0, 1)).unflatten(0, histories.shape[:2])
    encoded = encoded * valid[..., None]
    output, _ = policy.advance(encoded[:, -1], encoded[:, :-1].flatten(1))
    return policy.actor(output)


def head_kl(learner, teacher):
    return sum(F.kl_div(new.log_softmax(-1), old.log_softmax(-1), reduction="batchmean", log_target=True)
               for new, old in zip(learner.split(HEADS, -1), teacher.split(HEADS, -1), strict=True))


class AnchorSampler:
    def __init__(self, path, expected_sha256, seed):
        path = Path(path)
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Frozen anchor dataset SHA-256 differs.")
        with np.load(path, allow_pickle=False) as data:
            frames, valid, logits, categories = (data[key] for key in ("histories", "valid", "teacher_logits", "category"))
            count = len(frames)
            if (not count or frames.shape != (count, 4, 378) or logits.shape != (count, sum(HEADS))
                    or valid.shape != (count, 4) or categories.shape != (count,)
                    or not np.isfinite(frames).all() or not np.isfinite(logits).all()
                    or not np.isin(valid, (0, 1)).all() or not valid[:, -1].all()
                    or np.any(valid[:, :-1] > valid[:, 1:])):
                raise ValueError("Invalid frozen anchor shape, history mask, or teacher outputs.")
            self.frames = torch.tensor(frames, dtype=torch.float32)
            self.valid = torch.tensor(valid, dtype=torch.float32)
            self.teacher = torch.tensor(logits, dtype=torch.float32)
            self.strata = {str(name): np.flatnonzero(categories == name) for name in np.unique(categories)}
        if not {"key", "unlock", "exit"} <= set(self.strata):
            raise ValueError("Required independently recorded key/unlock/exit anchors are absent.")
        self.rng = np.random.default_rng(seed)

    def loss(self, policy, rows_per_category=8):
        indices = np.concatenate([self.rng.choice(rows, rows_per_category, replace=len(rows) < rows_per_category)
                                  for rows in self.strata.values()])
        return head_kl(stack_logits(policy, self.frames[indices], self.valid[indices]), self.teacher[indices])

    @torch.no_grad()
    def measure(self, policy):
        result = {}
        for name, rows in self.strata.items():
            losses = [head_kl(stack_logits(policy, self.frames[indices], self.valid[indices]), self.teacher[indices]).item()
                      * len(indices) for indices in np.array_split(rows, (len(rows) + 127) // 128)]
            result[name] = {"anchors": len(rows), "kl_nats": sum(losses) / len(rows)}
        return result
