"""Training losses for a frozen teacher and reflected four-frame context."""

from copy import deepcopy

import torch
from torch.nn import functional as F

from submission.policy import HEADS
from training.reflection import reflect_logits, reflect_packed


def context(current, carried):
    """Recover raw frames and their validity from the existing rollout history."""
    frames, ongoing = zip(*carried) if carried else ((), ())
    padding = 3 - len(frames)
    missing_frame = torch.zeros_like(current)
    ones = current.new_ones(len(current))
    packed = torch.stack((missing_frame,) * padding + frames + (current,), dim=1)
    valid = torch.stack((torch.zeros_like(ones),) * padding + ongoing + (ones,), dim=1)
    return packed, valid.flip(1).cumprod(1).flip(1)


def stacked_logits(policy, frames, valid):
    encoded = policy.encode(frames.flatten(0, 1)).unflatten(0, frames.shape[:2])
    encoded = encoded * valid[..., None]
    output, _ = policy.advance(encoded[:, -1], encoded[:, :-1].flatten(1))
    return policy.actor(output)


def distribution(logits):
    return tuple(head.log_softmax(-1) for head in logits.split(HEADS, dim=-1))


def divergence(target, predicted):
    return sum(F.kl_div(q, p, reduction="batchmean", log_target=True) for p, q in zip(target, predicted))


class ReferenceRollout:
    def __init__(self, policy, teacher, histories, valid, reference_coef, symmetry_coef):
        self.policy = policy
        self.histories = histories
        self.valid = valid
        self.reference_coef = reference_coef
        self.symmetry_coef = symmetry_coef
        with torch.no_grad():
            self.reference = tuple(
                head.unflatten(0, histories.shape[:2])
                for head in distribution(stacked_logits(teacher, histories.flatten(0, 1), valid.flatten(0, 1)))
            )
        self.mirrored = reflect_packed(histories) if symmetry_coef else None

    def loss(self, indices, worker):
        frames = self.histories[indices, worker].flatten(0, 1)
        valid = self.valid[indices, worker].flatten(0, 1)
        original = distribution(stacked_logits(self.policy, frames, valid))
        target = tuple(head[indices, worker].flatten(0, 1) for head in self.reference)
        loss = self.reference_coef * divergence(target, original)
        if self.mirrored is not None:
            mirrored = distribution(reflect_logits(stacked_logits(
                self.policy, self.mirrored[indices, worker].flatten(0, 1), valid
            )))
            loss = loss + self.symmetry_coef * self.consistency(original, mirrored)
        return loss

    @staticmethod
    def consistency(original, mirrored):
        return .5 * (divergence(tuple(p.detach() for p in original), mirrored) +
                     divergence(tuple(p.detach() for p in mirrored), original))

    @torch.no_grad()
    def measure(self):
        valid = self.valid.flatten(0, 1)
        original = distribution(stacked_logits(self.policy, self.histories.flatten(0, 1), valid))
        reference = tuple(head.flatten(0, 1) for head in self.reference)
        # Measure both arms on the same real and reflected histories after each PPO update.
        frames = self.mirrored if self.mirrored is not None else reflect_packed(self.histories)
        mirrored = distribution(reflect_logits(stacked_logits(self.policy, frames.flatten(0, 1), valid)))
        return {"reference_kl_nats": float(divergence(reference, original)),
                "reflection_kl_nats": float(self.consistency(original, mirrored))}


def frozen_teacher(policy):
    return deepcopy(policy).eval().requires_grad_(False)
