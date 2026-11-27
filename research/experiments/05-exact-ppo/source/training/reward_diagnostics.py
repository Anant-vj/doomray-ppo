"""Component statistics and sampled policy-gradient directions."""

import numpy as np
import torch

from training.credit import COLUMNS, GAMMA, REWARDS


def current_context(current, carried):
    frames, ongoing = zip(*carried) if carried else ((), ())
    missing = 3 - len(frames)
    empty = torch.zeros_like(current)
    yes = current.new_ones(len(current))
    histories = torch.stack((empty,) * missing + frames + (current,), dim=1)
    valid = torch.stack((torch.zeros_like(yes),) * missing + ongoing + (yes,), dim=1)
    return histories, valid.flip(1).cumprod(1).flip(1)


def component_statistics(steps):
    array = steps.reshape(-1, len(COLUMNS))
    values = zip(array.sum(0), array.mean(0), array.var(0), (array != 0).sum(0), strict=True)
    return {name: dict(total=float(total), mean=float(mean), variance=float(var), nonzero=int(count))
            for name, (total, mean, var, count) in zip(COLUMNS, values, strict=True)}


def measured_gradients(policy, contexts, actions, components, boundaries):
    """Use on-policy reward-to-go for suffixes with an observed episode end."""
    import torch.nn.functional as F

    groups = {
        "combat_reward": ("native_map_kills", "dealt"),
        "player_damage": ("player_damage",),
        "progression": ("native_exit", "exit_bonus", "keys", "unlocks"),
        "survival": ("native_death", "damage", "healed"),
        "exploration": ("coverage",),
        "navigation_proxy": ("coverage", "doors"),
    }
    columns = np.stack([components[..., [COLUMNS.index(c) for c in group]].sum(-1)
                        for group in groups.values()], axis=-1)
    observed = np.zeros(columns.shape[:2], dtype=bool)
    returns = np.zeros_like(columns)
    following = np.zeros_like(columns[0])
    valid = np.zeros(columns.shape[1], dtype=bool)
    for t in range(len(columns) - 1, -1, -1):
        boundary = boundaries[t].astype(bool)
        valid |= boundary
        following = columns[t] + GAMMA * following * (~boundary[:, None])
        observed[t] = valid
        returns[t] = following
    count = int(observed.sum())
    diagnostics = {"complete_terminal_suffix_decisions": count, "total_rollout_decisions": int(observed.size),
                   "estimator": "Component Monte Carlo reward-to-go with zero baseline on observed complete suffixes."}
    if not count:
        return diagnostics
    mask = torch.as_tensor(observed)
    history, valid_frames = map(torch.stack, zip(*contexts))
    from submission.policy import HEADS
    from training.preservation import stack_logits

    logits = stack_logits(policy, history[mask], valid_frames[mask])
    selected = actions[mask]
    log_prob = sum(F.log_softmax(head, -1).gather(-1, selected[:, i, None]).squeeze(-1)
                   for i, head in enumerate(logits.split(HEADS, -1)))
    targets = torch.as_tensor(returns[observed], dtype=log_prob.dtype)
    parameters = tuple(policy.parameters())
    gradients = {}
    for i, name in enumerate(groups):
        grad = torch.autograd.grad(-(log_prob * targets[:, i]).mean(), parameters,
                                   retain_graph=i + 1 < len(groups), allow_unused=True)
        gradients[name] = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                                     for p, g in zip(parameters, grad, strict=True)])
    norms = {name: float(gradient.norm()) for name, gradient in gradients.items()}
    pairs = (("combat_reward", "progression"), ("combat_reward", "survival"),
             ("player_damage", "progression"), ("player_damage", "survival"),
             ("exploration", "progression"), ("navigation_proxy", "combat_reward"),
             ("navigation_proxy", "player_damage"))
    diagnostics["gradient_norms"] = norms
    diagnostics["cosine"] = {
        f"{first}_vs_{second}": float(torch.dot(gradients[first], gradients[second])) / (norms[first] * norms[second])
        if norms[first] * norms[second] > 1e-10 else None for first, second in pairs
    }
    diagnostics["mean_discounted_component_return"] = dict(zip(groups, targets.mean(0).tolist(), strict=True))
    return diagnostics
