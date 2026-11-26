"""Recurrent PPO with contiguous chunks and explicit episode-boundary masks."""
import torch


def advantages(rewards, values, following, boundaries, gamma=.99, decay=.95):
    carry = torch.zeros_like(values[-1])
    result = torch.empty_like(values)
    for t in reversed(range(len(rewards))):
        error = rewards[t] + gamma * following[t] - values[t]
        carry = error + gamma * decay * (1 - boundaries[t]) * carry
        result[t] = carry
    return result


def optimize(policy, optimizer, batch, chunk=16, epochs=3):
    obs, initial, starts, actions, old_logp, returns, adv = batch
    time, workers = starts.shape
    if time % chunk:
        raise ValueError("Rollout length must be a multiple of sequence length.")
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    offsets = torch.arange(chunk, device=obs.device)[:, None]
    groups = time * workers // chunk
    metrics = []
    for _ in range(epochs):
        for ids in torch.randperm(groups, device=obs.device).split(8):
            worker = ids % workers
            indices = ids.div(workers, rounding_mode="floor") * chunk + offsets
            hidden = initial[indices[0], worker]
            encoded = policy.encode(obs[indices, worker].flatten(0, 1)).reshape(chunk, len(ids), -1)
            outputs = []
            for i in range(chunk):
                hidden = hidden * (1 - starts[indices[i], worker, None])
                output, hidden = policy.advance(encoded[i], hidden)
                outputs.append(output)
            output = torch.cat(outputs)
            _, logp, entropy, value = policy.decision(output, actions[indices, worker].flatten(0, 1))
            ratio = (logp - old_logp[indices, worker].flatten()).exp()
            advantage = adv[indices, worker].flatten()
            actor = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            critic = (value - returns[indices, worker].flatten()).square().mean()
            loss = actor + .5 * critic - .01 * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            metrics.append((float(loss.detach()), float(grad)))
    return len(metrics), sum(row[0] for row in metrics) / len(metrics)
