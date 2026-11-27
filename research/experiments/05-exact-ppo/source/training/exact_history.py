"""Rebuild four-decision policy context from packed frames for PPO updates."""


def stack_output(policy, histories, valid):
    """Apply the current learner to per-decision histories shaped [batch, 4, features]."""
    encoded = policy.encode(histories.flatten(0, 1)).unflatten(0, histories.shape[:2])
    # A zero input still yields a nonzero encoding. Mask encoded slots to
    # reproduce the agent's literal zero history at every episode reset.
    encoded = encoded * valid[..., None]
    output, _ = policy.advance(encoded[:, -1], encoded[:, :-1].flatten(1))
    return output
