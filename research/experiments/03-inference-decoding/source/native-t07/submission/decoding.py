"""Fixed evaluation-only decoding of the released C749 action heads."""

from functools import partial
from types import MappingProxyType


MODES = MappingProxyType({"T1-control": 1.0, "T07": 0.7, "T13": 1.3, "argmax": None})


def _decision(output, *, actor, critic, heads, temperature):
    import torch
    from torch.distributions import Categorical

    logits = actor(output).split(heads, dim=-1)
    distributions = [Categorical(logits=head if temperature is None else head / temperature)
                     for head in logits]
    if temperature is None:
        choices = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)
    else:
        choices = torch.stack([distribution.sample() for distribution in distributions], dim=-1)
    log_probability = sum(distribution.log_prob(choices[:, index])
                          for index, distribution in enumerate(distributions))
    entropy = sum(distribution.entropy() for distribution in distributions)
    return choices, log_probability, entropy, critic(output).squeeze(-1)


def configure_agent(base_actor, mode):
    """Return the unchanged control or an actor with one fixed decoding choice."""
    temperature = MODES[mode]
    if mode == "T1-control":
        return base_actor

    class FixedDecodingAgent(base_actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            policy = self.policy
            heads = type(policy).decision.__globals__["HEADS"]
            if heads != (3, 3, 3, 2, 2, 2):
                raise ValueError("The frozen C749 categorical heads differ.")
            policy.decision = partial(
                _decision, actor=policy.actor, critic=policy.critic, heads=heads, temperature=temperature
            )

    return FixedDecodingAgent
