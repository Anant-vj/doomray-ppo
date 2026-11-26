"""Actor and critic for DevForge's 360-degree ray and masked entity contract."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

# Each categorical head selects one value on an independent control axis.
HEADS = (3, 3, 3, 2, 2, 2)
VITAL_SCALE = np.array([100, 100, 7, 100, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 20, 20, 20], np.float32)


def buttons(choice):
    choice = np.asarray(choice)
    return np.stack([choice[..., 0] == 1, choice[..., 0] == 2,
                     choice[..., 1] == 1, choice[..., 1] == 2,
                     choice[..., 2] == 1, choice[..., 2] == 2,
                     choice[..., 3], choice[..., 4], choice[..., 5]], axis=-1).astype(np.int8)


def pack(obs, previous):
    rays = obs["rays"]
    entities = obs["visible_entities"]
    features = np.array(entities["features"], dtype=np.float32, copy=True)
    features[:, 3:] /= 20
    return np.concatenate((rays["distance"], rays["hit_type"], features.ravel(),
                           entities["entity_type"], entities["mask"],
                           obs["vitals"] / VITAL_SCALE, previous)).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, memory="gru"):
        super().__init__()
        self.memory = memory
        self.ray_type = nn.Embedding(6, 4)
        self.rays = nn.Sequential(nn.Conv1d(5, 16, 5, 2, 2, padding_mode="circular"), nn.ReLU(),
                                  nn.Conv1d(16, 16, 5, 2, 2, padding_mode="circular"), nn.ReLU(),
                                  nn.AvgPool1d(2), nn.Flatten())
        self.entity_type = nn.Embedding(12, 6)
        self.entity = nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 24), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(128 + 48 + 26, 128), nn.LayerNorm(128), nn.Tanh())
        self.update = nn.GRUCell(128, 128) if memory == "gru" else nn.Sequential(
            nn.Linear(128 * 4, 128), nn.LayerNorm(128), nn.Tanh())
        self.actor = nn.Linear(128, sum(HEADS))
        self.critic = nn.Linear(128, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.02)
        nn.init.zeros_(self.actor.bias)

    def initial(self, n, device):
        return torch.zeros(n, 128 if self.memory == "gru" else 384, device=device)

    def encode(self, x):
        distance, hits, features, types, mask, context = x.split((64, 64, 160, 32, 32, 26), -1)
        ordered = torch.cat((distance.unsqueeze(-1), self.ray_type(hits.long())), -1)
        ray = self.rays(ordered.transpose(1, 2))
        entity = self.entity(torch.cat((features.reshape(-1, 32, 5).clamp(-8, 8),
                                        self.entity_type(types.long())), dim=-1))
        masked = entity * mask.unsqueeze(-1)
        mean = masked.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.fusion(torch.cat((ray, mean, masked.amax(1), context.clamp(-8, 8)), -1))

    def advance(self, encoded, state):
        if self.memory == "gru":
            output = self.update(encoded, state)
            return output, output
        stacked = torch.cat((state, encoded), -1)
        return self.update(stacked), stacked[:, 128:]

    def decision(self, output, choices=None):
        distributions = [Categorical(logits=head) for head in self.actor(output).split(HEADS, dim=-1)]
        if choices is None:
            choices = torch.stack([dist.sample() for dist in distributions], -1)
        logp = sum(dist.log_prob(choices[:, i]) for i, dist in enumerate(distributions))
        return choices, logp, sum(dist.entropy() for dist in distributions), self.critic(output).squeeze(-1)
