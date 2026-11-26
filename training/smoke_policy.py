"""Offline regression check for PPO sequence indexing, reset masks, actions and checkpoint loading."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from submission import agent
from submission.policy import Policy, pack, buttons
from training.ppo import optimize, advantages


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    obs = {"rays": {"distance": np.ones(64), "hit_type": np.arange(64)%6},
           "visible_entities": {"features": np.zeros((32,5)), "entity_type": np.zeros(32),
                                "mask": np.zeros(32)}, "vitals": np.zeros(17)}
    frame = torch.from_numpy(pack(obs, np.zeros(9)))
    for mode in ("stack","gru"):
        model = Policy(mode)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        x = frame.repeat(32,3,1)
        x[4:,1,0] = .5
        starts = torch.zeros(32,3)
        starts[0] = 1
        starts[18,1] = 1
        h = model.initial(3,"cpu")
        rows = []
        with torch.no_grad():
            for t in range(32):
                h = h * (1-starts[t,:,None])
                o, hn = model.advance(model.encode(x[t]), h)
                a, lp, _, v = model.decision(o)
                rows.append((h,a,lp,v))
                h = hn
        init, a, lp, v = map(torch.stack,zip(*rows))
        assert buttons(a.numpy()).shape == (32,3,9)
        boundary = torch.zeros_like(v)
        boundary[17,1] = 1
        reward = torch.zeros_like(v)
        reward[18:,1] = 1
        adv = advantages(reward, v, torch.zeros_like(v), boundary)
        assert torch.isclose(adv[17,1], -v[17,1])
        old = model.actor.weight.detach().clone()
        n, loss = optimize(model,opt,(x,init,starts,a,lp,adv+v,adv),epochs=2)
        assert n and np.isfinite(loss) and not torch.equal(old,model.actor.weight)
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent,"WEIGHTS_DIR",Path(tmp)):
            torch.save(dict(memory=mode,weights=model.state_dict()), Path(tmp)/"policy.pt")
            candidate = agent.DoomAgent(None, SimpleNamespace(shape=(9,)), "cpu")
            action = candidate.act(obs)
            assert action.dtype == np.int8 and set(action.tolist()) <= {0,1}
            candidate.reset()
            assert not candidate.state.any() and not candidate.previous.any()
        print(mode, "synthetic optimizer steps", n, "finite loss", round(loss,5), "save/load/reset OK")


if __name__ == "__main__":
    main()
