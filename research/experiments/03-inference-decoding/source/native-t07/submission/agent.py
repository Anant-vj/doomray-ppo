"""DevForge inference entry point. Load the checkpoint selected on development maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from submission.policy import Policy, buttons, pack

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        if tuple(action_space.shape) != (9,):
            raise ValueError("This policy expects DevForge MultiBinary(9).")
        torch.set_num_threads(1)
        saved = torch.load(WEIGHTS_DIR / "policy.pt", map_location=device, weights_only=True)
        self.device = torch.device(device)
        self.policy = Policy(saved["memory"]).to(device).eval()
        self.policy.load_state_dict(saved["weights"])
        self.reset()

    def reset(self) -> None:
        self.state = self.policy.initial(1, self.device)
        self.previous = np.zeros(9, dtype=np.int8)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        frame = torch.as_tensor(pack(observation, self.previous), device=self.device)[None]
        output, self.state = self.policy.advance(self.policy.encode(frame), self.state)
        choice, _, _, _ = self.policy.decision(output)
        self.previous = buttons(choice[0].cpu().numpy())
        return self.previous.copy()

# Fixed T0.7 decoding for this isolated validation checkout.
from submission.decoding import configure_agent

DoomAgent = configure_agent(DoomAgent, "T07")
