"""
PPO Controller: interface between the agent-agnostic environment and the PPO agent.

Same interface as DQNController: state tensor, valid action indices, step by action index,
and last transition for training. PPO agent uses (state, action, reward, next_state, done)
and adds its own log_prob and value when storing transitions.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, List, Optional

from environment import GeneralsEnv


class PPOController:
    """
    Interface between the environment and the PPO agent.

    - Translates env state -> agent input (tensor on device).
    - Translates env valid actions (explicit moves) -> list of indices for the agent.
    - Translates agent action index -> explicit action for env.
    - Stores (state, action_idx, reward, next_state, done) for training (same as DQN).
    """

    def __init__(self, env: GeneralsEnv, device: torch.device):
        self.env = env
        self.device = device
        self._state_raw: Optional[np.ndarray] = None
        self._valid_explicit: List[Tuple[int, int, int, int]] = []
        self._last_transition: Optional[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = None

    def _raw_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert raw env state to tensor (batch=1, H, W, C)."""
        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return t

    def reset(self) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        """Reset env and return state tensor, valid action indices, info."""
        self._state_raw = self.env.reset()
        self._valid_explicit = self.env.get_valid_actions_explicit()
        state_tensor = self._raw_to_tensor(self._state_raw)
        valid_indices = list(range(len(self._valid_explicit)))
        info = {
            "territory": self.env._count_territory(),
            "army": self.env._count_army(),
        }
        return state_tensor, valid_indices, info

    def get_state_for_agent(self) -> torch.Tensor:
        """Current state as (1, H, W, C) tensor."""
        self._state_raw = self.env.get_state()
        return self._raw_to_tensor(self._state_raw)

    def get_valid_actions_for_agent(self) -> List[int]:
        """Valid actions as indices [0..n-1] for the agent."""
        self._valid_explicit = self.env.get_valid_actions_explicit()
        return list(range(len(self._valid_explicit)))

    def step(
        self, agent_action_idx: int
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Apply agent's action index, step env, store transition, return (next_state, reward, done, info)."""
        state_tensor = self.get_state_for_agent()
        self._valid_explicit = self.env.get_valid_actions_explicit()

        explicit_action: Optional[Tuple[int, int, int, int]] = None
        if self._valid_explicit:
            if 0 <= agent_action_idx < len(self._valid_explicit):
                explicit_action = self._valid_explicit[agent_action_idx]
            else:
                explicit_action = self._valid_explicit[0]

        next_state_raw, reward, done, info = self.env.step_explicit(explicit_action)
        next_state_tensor = self._raw_to_tensor(next_state_raw)

        self._state_raw = next_state_raw
        self._valid_explicit = self.env.get_valid_actions_explicit()

        self._last_transition = (
            state_tensor,
            agent_action_idx,
            reward,
            next_state_tensor,
            done,
        )
        return next_state_tensor, reward, done, info

    def get_last_transition(
        self,
    ) -> Optional[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        """Return last (state, action_idx, reward, next_state, done) for training."""
        return self._last_transition

    def is_done(self) -> bool:
        """Whether the current episode is done."""
        return self.env._is_done()
