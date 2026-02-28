"""
DQN Controller: interface between the agent-agnostic environment and the DQN agent.

Translates env state and valid actions into the format required by the DQN agent.
Translates agent output (action index) into an explicit action for the env.
Stores transitions in DQN training format after each step.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, List, Optional

from environment import GeneralsEnv


class DQNController:
    """
    Interface between the environment and the DQN agent.

    - Translates env state -> DQN input (tensor on device).
    - Translates env valid actions (explicit moves) -> list of indices for the agent.
    - Translates agent action index -> explicit action for env.
    - Stores (state, action_idx, reward, next_state, done) in DQN format for training.
    """

    def __init__(self, env: GeneralsEnv, device: torch.device):
        self.env = env
        self.device = device
        # Cache after reset/step
        self._state_raw: Optional[np.ndarray] = None
        self._valid_explicit: List[Tuple[int, int, int, int]] = []
        # Last transition for training (state, action_idx, reward, next_state, done)
        self._last_transition: Optional[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = None

    def _raw_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert raw env state to DQN tensor (batch=1, H, W, C)."""
        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return t

    def reset(self) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        """
        Reset env and return DQN-formatted state and valid action indices.
        Returns: (state_tensor, valid_action_indices, info).
        """
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
        """Current state in DQN format (1, H, W, C) tensor."""
        if self._state_raw is None:
            self._state_raw = self.env.get_state()
        return self._raw_to_tensor(self._state_raw)

    def get_valid_actions_for_agent(self) -> List[int]:
        """Valid actions as indices [0..n-1] for the agent."""
        self._valid_explicit = self.env.get_valid_actions_explicit()
        return list(range(len(self._valid_explicit)))

    def step(
        self, agent_action_idx: int
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Apply agent's action (index), step env, store transition, return result.
        If no valid actions or index out of range, performs no-op step.
        Returns: (next_state_tensor, reward, done, info).
        """
        state_tensor = self.get_state_for_agent()
        self._valid_explicit = self.env.get_valid_actions_explicit()

        explicit_action: Optional[Tuple[int, int, int, int]] = None
        if self._valid_explicit and 0 <= agent_action_idx < len(self._valid_explicit):
            explicit_action = self._valid_explicit[agent_action_idx]

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
        """
        Return the last (state, action_idx, reward, next_state, done) for replay.
        Returns None if no step has been taken yet.
        """
        return self._last_transition

    def is_done(self) -> bool:
        """Whether the current episode is done (e.g. game over or step limit)."""
        return self.env._is_done()
