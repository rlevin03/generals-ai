"""
Agents package: game-agnostic RL agents, compatible with controllers.
"""

from .DQN_agent import (
    QNetwork,
    ReplayBuffer,
    select_action,
    train_dqn,
    DQNAgent,
    ControllerProtocol,
    DEFAULT_STATE_HWC,
    DEFAULT_MAX_ACTIONS,
)
from .PPO_agent import PPOAgent, ActorCritic

__all__ = [
    "QNetwork",
    "ReplayBuffer",
    "select_action",
    "train_dqn",
    "DQNAgent",
    "ControllerProtocol",
    "DEFAULT_STATE_HWC",
    "DEFAULT_MAX_ACTIONS",
    "PPOAgent",
    "ActorCritic",
]
