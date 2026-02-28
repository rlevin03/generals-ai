"""
Agents package: game-agnostic RL agents, compatible with controllers.
"""

from .DQN_agent import (
    QNetwork,
    ReplayBuffer,
    select_action,
    train_dqn,
    ControllerProtocol,
    DEFAULT_STATE_HWC,
    DEFAULT_MAX_ACTIONS,
)

__all__ = [
    "QNetwork",
    "ReplayBuffer",
    "select_action",
    "train_dqn",
    "ControllerProtocol",
    "DEFAULT_STATE_HWC",
    "DEFAULT_MAX_ACTIONS",
]
