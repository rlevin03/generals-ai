"""
Proximal Policy Optimization (PPO) Agent — same structure as DQN: controller-compatible.

Uses actor-critic with shared backbone. Collects (s, a, r, s', done, log_prob) and
trains with GAE, clipped surrogate loss, value loss, and entropy bonus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Optional, Any, Protocol

DEFAULT_STATE_HWC = (20, 25, 6)
DEFAULT_MAX_ACTIONS = 25 * 20 * 4


class ControllerProtocol(Protocol):
    """Protocol for controller: same interface as DQN/PPO controller."""

    def get_state_for_agent(self) -> torch.Tensor: ...
    def get_valid_actions_for_agent(self) -> List[int]: ...
    def step(self, agent_action_idx: int) -> Tuple[torch.Tensor, float, bool, Any]: ...
    def get_last_transition(
        self,
    ) -> Optional[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]: ...


class EnvProtocol(Protocol):
    """Protocol for env when agent interacts directly (no controller)."""

    def get_state_tensor(self, device: Optional[torch.device]) -> torch.Tensor: ...
    def get_valid_action_indices(self) -> List[int]: ...


def _to_chw(x: torch.Tensor, state_shape: Tuple[int, int, int]) -> torch.Tensor:
    """(B, H, W, C) -> (B, C, H, W) if needed."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == state_shape[2]:
        x = x.permute(0, 3, 1, 2)
    return x


class ActorCritic(nn.Module):
    """
    Shared backbone + actor (policy) and critic (value). Same conv/fc structure as DQN.
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int] = DEFAULT_STATE_HWC,
        max_actions: int = DEFAULT_MAX_ACTIONS,
    ):
        super(ActorCritic, self).__init__()
        h, w, c = state_shape
        self._state_shape = state_shape
        self._max_actions = max_actions

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        conv_out_size = 128 * h * w
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, max_actions)
        self.critic = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, H, W, C) or (B, C, H, W). Returns (logits, value)."""
        x = _to_chw(x, self._state_shape)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_chw(x, self._state_shape)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return self.critic(x).squeeze(-1)


class RolloutBuffer:
    """Stores (state, action, reward, next_state, done, log_prob) for PPO."""

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        log_prob: float,
    ) -> None:
        self.states.append(state.detach().cpu() if state.is_cuda else state.detach())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.detach().cpu() if next_state.is_cuda else next_state.detach())
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()

    def __len__(self) -> int:
        return len(self.states)


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and returns. next_values[t] = V(s_{t+1}) for bootstrapping."""
    advantages = []
    n = len(rewards)
    for t in reversed(range(n)):
        next_val = next_values[t] if t < len(next_values) else 0.0
        delta = rewards[t] + gamma * next_val * (1.0 - float(dones[t])) - values[t]
        if t == n - 1:
            last_gae = delta
        else:
            last_gae = delta + gamma * gae_lambda * advantages[-1] * (1.0 - float(dones[t]))
        advantages.append(last_gae)
    advantages = list(reversed(advantages))
    returns_t = [a + v for a, v in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns_t, dtype=torch.float32)


class PPOAgent:
    """
    PPO agent: act(controller) returns action and stores log_prob/value for push_transition.
    push_transition(state, action, reward, next_state, done, log_prob).
    train_step() runs when buffer has enough steps: GAE + PPO epochs.
    """

    def __init__(
        self,
        device: torch.device,
        state_shape: Tuple[int, int, int] = DEFAULT_STATE_HWC,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        rollout_size: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 4,
    ):
        self.device = device
        self.state_shape = state_shape
        self.max_actions = max_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_size = rollout_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.net = ActorCritic(state_shape=state_shape, max_actions=max_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self._last_log_prob: Optional[float] = None
        self._last_value: Optional[float] = None

    def act(self, env: EnvProtocol) -> int:
        """Sample action from policy over valid indices; state and valid from env. Store log_prob and value for push_transition."""
        state = env.get_state_tensor(self.device)
        valid_indices = env.get_valid_action_indices()
        if not valid_indices:
            self._last_log_prob = 0.0
            self._last_value = 0.0
            return 0

        state_chw = _to_chw(state.to(self.device), self.state_shape)
        logits, value = self.net(state_chw)
        logits = logits.squeeze(0)
        value = value.squeeze(0).item()

        # Mask invalid actions
        mask = torch.full((self.max_actions,), float("-inf"), device=self.device)
        mask[valid_indices] = 0
        logits_masked = logits + mask

        dist = torch.distributions.Categorical(logits=logits_masked)
        action_idx = dist.sample().item()
        if action_idx not in valid_indices:
            action_idx = valid_indices[0]
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()

        self._last_log_prob = log_prob
        self._last_value = value
        return action_idx

    def get_last_log_prob_and_value(self) -> Tuple[float, float]:
        """After act(), return (log_prob, value) for the transition."""
        return self._last_log_prob or 0.0, self._last_value or 0.0

    def push_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        log_prob: float,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done, log_prob)

    def train_step(self) -> Optional[float]:
        """
        If buffer has at least rollout_size steps, compute GAE, run n_epochs PPO update, clear buffer.
        Returns mean loss over epochs or None if not enough data.
        """
        if len(self.buffer) < self.rollout_size:
            return None

        states = torch.stack(self.buffer.states).to(self.device)
        next_states = torch.stack(self.buffer.next_states).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            states_chw = _to_chw(states, self.state_shape)
            next_states_chw = _to_chw(next_states, self.state_shape)
            _, values = self.net(states_chw)
            next_values = self.net.get_value(next_states_chw)
            values = values.cpu().tolist()
            next_values = next_values.cpu().tolist()

        advantages, returns_t = compute_gae(
            rewards, values, next_values, dones, self.gamma, self.gae_lambda
        )
        advantages = advantages.to(self.device)
        returns_t = returns_t.to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_sum = 0.0
        n_batches = 0
        indices = torch.randperm(len(self.buffer.states), device=self.device)

        for _ in range(self.n_epochs):
            for start in range(0, len(self.buffer.states), self.batch_size):
                end = min(start + self.batch_size, len(self.buffer.states))
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns_t[mb_indices]

                mb_states_chw = _to_chw(mb_states, self.state_shape)
                logits, values_pred = self.net(mb_states_chw)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_loss_sum += loss.item()
                n_batches += 1

        self.buffer.clear()
        return total_loss_sum / n_batches if n_batches else None

    def decay_epsilon(self) -> None:
        """No-op for PPO (no epsilon); kept for interface compatibility with train loop."""
        pass

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "state_shape": self.state_shape,
                "max_actions": self.max_actions,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device,
        lr: float = 3e-4,
        **kwargs: Any,
    ) -> "PPOAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent = cls(
            device=device,
            state_shape=tuple(ckpt["state_shape"]),
            max_actions=int(ckpt["max_actions"]),
            lr=lr,
            **kwargs,
        )
        agent.net.load_state_dict(ckpt["net"])
        return agent
