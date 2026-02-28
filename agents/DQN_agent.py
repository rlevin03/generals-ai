"""
Deep Q-Network (DQN) Agent — game-agnostic, controller-compatible.

Consumes state and valid action indices from a controller; outputs an action index.
No dependency on the game or environment. Compatible with DQNController.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Optional, Any, Protocol

# Default state shape (H, W, C) and max actions for the network (controller uses indices 0..n-1)
DEFAULT_STATE_HWC = (20, 25, 6)
DEFAULT_MAX_ACTIONS = 25 * 20 * 4  # max valid actions we ever index into


class ControllerProtocol(Protocol):
    """Protocol for a DQN controller: agent only depends on this interface."""

    def reset(self) -> Tuple[torch.Tensor, List[int], Any]:
        ...

    def get_state_for_agent(self) -> torch.Tensor:
        ...

    def get_valid_actions_for_agent(self) -> List[int]:
        ...

    def step(self, agent_action_idx: int) -> Tuple[torch.Tensor, float, bool, Any]:
        ...

    def get_last_transition(
        self,
    ) -> Optional[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        ...


class QNetwork(nn.Module):
    """
    Dueling DQN that maps state (B, C, H, W) to Q-values for a fixed maximum number of actions.
    Game-agnostic: only assumes state shape and max action dimension.
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int] = DEFAULT_STATE_HWC,
        max_actions: int = DEFAULT_MAX_ACTIONS,
    ):
        super(QNetwork, self).__init__()
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
        )
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, max_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W). Returns (B, max_actions) Q-values.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != self._state_shape[2]:  # assume (B, H, W, C) if channel last
            if x.shape[-1] == self._state_shape[2]:
                x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Stores (state, action_idx, reward, next_state, done) for training. Accepts tensors."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        state_cpu = state.detach().cpu() if state.is_cuda else state
        next_cpu = next_state.detach().cpu() if next_state.is_cuda else next_state
        self.buffer.append((state_cpu, action, reward, next_cpu, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64, device=states[0].device),
            torch.tensor(rewards, dtype=torch.float32, device=states[0].device),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32, device=states[0].device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(
    q_net: QNetwork,
    state: torch.Tensor,
    valid_indices: List[int],
    epsilon: float,
    device: torch.device,
) -> int:
    """
    Epsilon-greedy action selection over controller-provided valid indices.
    state: (1, H, W, C) or (1, C, H, W). Returns an index in valid_indices (or 0 if empty).
    """
    if not valid_indices:
        return 0
    if state.dim() == 3:
        state = state.unsqueeze(0)
    if state.shape[-1] == 6:
        state = state.permute(0, 3, 1, 2)
    state = state.to(device)
    q_net.eval()
    with torch.no_grad():
        q_values = q_net(state).squeeze(0)
    q_net.train()
    if random.random() < epsilon:
        return random.choice(valid_indices)
    valid_t = torch.tensor(valid_indices, dtype=torch.long, device=q_values.device)
    q_valid = q_values[valid_t]
    best_local = q_valid.argmax().item()
    return valid_indices[best_local]


def train_dqn(
    controller: ControllerProtocol,
    device: torch.device,
    num_episodes: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,
    buffer_capacity: int = 100_000,
    min_buffer_size: int = 1000,
    target_update_freq: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 1e-5,
    state_shape: Tuple[int, int, int] = DEFAULT_STATE_HWC,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> None:
    """
    Train DQN using a controller. Agent is game-agnostic; controller provides state and valid actions.
    """
    online_net = QNetwork(state_shape=state_shape, max_actions=max_actions).to(device)
    target_net = QNetwork(state_shape=state_shape, max_actions=max_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    total_steps = 0

    print(f"Starting DQN training for {num_episodes} episodes (controller-based)...")
    print(f"State shape: {state_shape}, Max actions: {max_actions}")

    for episode in range(num_episodes):
        state, valid_indices, info = controller.reset()
        state = state.to(device)
        episode_reward = 0.0
        step_count = 0

        while True:
            action_idx = select_action(
                online_net, state, valid_indices, epsilon, device
            )
            next_state, reward, done, info = controller.step(action_idx)
            next_state = next_state.to(device)
            episode_reward += reward
            step_count += 1

            trans = controller.get_last_transition()
            if trans is not None:
                s, a, r, s_next, d = trans
                replay.push(s, a, r, s_next, d)

            state = next_state
            valid_indices = controller.get_valid_actions_for_agent()

            if len(replay) >= min_buffer_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay.sample(
                    batch_size
                )
                states_b = states_b.to(device)
                next_states_b = next_states_b.to(device)
                if states_b.shape[-1] == 6:
                    states_b = states_b.permute(0, 3, 1, 2)
                    next_states_b = next_states_b.permute(0, 3, 1, 2)
                current_q = online_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + gamma * next_q * (1 - dones_b)
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(online_net.state_dict())

            epsilon = max(epsilon_end, epsilon - epsilon_decay)
            total_steps += 1

            if done:
                break

        if episode % 100 == 0:
            print(
                f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, Steps: {step_count}"
            )


class DQNAgent:
    """
    Single DQN agent for self-play: can act, store transitions, and perform one training step.
    Used when 4 DQNs play against each other with training every N moves.
    """

    def __init__(
        self,
        device: torch.device,
        state_shape: Tuple[int, int, int] = DEFAULT_STATE_HWC,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        lr: float = 1e-4,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        min_buffer_size: int = 1000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 1e-5,
    ):
        self.device = device
        self.state_shape = state_shape
        self.max_actions = max_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.online_net = QNetwork(state_shape=state_shape, max_actions=max_actions).to(device)
        self.target_net = QNetwork(state_shape=state_shape, max_actions=max_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity)
        self._train_steps = 0

    def act(self, controller: ControllerProtocol) -> int:
        """Pick a valid action; never sit out when valid actions exist."""
        state = controller.get_state_for_agent()
        valid_indices = controller.get_valid_actions_for_agent()
        return select_action(
            self.online_net, state, valid_indices, self.epsilon, self.device
        )

    def push_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Append one transition to this agent's replay buffer."""
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Sample a batch from replay and perform one gradient step.
        Returns loss if training was performed, else None.
        """
        if len(self.replay) < self.min_buffer_size:
            return None
        states_b, actions_b, rewards_b, next_states_b, dones_b = self.replay.sample(
            self.batch_size
        )
        states_b = states_b.to(self.device)
        next_states_b = next_states_b.to(self.device)
        actions_b = actions_b.to(self.device)
        rewards_b = rewards_b.to(self.device)
        dones_b = dones_b.to(self.device)
        if states_b.shape[-1] == 6:
            states_b = states_b.permute(0, 3, 1, 2)
            next_states_b = next_states_b.permute(0, 3, 1, 2)
        current_q = self.online_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_b).max(1)[0]
            target_q = rewards_b + self.gamma * next_q * (1 - dones_b)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()

    def decay_epsilon(self) -> None:
        """Decay exploration after each game (optional)."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)


def main() -> None:
    """Run DQN training with the DQN controller and environment."""
    from environment import GeneralsEnv
    from controller import DQNController

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GeneralsEnv(player_id=0, num_opponents=1)
    controller = DQNController(env, device)
    train_dqn(controller, device, num_episodes=1000)
    print("Training completed!")


if __name__ == "__main__":
    main()
