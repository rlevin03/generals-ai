import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from generals_rl_env_gpu import GeneralsEnv
from generals import GRID_WIDTH, GRID_HEIGHT


DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
PASS_ACTION = None  


def action_to_coords(action: int):
    idx = action // 4
    return idx % GRID_WIDTH, idx // GRID_WIDTH


def get_valid_actions(state_flat: np.ndarray, player_id: int):

    valid = []
    state = state_flat.reshape(GRID_HEIGHT, GRID_WIDTH, 4)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            owner, army, _, _ = state[y, x]
            if int(owner) == player_id and army >= 2:
                for d, (dx, dy) in enumerate(DIRS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        valid.append((y * GRID_WIDTH + x) * 4 + d)
    return valid


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Fully connected layers - matching saved model exactly
        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 25, 512),  # Updated to match environment dimensions
            nn.ReLU(),
            nn.Linear(512, 256),            # fc.1
            nn.ReLU(),
            nn.Linear(256, 256),            # fc.2
            nn.ReLU(),
            nn.Linear(256, 256),            # fc.3
            nn.ReLU(),
            nn.Linear(256, 256)             # fc.4
        )
        
        # Dueling streams
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, action_dim)
        
    def forward(self, x: torch.Tensor):
        # Reshape input if needed (from flattened to 2D)
        if len(x.shape) == 2:  # If input is flattened
            x = x.view(-1, 6, 20, 25)  # Updated to match environment dimensions
        
        # Convolutional layers
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.fc(x)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(
    q_net: QNetwork,
    state: torch.Tensor,
    epsilon: float,
    action_dim: int,
    player_id: int,
):
    valid = get_valid_actions(state.numpy(), player_id)
    if not valid:
        return PASS_ACTION

    if random.random() < epsilon:
        return random.choice(valid)
    else:
        with torch.no_grad():
            q_vals = q_net(state.unsqueeze(0)).squeeze(0) 
            mask = torch.full_like(q_vals, float('-inf'))
            mask[valid] = 0.0
            q_masked = q_vals + mask
            return int(q_masked.argmax().item())


def train_dqn(
    env: GeneralsEnv,
    num_episodes=1000,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    buffer_capacity=100000,
    min_buffer_size=1000,
    target_update_freq=1000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=1e-5,
):
    raw0 = env.reset()
    state_dim = len(raw0)
    action_dim = env.action_space.n

    online = QNetwork(state_dim, action_dim)
    target = QNetwork(state_dim, action_dim)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    total_steps = 0

    for ep in range(num_episodes):
        raw = env.reset()
        state = torch.tensor(raw, dtype=torch.float32)
        last_base_reward = env.compute_reward() 

        ep_reward = 0.0
        step = 0

        while True:
            action = select_action(online, state, epsilon, action_dim, env.player_id)

            if action is PASS_ACTION:
                env.game.update()
                next_raw = env.extract_state()
                done = env.game.game_over
                base_reward = env.compute_reward()
                reward = base_reward - last_base_reward
                last_base_reward = base_reward
            else:
                fx, fy = action_to_coords(action)
                dx, dy = DIRS[action % 4]
                tx, ty = fx + dx, fy + dy
                print(f"Step {step}: 从 ({fx},{fy}) → ({tx},{ty})")
                next_raw, _, done, _ = env.step(action)
                base_reward = env.compute_reward()
                reward = base_reward - last_base_reward
                last_base_reward = base_reward
                ep_reward += reward
                step += 1

            next_state = torch.tensor(next_raw, dtype=torch.float32)

            if action is not PASS_ACTION:
                buffer.push(state.numpy(), action, reward, next_state.numpy(), done)

            state, raw = next_state, next_raw

            if len(buffer) >= min_buffer_size:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)
                q_vals = online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(ns_b).max(1)[0]
                    target_q = r_b + gamma * next_q * (1 - d_b)
                loss = nn.MSELoss()(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if total_steps % target_update_freq == 0:
                    target.load_state_dict(online.state_dict())

            epsilon = max(epsilon_end, epsilon - epsilon_decay)
            total_steps += 1

            if done:
                break


def main():
    env = GeneralsEnv(player_id=0)
    train_dqn(env)


if __name__ == "__main__":
    main()

