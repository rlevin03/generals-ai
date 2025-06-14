############################################################
# ppo_agent_masked.py  –  Dict-obs, MultiInputPolicy, working mask
############################################################
from sb3_contrib import MaskablePPO
try:                   # ActionMasker path variations
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    from sb3_contrib.common.maskable.wrappers import ActionMasker

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
import numpy as np
import os, time

from generals_rl_env_ppo import GeneralsEnv, GRID_WIDTH, GRID_HEIGHT

# ───────────────────────── Feature extractor ─────────────────────────
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class MultiInputCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=128):
        super().__init__(obs_space, features_dim)
        c = obs_space["state"].shape[-1]
        h, w = obs_space["state"].shape[:2]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        cnn_out = 64 * (h // 8) * (w // 8)
        flat_in  = h * w + h * w * 4
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + flat_in, 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        s = obs["state"].permute(0, 3, 1, 2)
        f = self.cnn(s)
        flat = th.cat([obs["owned_cells"].flatten(1),
                       obs["valid_moves"].flatten(1)], dim=1)
        return self.fc(th.cat([f, flat], dim=1))

# ───────────────────────── Metrics callback ─────────────────────────
class MetricsCallback(BaseCallback):
    def __init__(self, env, freq=10_000):
        super().__init__()
        self.env, self.freq, self.start = env, freq, time.time()

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            speed = self.n_calls / (time.time() - self.start)
            print(f"Step {self.n_calls} | {speed:.1f} steps/s")
        return True

# ───────────────────────── Agent wrapper ─────────────────────────
class GeneralsPPO:
    def __init__(self, player_id=0, n_envs=4):
        self.player_id = player_id
        self.env = self._make_vec_envs(n_envs)
        os.makedirs("models", exist_ok=True)

        self.model = MaskablePPO(
            "MultiInputPolicy",
            self.env,
            policy_kwargs=dict(
                features_extractor_class=MultiInputCNN,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[128,128], vf=[128,128]),
            ),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            verbose=1,
            tensorboard_log="./logs/",
        )

    # ---------- wrappers ----------
    def _wrap(self, env):
        # ActionMasker FIRST so mask stays in obs dict
        return ActionMasker(env, lambda e: e.get_action_mask())

    def _make_vec_envs(self, n):
        def make():
            return self._wrap(GeneralsEnv(player_id=self.player_id))
        return SubprocVecEnv([make for _ in range(n)])

    # ---------- training ----------
    def train(self, total_steps=50_000):
        self.model.learn(total_steps, callback=MetricsCallback(self.env))
        self.model.save("models/ppo_final")

# quick driver -------------------------------------------------------
if __name__ == "__main__":
    agent = GeneralsPPO()
    agent.train(10_000)
