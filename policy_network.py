import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape=(20, 25, 6), n_actions: int = 1600, hidden_size: int = 128):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        # Convolutional torso -------------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        conv_output_size = 64 * obs_shape[0] * obs_shape[1]  # 64 channels × H × W

        # Fully‑connected head ----------------------------------------------
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_logits = nn.Linear(hidden_size, n_actions)
        
        # Value head
        self.fc_value = nn.Linear(hidden_size, 1)

        # Normalisation layers (helpful for REINFORCE high‑variance signals)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # ---- weight init ----------------------------------------------------
        self.apply(self._init_weights)

    # ---------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters
        ----------
        obs : (*batch*, 20, 25, 6) float32 tensor in [0, 1] range (or raw).
        Returns
        -------
        logits : (*batch*, n_actions) – **unnormalised** action logits suitable
                  for ``torch.distributions.Categorical``.
        value : (*batch*, 1) - value estimate for the current state
        """
        batch_size = obs.shape[0]
        # Reshape to NCHW for the conv stack
        x = obs.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        # 3‑layer MLP with a residual connection
        x = F.relu(self.layer_norm1(self.fc1(x)))
        residual = x
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = x + residual
        x = F.relu(self.fc3(x))
        logits = self.fc_logits(x)
        value = self.fc_value(x)
        return logits, value

    # ------------------------------------------------------------------
    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Categorical distribution over all possible discrete moves."""
        logits, _ = self.forward(obs)
        return torch.distributions.Categorical(logits=logits) 