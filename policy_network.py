"""
Policy Network for Reinforcement Learning in Generals Game.

This module implements a convolutional neural network policy for the Generals game
environment. The network uses a CNN backbone to process game state observations
and outputs action logits and state values for REINFORCE training.

Most of this code is AI generated.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Convolutional neural network policy for the Generals game.
    
    This network processes game state observations through a CNN backbone and
    outputs action logits and state values. It's designed for REINFORCE training
    with discrete action spaces.
    
    Attributes:
        obs_shape (tuple): Shape of input observations (height, width, channels)
        n_actions (int): Number of possible discrete actions
        conv (nn.Sequential): Convolutional layers for feature extraction
        fc1, fc2, fc3 (nn.Linear): Fully connected layers for policy head
        fc_logits (nn.Linear): Output layer for action logits
        fc_value (nn.Linear): Output layer for state value estimation
        layer_norm1, layer_norm2 (nn.LayerNorm): Layer normalization for training stability
    """
    
    def __init__(self, obs_shape=(20, 25, 6), n_actions: int = 1600, hidden_size: int = 128):
        """
        Initialize the PolicyNetwork.
        
        Args:
            obs_shape (tuple): Shape of input observations (height, width, channels).
                              Defaults to (20, 25, 6) for the Generals game.
            n_actions (int): Number of possible discrete actions. Defaults to 1600.
            hidden_size (int): Size of hidden layers in the fully connected network.
                              Defaults to 128.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        # Convolutional backbone for feature extraction
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

        # Calculate output size of convolutional layers
        conv_output_size = 64 * obs_shape[0] * obs_shape[1]  # 64 channels × H × W

        # Policy head (fully connected layers)
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_logits = nn.Linear(hidden_size, n_actions)
        
        # Value head for state value estimation
        self.fc_value = nn.Linear(hidden_size, 1)

        # Layer normalization for training stability (helps with REINFORCE's high variance)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Initialize network weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize network weights using Kaiming initialization.
        
        Args:
            module: Neural network module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Input observations tensor of shape (*batch*, 20, 25, 6) 
                 with values in [0, 1] range or raw values.
        
        Returns:
            tuple: (logits, value) where:
                - logits: Unnormalized action logits of shape (*batch*, n_actions)
                - value: State value estimate of shape (*batch*, 1)
        """
        batch_size = obs.shape[0]
        
        # Reshape to NCHW format for convolutional layers
        x = obs.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Extract features through convolutional backbone
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        
        # Process through fully connected layers with residual connection
        x = F.relu(self.layer_norm1(self.fc1(x)))
        residual = x
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = x + residual  # Residual connection for better gradient flow
        x = F.relu(self.fc3(x))
        
        # Output action logits and state value
        logits = self.fc_logits(x)
        value = self.fc_value(x)
        
        return logits, value

    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get categorical distribution over all possible discrete actions.
        
        Args:
            obs: Input observations tensor of shape (*batch*, 20, 25, 6)
        
        Returns:
            Categorical distribution over the action space
        """
        logits, _ = self.forward(obs)
        return torch.distributions.Categorical(logits=logits) 