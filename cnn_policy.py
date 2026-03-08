import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SnakeCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        shape = observation_space.shape
        h, w, c = shape
        assert len(shape) == 3, (
            f"\n\n[SnakeCNN] Expected a 3-D observation space but got shape {shape}.\n"
            f"Make sure you replaced env.py with the CNN version.\n"
            f"The observation_space should be Box(0,255,(15,20,3)) -- got {shape} instead.\n"
        )

        if shape[0] <= shape[1] and shape[0] <= shape[2]:
            c, h, w = shape
        else:
            h, w, c = shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    