from typing import cast
import torch as th
import torch.nn as nn
from gymnasium.spaces import Dict as DictSpace
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .norm import ObservationDict


def create_mlp(input_dim: int, layer_sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """
    Create an MLP with specified layer sizes.
    
    Args:
        input_dim: Input dimension
        layer_sizes: List of hidden layer sizes (including output layer)
        activation: Activation function class (applied to all layers except the last)
    
    Returns:
        nn.Sequential MLP
    """
    if not layer_sizes:
        return nn.Sequential(nn.Identity())
    
    layers = []
    prev_dim = input_dim
    
    for i, layer_size in enumerate(layer_sizes):
        layers.append(nn.Linear(prev_dim, layer_size))
        # Add activation to all layers except the last one
        if i < len(layer_sizes) - 1:
            layers.append(activation())
        prev_dim = layer_size
    
    return nn.Sequential(*layers)

class SBGlobalExtractor(BaseFeaturesExtractor):
    """
    Returns a *pair*:
        sb_feat   – (B, C_sb, H, W)   : per-pixel feature map
        glob_emb  – (B, C_glob)       : frame-level embedding
    """

    def __init__(
        self,
        observation_space: DictSpace,
        sb_channels: int = 64,
        glob_hidden: int = 64,
        glob_layers: list[int] | None = None,
    ):
        super().__init__(observation_space, features_dim=1)  # dummy; not used

        if glob_layers is None:
            glob_layers = [glob_hidden, glob_hidden]

        # Forcibly infer typing for the space
        typed_obs_space: ObservationDict = cast(ObservationDict, observation_space)
        
        # Shapes
        H, W, C_in = typed_obs_space["superblock"].shape
        F_in       = typed_obs_space["frame"].shape[0]

        # sb branch
        self.conv_sb = nn.Sequential(
            nn.Conv2d(C_in, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, sb_channels, 3, padding=1), nn.ReLU(),
        )

        # frame branch - flexible MLP
        self.mlp_glob = create_mlp(F_in, glob_layers)

        self.sb_channels_out = sb_channels
        self.glob_dim_out    = glob_layers[-1] if glob_layers else F_in

    def forward(self, obs: ObservationDict[th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        # obs["superblock"]  (B, H, W, C_in)  →  (B, C_in, H, W)
        sb = obs["superblock"].permute(0, 3, 1, 2)
        sb_feat: th.Tensor = self.conv_sb(sb) # (B, C_sb, H, W)

        frame_feat: th.Tensor = self.mlp_glob(obs["frame"]) # (B, C_glob)
        return sb_feat, frame_feat
