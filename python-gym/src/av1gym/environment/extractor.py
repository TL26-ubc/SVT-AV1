from typing import cast
import torch as th
import torch.nn as nn
from gymnasium.spaces import Dict as DictSpace
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .norm import ObservationDict

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
        glob_hidden: int = 64
    ):
        super().__init__(observation_space, features_dim=1)  # dummy; not used

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

        # frame branch
        self.mlp_glob = nn.Sequential(
            nn.Linear(F_in, glob_hidden), nn.ReLU(),
            nn.Linear(glob_hidden, glob_hidden), nn.ReLU(),
        )

        self.sb_channels_out = sb_channels
        self.glob_dim_out    = glob_hidden

    def forward(self, obs: ObservationDict[th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        # obs["superblock"]  (B, H, W, C_in)  →  (B, C_in, H, W)
        sb = obs["superblock"].permute(0, 3, 1, 2)
        sb_feat: th.Tensor = self.conv_sb(sb) # (B, C_sb, H, W)

        frame_feat: th.Tensor = self.mlp_glob(obs["frame"]) # (B, C_glob)
        return sb_feat, frame_feat
