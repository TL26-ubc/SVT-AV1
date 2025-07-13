import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable
from gymnasium import spaces
from .extractor import SBGlobalExtractor
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

class SBGlobalActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy whose *policy* head is the FiLM-conditioned
    per-super-block (1x1 conv) architecture. The value head keeps
    the default SB3 MLP built from the pooled feature vector.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule: Callable[[float], float],
        sb_channels: int = 64,
        glob_hidden: int = 64,
        glob_layers: list[int] | None = None,
        value_layers: list[int] | None = None,
        **kwargs,
    ):
        if glob_layers is None:
            glob_layers = [glob_hidden, glob_hidden]
        if value_layers is None:
            value_layers = [128, 1]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=nn.ReLU,
            features_extractor_class=SBGlobalExtractor,
            features_extractor_kwargs=dict(
                sb_channels=sb_channels,
                glob_hidden=glob_hidden,
                glob_layers=glob_layers,
            ),
            share_features_extractor=True,
            **kwargs,
        )

        # Build FiLM + 1x1 conv policy head
        C_sb   = sb_channels
        C_glob = glob_layers[-1]  # Output dimension of global MLP
        K      = int(action_space.nvec[0]) # deltaqp bins per sb

        self.film = nn.Linear(C_glob, 2 * C_sb)
        self.head = nn.Conv2d(C_sb, K, kernel_size=1)

        # For value network, reuse the pooled vector
        self.value_net = create_mlp(C_sb + C_glob, value_layers)

        self.action_net = nn.Identity()

    def forward( # type: ignore
        self,
        obs: ObservationDict[th.Tensor],
        deterministic: bool = False
    ):
        # extract features
        sb_feat, glob_emb = self.features_extractor(obs)

        gamma, beta = th.chunk(self.film(glob_emb), 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        sb_mod = (1 + gamma) * sb_feat + beta

        logits = self.head(sb_mod) # (B, K, H, W)
        B, K, H, W = logits.shape
        action_logits = logits.permute(0, 2, 3, 1).reshape(B, -1) # (B, Sum over nvec)

        # critic
        sb_gap     = th.mean(sb_feat, dim=(2, 3))
        latent_vf  = th.cat([sb_gap, glob_emb], dim=1)
        values     = self.value_net(latent_vf)

        # distribution, sampling
        dist       = self.action_dist.proba_distribution(action_logits=action_logits)
        actions    = dist.get_actions(deterministic=deterministic)
        log_prob   = dist.log_prob(actions)

        return actions, values, log_prob
    
    def _latent_from_obs(self, obs: ObservationDict[th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        sb_feat, glob_emb = self.features_extractor(obs)

        # FiLM actor path
        gamma, beta = th.chunk(self.film(glob_emb), 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        sb_mod   = (1 + gamma) * sb_feat + beta
        logits   = self.head(sb_mod) # (B,K,H,W)
        B, K, H, W = logits.shape
        latent_pi = logits.permute(0, 2, 3, 1).reshape(B, -1) # (B, Σ nvec)

        # critic path
        sb_gap   = th.mean(sb_feat, dim=(2, 3)) # (B,C_sb)
        latent_vf = th.cat([sb_gap, glob_emb], dim=1) # (B,C_sb+C_glob)

        return latent_pi, latent_vf
    
    def predict_values(self, obs: ObservationDict[th.Tensor]) -> th.Tensor: # type: ignore[override]
        with th.no_grad():
            _, latent_vf = self._latent_from_obs(obs)
            return self.value_net(latent_vf)
    
    def evaluate_actions( # type: ignore[override]
        self,
        obs: ObservationDict[th.Tensor],
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._latent_from_obs(obs)
        dist       = self.action_dist.proba_distribution(action_logits=latent_pi)
        log_prob   = dist.log_prob(actions)
        entropy    = dist.entropy()
        assert entropy is not None
        values     = self.value_net(latent_vf)
        return values, log_prob, entropy
