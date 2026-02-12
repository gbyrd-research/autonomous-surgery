from dataclasses import dataclass
from typing import Optional, Any, Tuple, Dict

import torch
import torch.nn as nn

from lift3d.models.representation import load_dino

@dataclass
class ActOutput:
    actions: torch.Tensor                   # [B, K, A] (Predicted Actions in REAL SCALE)
    is_pad_hat: Optional[torch.Tensor ]      # [B, K] logits (optional)
    mu: Optional[torch.Tensor ]              # posterior mu   [B, Z]
    logvar: Optional[torch.Tensor ]          # posterior logvar [B, Z]
    prior_mu: Optional[torch.Tensor ]        # prior mu       [B, Z]
    prior_logvar: Optional[torch.Tensor ]    # prior logvar   [B, Z]
    tokens: Optional[torch.Tensor ]          # [B, N, D] (memory actually fed to decoder)
    global_feat: Optional[torch.Tensor ]     # [B, D]

def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

class _LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for sequences [B,T,D]."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        idx = torch.arange(t, device=x.device)
        return x + self.pos(idx)[None, :, :]

class RepresentationACTActor(nn.Module):
    def __init__(
            self,
            action_dim: int,
            robot_state_dim: int,

            # --- compatibility fields ---
            representation_encoder: nn.Module,

            # --- ACT/DETR-VAE core ---
            chunk_size: int = 50,
            model_emb_dim: int = 512,
            nhead: int = 8,
            num_decoder_layers: int = 4,
            num_encoder_layers: int = 4,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,

            latent_dim: int = 64,
            max_action_seq_len: int = 512,

            # behavior
            return_dict: bool = True,
            sample_prior: bool = False,   # inference: sample from prior or use mean
            **kwargs,
    ):
        super().__init__()
        _ = kwargs  # swallow unexpected hydra keys

        self.representation_encoder = representation_encoder

        self.robot_state_dim = int(robot_state_dim)
        self.action_dim = int(action_dim)

        self.K = int(chunk_size)
        self.model_emb_dim = int(model_emb_dim)
        self.latent_dim = int(latent_dim)
        self.return_dict = bool(return_dict)
        self.sample_prior = bool(sample_prior)

        # robot state -> d_model
        self.robot_to_d = nn.Linear(self.robot_state_dim, self.model_emb_dim)

        # ----- posterior q(z|a,obs): TransformerEncoder over [CLS, OBS, a1..aK] -----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.action_to_d = nn.Linear(self.action_dim, self.model_emb_dim)
        # +2 for CLS + OBS
        self.act_pos = _LearnedPositionalEncoding(max_action_seq_len + 2, self.model_emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.posterior_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.posterior_proj = nn.Linear(self.model_emb_dim, 2 * self.latent_dim)  # -> mu/logvar

        # ----- conditional prior p(z|obs): MLP(global_obs) -----
        self.prior_net = nn.Sequential(
            nn.Linear(self.model_emb_dim, self.model_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_emb_dim, 2 * self.latent_dim),
        )

        # latent z -> model space d_model
        self.latent_out = nn.Linear(self.latent_dim, self.model_emb_dim)

        # ----- DETR-style decoder head -----
        self.query_embed = nn.Embedding(self.K, self.model_emb_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.model_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.action_head = nn.Linear(self.model_emb_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.model_emb_dim, 1)
        nn.init.constant_(self.is_pad_head.bias, 0.0)

        # ---- token hook state ----
        self._pc_tokens_hooked = False
        self._pc_tokens_cache: Dict[str, torch.Tensor] = {}

        # Auto-Normalization Buffers
        # These will be saved in the .pth file but not trained via gradient descent.
        self.register_buffer('action_mean', torch.zeros(self.action_dim))
        self.register_buffer('action_std', torch.ones(self.action_dim))
        self.register_buffer('depth_geom_mean', torch.zeros(self.representation_encoder._get_depth_emb_dim()))
        self.register_buffer('depth_geom_std', torch.zeros(self.representation_encoder._get_depth_emb_dim()))
        self.register_buffer('qpos_mean', torch.zeros(self.robot_state_dim))
        self.register_buffer('qpos_std', torch.ones(self.robot_state_dim))
        self.register_buffer('is_norm_initialized', torch.tensor(0, dtype=torch.bool))

    # normalization methods
    def set_norm_stats(self, action_mean, action_std, depth_geom_mean, depth_geom_std, qpos_mean, qpos_std):
        """Called by training script to inject dataset statistics."""
        # Safety clip to avoid div by zero
        action_std = torch.clip(action_std, min=1e-5)
        qpos_std = torch.clip(qpos_std, min=1e-5)

        self.action_mean.data = action_mean
        self.action_std.data = action_std
        self.depth_geom_mean.data = depth_geom_mean
        self.depth_geom_std.data = depth_geom_std
        self.qpos_mean.data = qpos_mean
        self.qpos_std.data = qpos_std
        self.is_norm_initialized.data = torch.tensor(1, dtype=torch.bool, device=self.is_norm_initialized.device)

        self.representation_encoder._add_depth_geom_stats(depth_geom_mean, depth_geom_std)
        print("ACTPolicy: Normalization stats updated and locked into model buffers.")

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.is_norm_initialized:
            return actions
        return (actions - self.action_mean) / self.action_std
    
    def normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        if not self.is_norm_initialized:
            return qpos
        return (qpos - self.qpos_mean) / self.qpos_std

    def unnormalize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        if not self.is_norm_initialized:
            return actions_norm
        return actions_norm * self.action_std + self.action_mean

    # ---------------------------------------------------------------------
    # posterior q(z|a,obs) and prior p(z|obs)
    # ---------------------------------------------------------------------
    def _posterior(
        self,
        conditioning: torch.Tensor,
        actions: torch.Tensor,
        is_pad: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        conditioning: [B,D]
        actions:  [B,K,A]
        is_pad:   [B,K] bool (True padded)
        """
        bs = actions.shape[0]

        a = self.action_to_d(actions)             # [B,K,D]
        cls = self.cls_token.expand(bs, -1, -1)   # [B,1,D]
        obs = conditioning[:, None, :]                # [B,1,D]
        x = torch.cat([cls, obs, a], dim=1)       # [B,2+K,D]

        if is_pad is None:
            pad_mask = torch.zeros((bs, 2 + self.K), dtype=torch.bool, device=actions.device)
        else:
            if is_pad.shape != (bs, self.K):
                raise ValueError(f"is_pad must be [B,K], got {tuple(is_pad.shape)}")
            head = torch.zeros((bs, 2), dtype=torch.bool, device=actions.device)
            pad_mask = torch.cat([head, is_pad.to(device=actions.device)], dim=1)

        x = self.act_pos(x)
        h = self.posterior_encoder(x, src_key_padding_mask=pad_mask)  # [B,2+K,D]
        h_cls = h[:, 0, :]                                            # [B,D]
        stats = self.posterior_proj(h_cls)                            # [B,2Z]
        mu = stats[:, : self.latent_dim]
        logvar = stats[:, self.latent_dim :]
        return mu, logvar

    def _prior(self, global_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.prior_net(global_d)  # [B,2Z]
        mu = stats[:, : self.latent_dim]
        logvar = stats[:, self.latent_dim :]
        return mu, logvar

    def forward(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        robot_states: torch.Tensor,
        texts: Any = None,
        actions: Optional[torch.Tensor] = None,   # [B,K,A] in training
        is_pad: Optional[torch.Tensor] = None,    # [B,K] bool
    ):
        # we must normalize the robot_states to avoid distributional shifts in our
        # training dimensions
        robot_states_norm = self.normalize_qpos(robot_states)

        global_token, tokens = self.representation_encoder(images, depth, robot_states_norm, texts)
        bs = global_token.shape[0]

        # prior always available
        prior_mu, prior_logvar = self._prior(global_token)

        mu, logvar = None, None
        
        # 3. Handle Actions (Training vs Inference)
        if actions is not None:
            # Training Mode
            if actions.dim() != 3 or actions.shape[1] != self.K:
                raise ValueError(f"actions must be [B,K,A] with K={self.K}, got {tuple(actions.shape)}")
            
            # Normalize GT Actions for VAE posterior
            # The VAE learns to reconstruct Normalized actions (N(0,1))
            actions_norm = self.normalize_actions(actions)
            
            mu, logvar = self._posterior(global_token, actions_norm, is_pad)
            z = reparametrize(mu, logvar)   # posterior sample
        else:
            # Inference Mode
            if self.sample_prior:
                z = reparametrize(prior_mu, prior_logvar)
            else:
                z = prior_mu

        # Decoder / Policy Head
        latent_d = self.latent_out(z)       # [B,D]
        memory = tokens + latent_d[:, None, :]  # [B,N,D]

        # DETR-style queries
        q = self.query_embed.weight[None, :, :].expand(bs, -1, -1)  # [B,K,D]
        tgt = q

        hs = self.decoder(tgt=tgt, memory=memory)  # [B,K,D]

        actions_hat_norm = self.action_head(hs)          # [B,K,A] (Normalized Space)
        is_pad_hat = self.is_pad_head(hs).squeeze(-1)    # [B,K]

              # 5. Un-normalize Output
        # Return predictions in Real Physical Scale so external Loss/Metric functions work as expected.
        # Note: Ideally loss should be computed in Normalized space, but un-normalizing here ensures
        # compatibility with existing code that expects raw units.
        if actions is not None:
            # 【Training mode】
            # Directly return the normalized predictions!
            # This way the loss is computed in normalized space, and the weights between
            # position and rotation are automatically balanced.
            if self.return_dict:
                return ActOutput(
                    actions=actions_hat_norm, # <--- Note here: the returned actions are normalized
                    is_pad_hat=is_pad_hat,
                    mu=mu,
                    logvar=logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    tokens=memory,
                    global_feat=global_token,
                )
            return actions_hat_norm
        else:
            actions_hat = self.unnormalize_actions(actions_hat_norm)

            if self.return_dict:
                return ActOutput(
                    actions=actions_hat,
                    is_pad_hat=is_pad_hat,
                    mu=mu,
                    logvar=logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    tokens=memory,
                    global_feat=global_token,
                )
            return actions_hat