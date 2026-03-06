from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ActOutput:
    actions: torch.Tensor
    is_pad_hat: Optional[torch.Tensor]
    mu: Optional[torch.Tensor]
    logvar: Optional[torch.Tensor]
    prior_mu: Optional[torch.Tensor]
    prior_logvar: Optional[torch.Tensor]
    tokens: Optional[torch.Tensor]
    global_feat: Optional[torch.Tensor]


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class RepresentationTransformerActor(nn.Module):

    def __init__(
        self,
        action_dim: int,
        robot_state_dim: int,
        representation_encoder: nn.Module,
        chunk_size: int = 50,
        model_emb_dim: int = 512,
        latent_dim: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        return_dict: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.representation_encoder = representation_encoder

        self.robot_state_dim = robot_state_dim
        self.action_dim = action_dim
        self.K = chunk_size
        self.model_emb_dim = model_emb_dim
        self.latent_dim = latent_dim
        self.return_dict = return_dict

        # ------------------------------------------------
        # Normalization buffers
        # ------------------------------------------------

        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

        self.register_buffer("qpos_mean", torch.zeros(robot_state_dim))
        self.register_buffer("qpos_std", torch.ones(robot_state_dim))

        self.register_buffer("is_norm_initialized", torch.tensor(False))

        # ------------------------------------------------
        # Observation transformer
        # ------------------------------------------------

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ------------------------------------------------
        # Decoder
        # ------------------------------------------------

        self.query_embed = nn.Embedding(self.K, model_emb_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=model_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.action_head = nn.Linear(model_emb_dim, action_dim)
        self.is_pad_head = nn.Linear(model_emb_dim, 1)

        # ------------------------------------------------
        # CVAE components
        # ------------------------------------------------

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * self.K, model_emb_dim),
            nn.ReLU(),
            nn.Linear(model_emb_dim, model_emb_dim),
        )

        self.posterior = nn.Linear(model_emb_dim * 2, latent_dim * 2)

        self.prior = nn.Linear(model_emb_dim, latent_dim * 2)

        self.latent_proj = nn.Linear(latent_dim, model_emb_dim)

    # ------------------------------------------------
    # Normalization utilities
    # ------------------------------------------------

    def set_norm_stats(self, action_mean, action_std, qpos_mean, qpos_std):

        action_std = torch.clamp(action_std, min=1e-5)
        qpos_std = torch.clamp(qpos_std, min=1e-5)

        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std)

        self.qpos_mean.copy_(qpos_mean)
        self.qpos_std.copy_(qpos_std)

        self.is_norm_initialized.fill_(True)

        print("Normalization stats loaded into model.")

    def normalize_actions(self, actions):

        if not self.is_norm_initialized:
            return actions

        return (actions - self.action_mean) / self.action_std

    def normalize_qpos(self, qpos):

        if not self.is_norm_initialized:
            return qpos

        return (qpos - self.qpos_mean) / self.qpos_std

    def unnormalize_actions(self, actions):

        if not self.is_norm_initialized:
            return actions

        return actions * self.action_std + self.action_mean

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(
        self,
        endoscope_image,
        wrist_l,
        wrist_r,
        robot_states,
        texts=None,
        action_chunk=None,
        action_is_pad=None,
        depth=None,
    ):

        # normalize robot states
        robot_states = self.normalize_qpos(robot_states)

        global_token, tokens = self.representation_encoder(
            endoscope_image,
            wrist_l,
            wrist_r,
            robot_states,
            texts,
        )

        bs = tokens.shape[0]

        memory = self.transformer(tokens)

        # ------------------------------------------------
        # PRIOR
        # ------------------------------------------------

        prior_stats = self.prior(global_token)
        prior_mu, prior_logvar = torch.chunk(prior_stats, 2, dim=-1)

        mu, logvar = None, None

        # ------------------------------------------------
        # POSTERIOR
        # ------------------------------------------------

        if action_chunk is not None:

            actions_norm = self.normalize_actions(action_chunk)

            flat_actions = actions_norm.reshape(bs, -1)

            act_feat = self.action_encoder(flat_actions)

            posterior_input = torch.cat([global_token, act_feat], dim=-1)

            stats = self.posterior(posterior_input)

            mu, logvar = torch.chunk(stats, 2, dim=-1)

            z = reparametrize(mu, logvar)

        else:

            z = prior_mu

        # ------------------------------------------------
        # inject latent
        # ------------------------------------------------

        z_embed = self.latent_proj(z)

        memory = memory + z_embed[:, None, :]

        # ------------------------------------------------
        # decode
        # ------------------------------------------------

        q = self.query_embed.weight[None].expand(bs, -1, -1)

        hs = self.decoder(tgt=q, memory=memory)

        actions_hat_norm = self.action_head(hs)

        is_pad_hat = self.is_pad_head(hs).squeeze(-1)

        # ------------------------------------------------
        # output handling
        # ------------------------------------------------

        if action_chunk is not None:

            # training → stay normalized
            actions_out = actions_hat_norm

        else:

            # inference → convert to real units
            actions_out = self.unnormalize_actions(actions_hat_norm)

        if self.return_dict:
            return ActOutput(
                actions=actions_out,
                is_pad_hat=is_pad_hat,
                mu=mu,
                logvar=logvar,
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                tokens=memory,
                global_feat=global_token,
            )

        return actions_out