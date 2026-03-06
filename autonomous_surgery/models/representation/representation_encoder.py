
from typing import Any, List, Optional

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel


class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # debug
        dropout = 0
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_q = nn.LayerNorm(emb_dim)
        self.ln_out = nn.LayerNorm(emb_dim)

    def forward(self, q, kv):
        """
        q:  [B, Nq, D]   (RGB patches)
        kv: [B, Nk, D]   (Depth patches)
        """
        q_norm = self.ln_q(q)
        attn_out, _ = self.attn(
            query=q_norm,
            key=kv,
            value=kv,
            need_weights=False,
        )
        return self.ln_out(q + attn_out)


class DinoV3Encoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            resize_size: int,
            num_register_tokens: int,
            **kwargs):
        super().__init__()
        valid_model_names = {
            "dinov3_convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m"
        }
        if model_name not in valid_model_names:
            raise NotImplementedError(f"Invalid model name ({model_name}). Must be one of {valid_model_names}.")

        self.resize_size = resize_size
        self.num_register_tokens = num_register_tokens

        pretrained_model_name = valid_model_names[model_name]
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.feature_extractor = AutoModel.from_pretrained(pretrained_model_name)

        # Freeze — we do not want to train the vision backbone.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    @property
    def device(self) -> torch.device:
        """Always returns the device the feature_extractor actually lives on."""
        return next(self.feature_extractor.parameters()).device

    def __call__(self, images: torch.Tensor, *args: Any, **kwds: Any) -> Any:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.feature_extractor(**inputs)
        outputs = outputs.last_hidden_state

        R = 4
        cls_token   = outputs[:, :1]        # (B, 1, D)
        reg_tokens  = outputs[:, 1:1+R]     # (B, R, D)
        patch_tokens = outputs[:, 1+R:]     # (B, N, D)
        return cls_token, reg_tokens, patch_tokens


class DepthEncoder(nn.Module):
    def __init__(
        self,
        patched_dim: int,
        intrinsics: Optional[List[List[float]]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.intrinsics = intrinsics
        self.patched_dim = patched_dim
        self.eps = eps

    def guess_camera_intrinsics(self, H: int, W: int, fov_deg: float = 60.0, device=None):
        if device is None:
            device = "cpu"
        f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        K = torch.tensor(
            [[f, 0.0, W / 2.0],
             [0.0, f, H / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        return K

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: [B, H, W] depth map in metres
        Returns:
            geom_feats: [B, N_patches, 8]
        """
        device = depth.device
        B, H, W = depth.shape

        if self.intrinsics is None:
            K = self.guess_camera_intrinsics(H, W, device=device).unsqueeze(0).expand(B, -1, -1)
        else:
            K = self.intrinsics
            if K.dim() == 2:
                K = K.unsqueeze(0).expand(B, -1, -1)
            elif K.dim() != 3:
                raise ValueError("Intrinsics must be [3,3] or [B,3,3]")

        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        xs = xs.unsqueeze(0).expand(B, -1, -1)
        ys = ys.unsqueeze(0).expand(B, -1, -1)

        z = depth
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        points = torch.stack([x, y, z], dim=-1)  # [B, H, W, 3]

        P = self.patched_dim
        ps_h = H // P
        ps_w = W // P
        points = points[:, : P * ps_h, : P * ps_w]
        points = (
            points.view(B, P, ps_h, P, ps_w, 3)
                  .permute(0, 1, 3, 2, 4, 5)
        )  # [B, P, P, ps_h, ps_w, 3]
        patches = points.reshape(B, P * P, ps_h * ps_w, 3)  # [B, N, Ppx, 3]

        mean_xyz = patches.mean(dim=2)                       # [B, N, 3]
        centered = patches - mean_xyz.unsqueeze(2)
        cov = torch.matmul(centered.transpose(-1, -2), centered) / (centered.shape[2] + self.eps)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normals = eigvecs[..., 0]
        normals = normals / (normals.norm(dim=-1, keepdim=True) + self.eps)

        z_vals   = patches[..., 2]
        depth_var = z_vals.var(dim=2, unbiased=False).unsqueeze(-1)
        planarity = (eigvals[..., 1] / (eigvals[..., 2] + self.eps)).unsqueeze(-1)

        return torch.cat([mean_xyz, normals, depth_var, planarity], dim=-1)  # [B, N, 8]


# ---------------------------------------------------------------------------
# Shared building-block helpers (used by both encoder variants)
# ---------------------------------------------------------------------------

def _make_img_resize_transform(resize_size: int):
    return v2.Compose([
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])


def _build_final_transformer_encoder(model_emb_dim: int, num_layers: int = 6, num_heads: int = 8):
    # debug
    dropout = 0
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=model_emb_dim,
        nhead=num_heads,
        dim_feedforward=4 * model_emb_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    final_ln = nn.LayerNorm(model_emb_dim)
    return transformer, final_ln


def _build_view_fusion(img_enc_emb_dim: int, num_layers: int = 4, num_heads: int = 8):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=img_enc_emb_dim,
        nhead=num_heads,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


# ---------------------------------------------------------------------------
# RepresentationEncoder  (with depth)
# ---------------------------------------------------------------------------

class RepresentationEncoder(nn.Module):
    """Multi-view RGB + depth + robot-state + text encoder."""

    def __init__(
            self,
            image_encoder: nn.Module,
            depth_encoder: nn.Module,
            robot_state_dim: int,
            model_emb_dim: int,
            patched_dim: int,
            resize_size: int,
            **kwargs
    ):
        super().__init__()
        self.model_emb_dim = model_emb_dim
        self.patched_dim   = patched_dim
        self.resize_size   = resize_size

        self.image_encoder = image_encoder
        self.depth_encoder = depth_encoder

        # Probe encoder output dimension with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, resize_size, resize_size)
            img_enc_emb_dim = image_encoder(dummy)[0].shape[-1]

        # View embeddings and multi-view fusion.
        self.view_embed = nn.Parameter(torch.randn(3, 1, img_enc_emb_dim))
        self.view_fusion = _build_view_fusion(img_enc_emb_dim)
        self.view_attn = nn.Sequential(
            nn.Linear(img_enc_emb_dim, img_enc_emb_dim),
            nn.GELU(),
            nn.Linear(img_enc_emb_dim, 1),
        )

        self.img_resize = _make_img_resize_transform(resize_size)

        # Projection layers.
        self.img_cls_emb_to_model_emb_dim   = nn.Linear(img_enc_emb_dim, model_emb_dim)
        self.img_patch_emb_to_model_emb_dim = nn.Linear(img_enc_emb_dim, model_emb_dim)
        self.robot_state_dim_to_model_emb_dim = nn.Linear(robot_state_dim, model_emb_dim)

        # Depth projection.
        depth_emb_dim = self._get_depth_emb_dim()
        self.depth_emb_to_model_emb_dim = nn.Linear(depth_emb_dim, model_emb_dim)

        # RGB ↔ depth cross-attention.
        self.rgb_depth_fusion = CrossAttentionBlock(emb_dim=model_emb_dim, num_heads=8)

        # Text encoder (frozen).
        self.tokenizer    = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.text_proj = nn.Linear(768, model_emb_dim)

        # Final transformer encoder.
        self.final_transformer, self.final_ln = _build_final_transformer_encoder(model_emb_dim)
        max_tokens = 1 + (resize_size ** 2) + 1 + 1  # cls + patches + robot_state + text
        self.final_pos_emb = nn.Parameter(torch.zeros(1, max_tokens, model_emb_dim))
        nn.init.trunc_normal_(self.final_pos_emb, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_depth_emb_dim(self) -> int:
        depth = torch.zeros(1, 1024, 1024)
        return self.depth_encoder(depth).shape[-1]

    def _add_depth_geom_stats(self, depth_geom_mean: torch.Tensor, depth_geom_std: torch.Tensor):
        self.depth_geom_mean = depth_geom_mean
        self.depth_geom_std  = depth_geom_std

    def _normalize_depth_geom(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.depth_geom_mean) / self.depth_geom_std

    def forward(
        self,
        endoscope_image: torch.Tensor,
        wrist_l: torch.Tensor,
        wrist_r: torch.Tensor,
        depth: torch.Tensor,
        robot_states: torch.Tensor,
        text: List[str],
    ):
        endo = self.img_resize(endoscope_image)
        wl   = self.img_resize(wrist_l)
        wr   = self.img_resize(wrist_r)

        cls_e,  _,  patch_e  = self.image_encoder(endo)
        cls_wl, _, patch_wl  = self.image_encoder(wl)
        cls_wr, _, patch_wr  = self.image_encoder(wr)

        patch_e  = patch_e  + self.view_embed[0]
        patch_wl = patch_wl + self.view_embed[1]
        patch_wr = patch_wr + self.view_embed[2]

        multi_view_tokens = torch.cat([patch_e, patch_wl, patch_wr], dim=1)
        fused_vision_tokens = self.view_fusion(multi_view_tokens)

        cls_stack   = torch.stack([cls_e, cls_wl, cls_wr], dim=1)  # (B, 3, D)
        attn_logits = self.view_attn(cls_stack)
        attn_weights = torch.softmax(attn_logits, dim=1)
        global_vision_cls = (cls_stack * attn_weights).sum(dim=1)   # (B, D)

        # Depth encoding + cross-attention with vision patches.
        depth_tokens = self.depth_encoder(depth)
        depth_tokens = self._normalize_depth_geom(depth_tokens)
        depth_tokens = self.depth_emb_to_model_emb_dim(depth_tokens)
        fused_vision_tokens = self.img_patch_emb_to_model_emb_dim(fused_vision_tokens)
        fused_vision_tokens = self.rgb_depth_fusion(q=fused_vision_tokens, kv=depth_tokens)

        global_vision_cls = self.img_cls_emb_to_model_emb_dim(global_vision_cls)

        robot_state_tokens = self.robot_state_dim_to_model_emb_dim(robot_states.unsqueeze(1))

        # Text encoding.
        tok = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_feat = self.text_encoder(**tok).last_hidden_state[:, 0]
        text_feat = self.text_proj(text_feat).unsqueeze(1)

        raw_tokens = torch.cat([global_vision_cls, fused_vision_tokens, robot_state_tokens, text_feat], dim=1)
        N = raw_tokens.shape[1]
        tokens = raw_tokens + self.final_pos_emb[:, :N]
        tokens = self.final_ln(self.final_transformer(tokens))

        global_token = tokens[:, 0]
        return global_token, tokens[:, 1:]


# ---------------------------------------------------------------------------
# RepresentationEncoderNoDepth  (robot-state optional, no dead parameters)
# ---------------------------------------------------------------------------

class RepresentationEncoderNoDepth(nn.Module):
    """Multi-view RGB + optional robot-state + text encoder (no depth branch)."""

    def __init__(
            self,
            image_encoder: nn.Module,
            robot_state_dim: int,
            model_emb_dim: int,
            resize_size: int,
            use_robot_state: bool = True,
            **kwargs
    ):
        super().__init__()

        self.model_emb_dim = model_emb_dim
        self.resize_size   = resize_size
        self.use_robot_state = use_robot_state

        self.image_encoder = image_encoder

        # ------------------------------------------------------------------
        # Probe encoder output dimension
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 3, resize_size, resize_size)
            img_enc_emb_dim = image_encoder(dummy)[0].shape[-1]

        # ------------------------------------------------------------------
        # Multi-view components
        # ------------------------------------------------------------------
        self.view_embed = nn.Parameter(torch.randn(3, 1, img_enc_emb_dim))
        self.view_fusion = _build_view_fusion(img_enc_emb_dim)

        self.view_attn = nn.Sequential(
            nn.Linear(img_enc_emb_dim, img_enc_emb_dim),
            nn.GELU(),
            nn.Linear(img_enc_emb_dim, 1),
        )

        self.img_resize = _make_img_resize_transform(resize_size)

        # ------------------------------------------------------------------
        # Projection layers
        # ------------------------------------------------------------------
        self.img_cls_emb_to_model_emb_dim   = nn.Linear(img_enc_emb_dim, model_emb_dim)
        self.img_patch_emb_to_model_emb_dim = nn.Linear(img_enc_emb_dim, model_emb_dim)

        if self.use_robot_state:
            self.robot_state_dim_to_model_emb_dim = nn.Linear(
                robot_state_dim,
                model_emb_dim
            )
        else:
            self.robot_state_dim_to_model_emb_dim = None

        # ------------------------------------------------------------------
        # Text encoder (frozen)
        # ------------------------------------------------------------------
        self.tokenizer    = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_encoder.eval()
        self.text_proj = nn.Linear(768, model_emb_dim)

        # ------------------------------------------------------------------
        # Final transformer
        # ------------------------------------------------------------------
        self.final_transformer, self.final_ln = \
            _build_final_transformer_encoder(model_emb_dim)

        # cls + patches + text (+ robot optional)
        max_tokens = 1 + (resize_size ** 2) + 1
        if self.use_robot_state:
            max_tokens += 1

        self.final_pos_emb = nn.Parameter(
            torch.zeros(1, max_tokens, model_emb_dim)
        )
        nn.init.trunc_normal_(self.final_pos_emb, std=0.02)

    # ----------------------------------------------------------------------
    # Device helper
    # ----------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        endoscope_image: torch.Tensor,
        wrist_l: torch.Tensor,
        wrist_r: torch.Tensor,
        robot_states: torch.Tensor = None,
        text: List[str] = None,
    ):
        # --------------------------------------------------------------
        # Resize images
        # --------------------------------------------------------------
        endo = self.img_resize(endoscope_image)
        wl   = self.img_resize(wrist_l)
        wr   = self.img_resize(wrist_r)

        # --------------------------------------------------------------
        # Image encoding
        # --------------------------------------------------------------
        cls_e,  _, patch_e  = self.image_encoder(endo)
        cls_wl, _, patch_wl = self.image_encoder(wl)
        cls_wr, _, patch_wr = self.image_encoder(wr)

        # Add view embeddings
        patch_e  = patch_e  + self.view_embed[0]
        patch_wl = patch_wl + self.view_embed[1]
        patch_wr = patch_wr + self.view_embed[2]

        multi_view_tokens   = torch.cat([patch_e, patch_wl, patch_wr], dim=1)
        fused_vision_tokens = self.view_fusion(multi_view_tokens)

        # Attention-weighted CLS fusion
        cls_stack    = torch.stack([cls_e, cls_wl, cls_wr], dim=1)
        attn_logits  = self.view_attn(cls_stack)
        attn_weights = torch.softmax(attn_logits, dim=1)
        global_vision_cls = (cls_stack * attn_weights).sum(dim=1)

        # Project to model dim
        global_vision_cls   = self.img_cls_emb_to_model_emb_dim(global_vision_cls)
        fused_vision_tokens = self.img_patch_emb_to_model_emb_dim(fused_vision_tokens)

        token_list = [
            global_vision_cls,
            fused_vision_tokens
        ]

        # --------------------------------------------------------------
        # Optional robot state branch
        # --------------------------------------------------------------
        if self.use_robot_state:
            if robot_states is None:
                raise ValueError("robot_states must be provided when use_robot_state=True")

            robot_state_tokens = self.robot_state_dim_to_model_emb_dim(
                robot_states.unsqueeze(1)
            )
            token_list.append(robot_state_tokens)

        # --------------------------------------------------------------
        # Text encoding
        # --------------------------------------------------------------
        if text is None:
            raise ValueError("text input must be provided")

        tok = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_feat = self.text_encoder(**tok).last_hidden_state[:, 0]

        text_feat = self.text_proj(text_feat).unsqueeze(1)
        token_list.append(text_feat)

        # --------------------------------------------------------------
        # Final transformer
        # --------------------------------------------------------------
        raw_tokens = torch.cat(token_list, dim=1)
        N = raw_tokens.shape[1]

        tokens = raw_tokens + self.final_pos_emb[:, :N]
        tokens = self.final_ln(self.final_transformer(tokens))

        global_token = tokens[:, 0]
        return global_token, tokens[:, 1:]