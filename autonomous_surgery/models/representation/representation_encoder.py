from typing import Any, List, Optional

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import pipeline, AutoTokenizer, AutoImageProcessor, AutoModel, CLIPTextModel, CLIPTokenizer

class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
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
            device: str,
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
        self.device = device

        pretrained_model_name = valid_model_names[model_name]
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.feature_extractor = AutoModel.from_pretrained(
            pretrained_model_name,
        )
        self.feature_extractor.to(self.device)

        # freeze the model weights. we do not want to train this
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
    
    def __call__(self, images: torch.Tensor, *args: Any, **kwds: Any) -> Any:
        inputs = self.processor(
            images=images, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.feature_extractor.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.feature_extractor(**inputs)
        outputs = outputs.last_hidden_state

        # we will return the cls token, register tokens, and img tokens
        R = 4                                # there are 4 register tokens
        cls_token = outputs[:, :1]           # (B, 1, D)
        reg_tokens = outputs[:, 1:1+R]       # (B, R, D)
        patch_tokens = outputs[:, 1+R:]      # (B, N, D)
        return cls_token, reg_tokens, patch_tokens
    
class DepthEncoder(nn.Module):
    def __init__(
        self,
        patched_dim: int,   # number of patches per side
        intrinsics: Optional[List[List[float]]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.intrinsics = intrinsics
        self.patched_dim = patched_dim
        self.eps = eps

    def guess_camera_intrinsics(
        self,
        H: int,
        W: int,
        fov_deg: float = 60.0,
        device=None,
    ):
        """
        Reasonable default camera intrinsics for debugging.

        Args:
            H, W: image height and width
            fov_deg: horizontal field of view in degrees

        Returns:
            K: [3, 3] intrinsics matrix
        """
        if device is None:
            device = "cpu"

        f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

        K = torch.tensor(
            [
                [f, 0.0, W / 2.0],
                [0.0, f, H / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        return K

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Encode local 3D geometry per patch (batched).

        Args:
            depth: [B, H, W] depth map in meters

        Returns:
            geom_feats: [B, N_patches, 8]
        """

        # return self.resize(depth).flatten(start_dim=1).unsqueeze(2)


        device = depth.device
        B, H, W = depth.shape

        # ------------------------------------------------------------
        # Intrinsics handling
        # ------------------------------------------------------------
        if self.intrinsics is None:
            K = self.guess_camera_intrinsics(
                H, W, device=device
            )  # [3, 3]
            K = K.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
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

        # ------------------------------------------------------------
        # Back-project depth to 3D
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Patchify (patched_dim x patched_dim patches)
        # ------------------------------------------------------------
        P = self.patched_dim

        # patch size in pixels (computed from input resolution)
        ps_h = H // P
        ps_w = W // P

        # crop so dimensions divide evenly
        points = points[:, : P * ps_h, : P * ps_w]

        # reshape into patches
        points = points.view(
            B,
            P, ps_h,
            P, ps_w,
            3
        ).permute(0, 1, 3, 2, 4, 5)   # [B, P, P, ps_h, ps_w, 3]

        # flatten patches
        patches = points.reshape(
            B,
            P * P,
            ps_h * ps_w,
            3
        )   # [B, N_patches, P_pixels, 3]

        # ------------------------------------------------------------
        # Mean 3D position
        # ------------------------------------------------------------
        mean_xyz = patches.mean(dim=2)  # [B, N, 3]

        # ------------------------------------------------------------
        # Covariance and normals
        # ------------------------------------------------------------
        centered = patches - mean_xyz.unsqueeze(2)  # [B, N, P, 3]

        cov = torch.matmul(
            centered.transpose(-1, -2),
            centered
        ) / (centered.shape[2] + self.eps)  # [B, N, 3, 3]

        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending

        normals = eigvecs[..., 0]  # [B, N, 3]
        normals = normals / (normals.norm(dim=-1, keepdim=True) + self.eps)

        # ------------------------------------------------------------
        # Depth variance
        # ------------------------------------------------------------
        z_vals = patches[..., 2]  # [B, N, P]
        depth_var = z_vals.var(dim=2, unbiased=False).unsqueeze(-1)  # [B, N, 1]

        # ------------------------------------------------------------
        # Planarity (λ1 / λ2)
        # ------------------------------------------------------------
        planarity = (
            eigvals[..., 1] / (eigvals[..., 2] + self.eps)
        ).unsqueeze(-1)  # [B, N, 1]

        # ------------------------------------------------------------
        # Final geometry descriptor
        # ------------------------------------------------------------
        geom_feats = torch.cat(
            [
                mean_xyz,   # 3
                normals,    # 3
                depth_var,  # 1
                planarity,  # 1
            ],
            dim=-1,
        )  # [B, N, 8]

        return geom_feats


class RepresentationEncoder(nn.Module):
    def __init__(
            self,
            device: str,
            image_encoder: nn.Module,
            depth_encoder: nn.Module,
            robot_state_dim: int,
            model_emb_dim: int,
            patched_dim: int,    # assuming square with resolution [patched_dim, patched_dim]
            resize_size: int,   # h, w the images are resized to before going through encoder
            **kwargs
        ):
        super().__init__()

        self.device = device
        self.model_emb_dim = model_emb_dim
        self.patched_dim = patched_dim
        self.resize_size = resize_size

        img_enc_emb_dim = image_encoder(torch.zeros((1, 3, resize_size, resize_size)))[0].shape[-1]
        self.view_embed = nn.Parameter(torch.randn(3, 1, img_enc_emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=img_enc_emb_dim,
            nhead=8,
            batch_first=True
        )
        self.view_fusion = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # used to determine the attention scores for each cls token for the images
        # so the model can learn how much weight to give to each global image representation
        self.view_attn = nn.Sequential(
            nn.Linear(img_enc_emb_dim, img_enc_emb_dim),
            nn.GELU(),
            nn.Linear(img_enc_emb_dim, 1)
        )

        self.img_resize = self._make_img_resize_transform()
        self.image_encoder = image_encoder
        self.img_cls_emb_to_model_emb_dim = nn.Linear(self._get_img_emb_dim(), model_emb_dim)
        self.img_reg_emb_to_model_emb_dim = nn.Linear(self._get_img_emb_dim(), model_emb_dim)
        self.img_patch_emb_to_model_emb_dim = nn.Linear(self._get_img_emb_dim(), model_emb_dim)
        self.depth_encoder = depth_encoder
        self.depth_emb_to_model_emb_dim = nn.Linear(self._get_depth_emb_dim(), model_emb_dim)
        self.robot_state_dim_to_model_emb_dim = nn.Linear(robot_state_dim, model_emb_dim)

        # for text encoding
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        self.text_proj = nn.Linear(768, self.model_emb_dim)

        self.rgb_depth_fusion = CrossAttentionBlock(
            emb_dim=self.model_emb_dim,
            num_heads=8,
        )

        self.final_transformer, self.final_pos_emb, self.final_ln = self._build_final_transformer_encoder()

    def _add_depth_geom_stats(self, depth_geom_mean: torch.Tensor, depth_geom_std: torch.Tensor):
        self.depth_geom_mean = depth_geom_mean
        self.depth_geom_std = depth_geom_std

    def _normalize_depth_geom(self, depth_encoder_output: torch.Tensor) -> torch.Tensor:
        return (depth_encoder_output - self.depth_geom_mean) / self.depth_geom_std

    def _get_img_emb_dim(self):
        image = torch.zeros((1, 3, self.resize_size, self.resize_size)).to(self.device)
        img_emb_d = self.image_encoder(image)[0].shape[-1]
        return img_emb_d
    
    def _get_depth_emb_dim(self):
        h = w = 1024 # this number doesn't matter for getting the depth embedding dims
        depth = torch.zeros((1, h, w))
        depth_emb_d = self.depth_encoder(depth).shape[-1]
        return depth_emb_d

    def _make_img_resize_transform(self):
        resize = v2.Resize((self.resize_size, self.resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        return v2.Compose([resize, to_float])
    
    def _build_final_transformer_encoder(self):
        self.num_transformer_layers = 6
        self.num_heads = 8
        self.ff_dim = 4 * self.model_emb_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_emb_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,   # IMPORTANT: your tensors are [B, T, D]
            norm_first=True,    # modern ViT-style
        )

        transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_transformer_layers,
        )

        self.max_tokens = 1 + 4 + (self.resize_size) ** 2 + 1
        pos_embed = nn.Parameter(
            torch.zeros(1, self.max_tokens, self.model_emb_dim)
        )
        nn.init.trunc_normal_(pos_embed, std=0.02)

        final_norm = nn.LayerNorm(self.model_emb_dim)

        return transformer, pos_embed, final_norm

    def forward(
        self,
        endoscope_image: torch.Tensor,
        depth: torch.Tensor,
        robot_states: torch.Tensor,
        text: List[str]
    ):
        # image encoding tokens
        # resized_imgs = self.img_resize(images)
        cls_tokens, reg_tokens, patch_tokens = self.image_encoder(images)

        cls_tokens = self.img_cls_emb_to_model_emb_dim(cls_tokens)
        reg_tokens = self.img_reg_emb_to_model_emb_dim(reg_tokens)
        patch_tokens = self.img_patch_emb_to_model_emb_dim(patch_tokens)

        # depth encoding tokens
        depth_tokens = self.depth_encoder(depth)
        depth_tokens = self._normalize_depth_geom(depth_tokens) # need to normalize here for training stability
        depth_tokens = self.depth_emb_to_model_emb_dim(depth_tokens)

        # robot state tokens
        robot_states = robot_states.unsqueeze(1)    # convert to [B, 1, robot_state_dim] shape
        robot_state_tokens = self.robot_state_dim_to_model_emb_dim(robot_states)

        # fuse depth and rgb tokens with cross attention
        fused_patch_tokens = self.rgb_depth_fusion(
            q=patch_tokens,
            kv=depth_tokens,
        )

        raw_token_encoding = torch.cat(
            (
                cls_tokens,
                reg_tokens,
                fused_patch_tokens,
                robot_state_tokens,
            ),
            dim=1,
        )

        N_tokens = raw_token_encoding.shape[1]
        tokens = raw_token_encoding + self.final_pos_emb[:, :N_tokens]

        # encode tokens into global representation in cls token
        tokens = self.final_ln(self.final_transformer(tokens))

        # take the CLS token, which should contain a global representation
        global_token = tokens[:, 0]

        # the encoder must return two things for DETR-style ACT
        # 1. A global_token for conditioning the learned posterior and prior
        #       encoder during CVAE training
        # 2. Memory tokens, which represent spatial, geometry-aware embeddings
        #       that are exactly what the actions should condition on. This
        #       will include the fused RGB + Detph tokens and the robot_state
        #       token

        return global_token, tokens[:, self.image_encoder.num_register_tokens+1:]
    
class RepresentationEncoderNoDepth(RepresentationEncoder):
        def __init__(
            self,
            device: str,
            image_encoder: nn.Module,
            depth_encoder: nn.Module,
            robot_state_dim: int,
            model_emb_dim: int,
            patched_dim: int,    # assuming square with resolution [patched_dim, patched_dim]
            resize_size: int,   # h, w the images are resized to before going through encoder
            **kwargs
        ):
            super().__init__(
                device,
                image_encoder,
                depth_encoder,
                robot_state_dim,
                model_emb_dim,
                patched_dim,    # assuming square with resolution [patched_dim, patched_dim]
                resize_size,   # h, w the images are resized to before going through encoder
                **kwargs
            )

        def forward(
            self,
            endoscope_image: torch.Tensor,
            wrist_l: torch.Tensor,
            wrist_r: torch.Tensor,
            robot_states: torch.Tensor,
            text: List[str]
        ):
            # image encoding tokens
            endo = self.img_resize(endoscope_image)
            wl = self.img_resize(wrist_l)
            wr = self.img_resize(wrist_r)

            # Encode each view with SAME encoder
            cls_e, reg_e, patch_e = self.image_encoder(endo)
            cls_wl, reg_wl, patch_wl = self.image_encoder(wl)
            cls_wr, reg_wr, patch_wr = self.image_encoder(wr)

            # this view embedding is necessary for the model to learn the position
            # of the features from different cameras
            patch_e  = patch_e  + self.view_embed[0]
            patch_wl = patch_wl + self.view_embed[1]
            patch_wr = patch_wr + self.view_embed[2]

            multi_view_tokens = torch.cat(
                [patch_e, patch_wl, patch_wr],
                dim=1
            )

            # fuse the patch tokens from multiple camera views into one group of 
            # fused tokens
            fused_vision_tokens = self.view_fusion(multi_view_tokens)

            # fuse the CLS tokens..
            # Stack CLS tokens
            cls_stack = torch.stack([cls_e, cls_wl, cls_wr], dim=1)  # (B, 3, D)

            # Compute attention weights
            attn_logits = self.view_attn(cls_stack)  # (B, 3, 1)
            attn_weights = torch.softmax(attn_logits, dim=1)

            # Weighted sum based on attention
            global_vision_cls = (cls_stack * attn_weights).sum(dim=1)

            # robot state tokens
            robot_states = robot_states.unsqueeze(1)    # convert to [B, 1, robot_state_dim] shape
            robot_state_tokens = self.robot_state_dim_to_model_emb_dim(robot_states)

            # encode text
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.text_encoder(**tokens)
            text_feat = outputs.last_hidden_state[:, 0]  # CLS token
            text_feat = self.text_proj(text_feat).unsqueeze(1)

            global_vision_cls = self.img_cls_emb_to_model_emb_dim(global_vision_cls)
            fused_vision_tokens = self.img_patch_emb_to_model_emb_dim(fused_vision_tokens)

            raw_token_encoding = torch.cat(
                (
                    global_vision_cls,
                    fused_vision_tokens,
                    robot_state_tokens,
                    text_feat,
                ),
                dim=1,
            )

            N_tokens = raw_token_encoding.shape[1]
            tokens = raw_token_encoding + self.final_pos_emb[:, :N_tokens]

            # encode tokens into global representation in cls token
            tokens = self.final_ln(self.final_transformer(tokens))

            # take the CLS token, which should contain a global representation
            global_token = tokens[:, 0]

            # the encoder must return two things for DETR-style ACT
            # 1. A global_token for conditioning the learned posterior and prior
            #       encoder during CVAE training
            # 2. Memory tokens, which represent spatial, geometry-aware embeddings
            #       that are exactly what the actions should condition on. This
            #       will include the fused RGB + Detph tokens and the robot_state
            #       token

            return global_token, tokens[:, 1:]


# old code that I may want to use later:
    # if self.upsample_patch_tokens:
    #     assert self.img_feat_upsampler is not None
    #     # we want to upsample the image tokens. to do so, we must change the shape
    #     # of the patch tokens from [B, N_patches, emb_dim] to [B, emb_dim, H_p, W_p]
    #     # where H and W are the height and width of the patches organized in a
    #     # square
    #     B, _, emb_dim = patch_tokens.shape
    #     H_p = W_p = self.resize_size // self.patched_dim
    #     if H_p * W_p != patch_tokens.shape[1]:
    #         raise Exception(f"Height and Width of reshaped patch tokens do not match total number of tokens.")
    #     patch_map = patch_tokens.reshape(B, H_p, W_p, emb_dim)
    #     patch_map = patch_map.permute(0, 3, 1, 2)  # (B, emb_dim, H_p, W_p)

    #     # upsample features to original image resolution
    #     patch_tokens = self.img_feat_upsampler(resized_imgs, patch_map)

    #     # flatten
    #     patch_tokens = patch_tokens.flatten(start_dim=2)
    #     patch_tokens = patch_tokens.permute(0, 2, 1)