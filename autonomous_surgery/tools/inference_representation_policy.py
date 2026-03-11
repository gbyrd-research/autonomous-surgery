#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import hydra
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class RepresentationPolicyInference:
    def __init__(
        self,
        config: DictConfig | str | pathlib.Path,
        checkpoint_path: str | pathlib.Path,
        robot_state_dim: int,
        action_dim: int,
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = self._load_config(config)
        self.device = torch.device(device)
        self.checkpoint_path = self._resolve_path(checkpoint_path)

        encoder_cfg = self.config.agent.instantiate_config.representation_encoder
        representation_encoder = instantiate(
            config=encoder_cfg,
            robot_state_dim=robot_state_dim,
        )

        self.model = instantiate(
            config=self.config.agent.instantiate_config,
            representation_encoder=representation_encoder,
            robot_state_dim=robot_state_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.load_checkpoint(self.checkpoint_path)
        self.model.eval()

    def _load_config(self, config: DictConfig | str | pathlib.Path) -> DictConfig:
        if isinstance(config, DictConfig):
            return config
        return OmegaConf.load(self._resolve_path(config))

    def _resolve_path(self, path: str | pathlib.Path) -> pathlib.Path:
        resolved = pathlib.Path(path).expanduser()
        if resolved.is_absolute():
            return resolved
        try:
            base_dir = pathlib.Path(hydra.utils.get_original_cwd())
        except ValueError:
            base_dir = pathlib.Path.cwd()
        return base_dir / resolved

    def load_checkpoint(self, checkpoint_path: str | pathlib.Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

    @torch.inference_mode()
    def infer(
        self,
        endoscope_image: torch.Tensor,
        wrist_l: torch.Tensor,
        wrist_r: torch.Tensor,
        robot_states: torch.Tensor,
        texts: Any = None,
        depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        endoscope_image = endoscope_image.to(self.device, non_blocking=True)
        wrist_l = wrist_l.to(self.device, non_blocking=True)
        wrist_r = wrist_r.to(self.device, non_blocking=True)
        robot_states = robot_states.to(self.device, non_blocking=True)
        if depth is not None:
            depth = depth.to(self.device, non_blocking=True)

        outputs = self.model(
            endoscope_image=endoscope_image,
            wrist_l=wrist_l,
            wrist_r=wrist_r,
            robot_states=robot_states,
            texts=texts,
            depth=depth,
        )

        actions = self.model.unnormalize_actions_mean_std(outputs.actions_norm)

        return actions, outputs

if __name__ == "__main__":

    hydra_output_dir = pathlib.Path("/home/grayson/surpass/outputs/2026-03-10/18-46-58")
    config = OmegaConf.load(hydra_output_dir.joinpath(".hydra/config.yaml"))
    checkpoint_path = hydra_output_dir.joinpath("last_model.pth")
    robot_state_dim = 8
    action_dim = 8
    text = "grasp"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model = RepresentationPolicyInference(
        config=config,
        checkpoint_path=checkpoint_path,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
        device=device,
    )

    endoscope_image = torch.randint(
        low=0,
        high=256,
        size=(1, 3, 540, 960),
        dtype=torch.uint8,
        device=device,
    )
    wrist_l = torch.randint(
        low=0,
        high=256,
        size=(1, 3, 540, 960),
        dtype=torch.uint8,
        device=device,
    )
    wrist_r = torch.randint(
        low=0,
        high=256,
        size=(1, 3, 480, 640),
        dtype=torch.uint8,
        device=device,
    )
    robot_states = torch.randn(1, robot_state_dim, device=device)
    texts = [text for _ in range(1)]

    for i in tqdm(range(1000)):
        actions, outputs = inference_model.infer(
            endoscope_image=endoscope_image,
            wrist_l=wrist_l,
            wrist_r=wrist_r,
            robot_states=robot_states,
            texts=texts,
        )

    print(f"Device: {device}")
    print(f"Checkpoint: {inference_model.checkpoint_path}")
    print(f"Actions shape: {tuple(actions.shape)}")
    print(f"Actions dtype: {actions.dtype}")
    print(f"Global feature shape: {tuple(outputs.global_feat.shape)}")
