#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic training script with normalization.

Features:
- Hydra configs
- Dataset normalization stats
- Checkpoint saving
"""

from __future__ import annotations

import pathlib

import hydra
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from autonomous_surgery.helpers.common import Logger, set_seed
from autonomous_surgery.helpers.pytorch import AverageMeter


# -----------------------------------------------------------
# Compute dataset normalization statistics
# -----------------------------------------------------------

def compute_norm_stats(model, dataset, config):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
    )

    actions = []
    states = []

    for batch in tqdm(loader, desc="Computing normalization stats"):

        _, _, _, state, action, _, _ = batch

        actions.append(action)
        states.append(state)

        if len(actions) > 200:
            break

    actions = torch.cat(actions)
    states = torch.cat(states)

    # --------------------------------------------------
    # ACTION STATS
    # --------------------------------------------------

    action_min = actions.amin(dim=(0,1))
    action_max = actions.amax(dim=(0,1))

    action_mean = actions.mean(dim=(0,1))
    action_std = actions.std(dim=(0,1))

    # avoid divide-by-zero
    action_range = (action_max - action_min).clamp(min=1e-6)

    # --------------------------------------------------
    # STATE STATS (still standardize)
    # --------------------------------------------------

    state_mean = states.mean(dim=(0,1))
    state_std = states.std(dim=(0,1)).clamp(min=1e-5)

    model.set_norm_stats(
        action_min,
        action_max,
        action_mean,
        action_std,
        state_mean,
        state_std
    )

    Logger.log_info("Normalization stats computed")


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="train_representation_policy_debug")
def main(config):

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    train_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        chunk_size=config.agent.instantiate_config.chunk_size,
        split="train",
    )

    valid_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        chunk_size=config.agent.instantiate_config.chunk_size,
        split="test",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
    )

    # -------------------------------------------------------
    # Infer dimensions
    # -------------------------------------------------------

    sample_batch = next(iter(train_loader))
    _, _, _, sample_state, sample_action, _, _ = sample_batch

    state_dim = sample_state.shape[-1]
    action_dim = sample_action.shape[-1]

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    encoder_cfg = config.agent.instantiate_config.representation_encoder

    representation_encoder = instantiate(
        config=encoder_cfg,
        robot_state_dim=state_dim,
    )

    model = instantiate(
        config=config.agent.instantiate_config,
        representation_encoder=representation_encoder,
        robot_state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)

    raw_model = model

    # -------------------------------------------------------
    # Compute normalization
    # -------------------------------------------------------

    compute_norm_stats(model, train_dataset, config)

    # -------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    use_norm = True
    for epoch in range(config.train.num_epochs):

        model.train()

        loss_meter = AverageMeter()

        for batch in tqdm(train_loader):

            endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

            endoscope_image = endoscope_image.to(device)
            wrist_l = wrist_l.to(device)
            wrist_r = wrist_r.to(device)
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            # normalize actions
            actions_norm = raw_model.normalize_actions_mean_std(action_chunk)

            optimizer.zero_grad()

            preds = model(
                endoscope_image,
                wrist_l,
                wrist_r,
                state,
                instruction_text,
                actions_norm,
                action_is_pad
            )

            if use_norm:
                loss = torch.nn.functional.mse_loss(
                    preds.actions,
                    actions_norm
                )
            else:
                loss = torch.nn.functional.mse_loss(
                    preds.actions,
                    action_chunk
                )

            loss.backward()

            optimizer.step()

            loss_meter.update(loss.item())

        Logger.log_info(f"Epoch {epoch} | Train Loss: {loss_meter.avg:.6f}")

        # -------------------------------------------------------
        # Validation
        # -------------------------------------------------------

        # model.eval()

        # val_meter = AverageMeter()

        # with torch.no_grad():

        #     for batch in valid_loader:

        #         endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

        #         endoscope_image = endoscope_image.to(device)
        #         wrist_l = wrist_l.to(device)
        #         wrist_r = wrist_r.to(device)
        #         state = state.to(device)
        #         action_chunk = action_chunk.to(device)

        #         actions_norm = raw_model.normalize_actions_mean_std(action_chunk)

        #         preds = model(
        #             endoscope_image,
        #             wrist_l,
        #             wrist_r,
        #             state,
        #             instruction_text,
        #             actions_norm,
        #             action_is_pad
        #         )

        #         if use_norm:
        #             loss = torch.nn.functional.mse_loss(
        #                 preds.actions,
        #                 actions_norm
        #             )
        #         else:
        #             loss = torch.nn.functional.mse_loss(
        #                 preds.actions,
        #                 action_chunk
        #             )

        #         val_meter.update(loss.item())

        # Logger.log_info(f"Epoch {epoch} | Val Loss: {val_meter.avg:.6f}")

        torch.save(
            raw_model.state_dict(),
            output_dir / "last_model.pth"
        )

    Logger.log_ok("Training finished!")


if __name__ == "__main__":
    main()