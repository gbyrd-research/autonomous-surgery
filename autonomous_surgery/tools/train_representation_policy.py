#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import pathlib

import hydra
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from autonomous_surgery.helpers.common import Logger, set_seed
from autonomous_surgery.helpers.pytorch import AverageMeter
from autonomous_surgery.loss.act_vae_loss import act_vae_loss


# -----------------------------------------------------------
# DDP setup
# -----------------------------------------------------------

def setup_ddp():

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return rank, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# -----------------------------------------------------------
# Compute normalization statistics
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

    action_min = actions.amin(dim=(0,1))
    action_max = actions.amax(dim=(0,1))
    action_mean = actions.mean(dim=(0,1))
    action_std = actions.std(dim=(0,1))

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
# Broadcast model buffers (normalization stats)
# -----------------------------------------------------------

def broadcast_model_buffers(model):

    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="train_representation_policy")
def main(config):

    rank, local_rank = setup_ddp()

    set_seed(config.seed + rank)

    device = torch.device(f"cuda:{local_rank}")

    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

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

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        sampler=train_sampler,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.dataloader.batch_size,
        sampler=valid_sampler,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
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

    # -------------------------------------------------------
    # Compute normalization (rank 0)
    # -------------------------------------------------------

    if rank == 0:
        compute_norm_stats(model, train_dataset, config)

    dist.barrier()

    broadcast_model_buffers(model)

    # -------------------------------------------------------
    # Wrap DDP
    # -------------------------------------------------------

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    raw_model = model.module

    # -------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    scaler = GradScaler(enabled=False)  # BF16 doesn't need scaling

    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    kl_weight = config.train.kl_weight

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------

    for epoch in range(config.train.num_epochs):

        train_sampler.set_epoch(epoch)

        model.train()

        loss_meter = AverageMeter()
        recon_meter = AverageMeter()
        kl_meter = AverageMeter()

        for batch in tqdm(train_loader, disable=(rank != 0)):

            endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

            endoscope_image = endoscope_image.to(device, non_blocking=True)
            wrist_l = wrist_l.to(device, non_blocking=True)
            wrist_r = wrist_r.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            action_chunk = action_chunk.to(device, non_blocking=True)

            actions_norm = raw_model.normalize_actions_mean_std(action_chunk)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):

                preds = model(
                    endoscope_image,
                    wrist_l,
                    wrist_r,
                    state,
                    instruction_text,
                    actions_norm,
                    action_is_pad
                )

                loss, loss_dict = act_vae_loss(
                    preds,
                    actions_norm,
                    action_is_pad,
                    kl_weight
                )

            loss.backward()

            optimizer.step()

            loss_meter.update(loss_dict["loss"].item())
            recon_meter.update(loss_dict["recon"].item())
            kl_meter.update(loss_dict["kl"].item())

        if rank == 0:
            Logger.log_info(
                f"Epoch {epoch} | "
                f"Train Loss: {loss_meter.avg:.6f} | "
                f"Recon: {recon_meter.avg:.6f} | "
                f"KL: {kl_meter.avg:.6f}"
            )

        # -------------------------------------------------------
        # Validation
        # -------------------------------------------------------

        do_validation = (
            epoch >= config.evaluation.num_skip_epochs and
            epoch % config.evaluation.validation_frequency_epochs == 0
        )

        if do_validation:

            model.eval()

            val_meter = AverageMeter()
            val_recon_meter = AverageMeter()
            val_kl_meter = AverageMeter()

            with torch.no_grad():

                for batch in tqdm(valid_loader, disable=(rank != 0)):

                    endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

                    endoscope_image = endoscope_image.to(device, non_blocking=True)
                    wrist_l = wrist_l.to(device, non_blocking=True)
                    wrist_r = wrist_r.to(device, non_blocking=True)
                    state = state.to(device, non_blocking=True)
                    action_chunk = action_chunk.to(device, non_blocking=True)

                    actions_norm = raw_model.normalize_actions_mean_std(action_chunk)

                    with autocast("cuda", dtype=torch.bfloat16):

                        preds = model(
                            endoscope_image,
                            wrist_l,
                            wrist_r,
                            state,
                            instruction_text,
                            actions_norm,
                            action_is_pad
                        )

                        loss, loss_dict = act_vae_loss(
                            preds,
                            actions_norm,
                            action_is_pad,
                            kl_weight
                        )

                    val_meter.update(loss_dict["loss"].item())
                    val_recon_meter.update(loss_dict["recon"].item())
                    val_kl_meter.update(loss_dict["kl"].item())

            if rank == 0:
                if rank == 0:
                    Logger.log_info(
                        f"Epoch {epoch} | "
                        f"Val Loss: {val_meter.avg:.6f} | "
                        f"Recon: {val_recon_meter.avg:.6f} | "
                        f"KL: {val_kl_meter.avg:.6f}"
                    )

        # -------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------

        if rank == 0:

            torch.save(
                raw_model.state_dict(),
                output_dir / "last_model.pth"
            )

    if rank == 0:
        Logger.log_ok("Training finished!")

    cleanup_ddp()


if __name__ == "__main__":
    main()