#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
autonomous_surgery/tools/train_representation_policy_new.py

Representation ACT training script with torchrun / DistributedDataParallel (DDP) support.

Launch with:
    torchrun --nproc_per_node=<NUM_GPUS> train_representation_policy_new.py [hydra overrides]

Single-GPU / CPU (unchanged behaviour):
    python train_representation_policy_new.py [hydra overrides]
"""

from __future__ import annotations

import json
import functools
import os
import pathlib
from typing import Optional, Tuple, Any, Dict

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from autonomous_surgery.helpers.common import Logger, WandBLogger, set_seed
from autonomous_surgery.helpers.pytorch import AverageMeter


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> Tuple[int, int, int]:
    """
    Initialise the default process group when launched via torchrun.
    Falls back gracefully to single-process mode when the env vars are absent.

    Returns
    -------
    rank        : global rank of this process (0 … world_size-1)
    local_rank  : local rank on this node  (used to select the GPU)
    world_size  : total number of processes
    """
    if "RANK" not in os.environ:
        # Not launched via torchrun – run as a regular single process.
        return 0, 0, 1

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Dataset fingerprint / normalisation persistence
# ---------------------------------------------------------------------------


def _to_item(v):
    if torch.is_tensor(v):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Normalisation-stat computation (rank-0 computes, then broadcasts to all)
# ---------------------------------------------------------------------------

def _compute_and_set_norm_stats(
    model, train_dataset, config, rank: int, world_size: int
) -> Tuple[int, int]:
    """
    Rank-0 computes statistics over the training set and broadcasts them to
    every other rank so that all processes use identical normalisation values.
    """
    device = torch.device(config.device if world_size == 1 else f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # Unwrap DDP to access model attributes / methods directly.
    raw_model = model.module if isinstance(model, DDP) else model

    # ------------------------------------------------------------------
    # 1.  Rank-0 computes stats
    # ------------------------------------------------------------------
    if is_main_process(rank):
        Logger.print_seperator()
        Logger.log_info(colored(" [Auto-Norm] Calculating dataset statistics...", "cyan"))

        stats_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
        )

        all_actions, all_qpos = [], []
        collected_samples = 0
        max_samples = 10_000

        raw_model.eval()
        with torch.no_grad():
            for batch in tqdm(stats_loader, desc="Computing Stats"):
                (
                    endoscope_images,
                    wrist_l,
                    wrist_r,
                    robot_states,
                    actions,
                    is_pad,
                    instruction_text,
                ) = batch

                all_actions.append(actions)
                all_qpos.append(robot_states)

                collected_samples += actions.shape[0]
                if collected_samples >= max_samples:
                    break

        all_actions = torch.cat(all_actions, dim=0)
        flat_actions = all_actions.reshape(-1, all_actions.shape[-1])
        action_mean = flat_actions.mean(dim=0)
        action_std = torch.clamp(flat_actions.std(dim=0), min=1e-5)

        all_qpos = torch.cat(all_qpos, dim=0)
        flat_qpos = all_qpos.reshape(-1, all_qpos.shape[-1])
        qpos_mean = flat_qpos.mean(dim=0)
        qpos_std = torch.clamp(flat_qpos.std(dim=0), min=1e-5)

        Logger.log_info(f" [Auto-Norm] Action Mean: {action_mean[:3].tolist()}...")
        Logger.log_info(f" [Auto-Norm] Action Std:  {action_std[:3].tolist()}...")
        Logger.print_seperator()
    else:
        # Non-root ranks create empty placeholders; shapes are filled after broadcast.
        action_mean = torch.zeros(1)
        action_std = torch.zeros(1)
        qpos_mean = torch.zeros(1)
        qpos_std = torch.zeros(1)

    # ------------------------------------------------------------------
    # 2.  Broadcast from rank-0 to all other ranks
    # ------------------------------------------------------------------
    if world_size > 1:
        # Broadcast tensor shapes first so non-root ranks can allocate.
        shapes = torch.zeros(2, dtype=torch.long, device=device)
        if is_main_process(rank):
            shapes[0] = action_mean.shape[0]
            shapes[1] = qpos_mean.shape[0]

        dist.broadcast(shapes, src=0)

        a_dim, q_dim = shapes.tolist()

        for tensor, dim, name in [
            (action_mean,     a_dim,  "action_mean"),
            (action_std,      a_dim,  "action_std"),
            (qpos_mean,       q_dim,  "qpos_mean"),
            (qpos_std,        q_dim,  "qpos_std"),
        ]:
            if not is_main_process(rank):
                tensor = torch.zeros(dim, device=device)
            else:
                tensor = tensor.to(device)

            dist.broadcast(tensor, src=0)

            # Re-bind in local scope so set_norm_stats receives correct tensors.
            if name == "action_mean":      action_mean      = tensor
            elif name == "action_std":     action_std       = tensor
            elif name == "qpos_mean":      qpos_mean        = tensor
            elif name == "qpos_std":       qpos_std         = tensor
    else:
        action_mean      = action_mean.to(device)
        action_std       = action_std.to(device)
        qpos_mean        = qpos_mean.to(device)
        qpos_std         = qpos_std.to(device)

    # ------------------------------------------------------------------
    # 3.  Inject into the model on every rank
    # ------------------------------------------------------------------
    if hasattr(raw_model, "set_norm_stats"):
        raw_model.set_norm_stats(
            action_mean, action_std,
            qpos_mean, qpos_std,
        )
        if is_main_process(rank):
            Logger.log_info(
                colored(" [Auto-Norm] Statistics injected into model successfully!", "green")
            )
    else:
        if is_main_process(rank):
            Logger.log_warning(
                colored(" [Warning] Model missing 'set_norm_stats'. Auto-Norm skipped.", "yellow")
            )

    return int(qpos_mean.shape[0]), int(action_mean.shape[0])


# ---------------------------------------------------------------------------
# Loss reduction across ranks
# ---------------------------------------------------------------------------

def reduce_loss(loss: torch.Tensor, world_size: int) -> torch.Tensor:
    """Average loss tensor across all DDP processes."""
    if world_size > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / world_size
    return loss


def meter_value_or_na(meter: AverageMeter) -> str:
    if meter.count == 0:
        return "n/a"
    return f"{meter.avg:.6f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="train_representation_policy")
def main(config):

    rank, local_rank, world_size = setup_ddp()

    # Each rank logs to its own seed offset so RNG states diverge intentionally.
    set_seed(config.seed + rank)

    # Determine per-rank device.
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
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

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    # ------------------------------------------------------------------
    # Samplers  (DistributedSampler shards data across ranks)
    # ------------------------------------------------------------------
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=config.dataloader.shuffle)
        if world_size > 1
        else None
    )
    valid_sampler = (
        DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

    DataLoaderConstructor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    sample_batch = next(iter(DataLoaderConstructor(train_dataset, shuffle=False)))
    _, _, _, sample_robot_state, sample_action, _, _ = sample_batch

    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=sample_robot_state.size(-1),
        action_dim=sample_action.size(-1),
    ).to(device)

    robot_state_dim, action_dim = _compute_and_set_norm_stats(
        model, train_dataset, config, rank, world_size
    )

    if is_main_process(rank):
        Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
        Logger.log_info(f'Action dim: {colored(action_dim, "red")}')

    # Wrap with DDP *after* norm stats are set (they live in raw model buffers).
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Convenience accessor for model internals regardless of DDP wrapping.
    raw_model = model.module if isinstance(model, DDP) else model

    # ------------------------------------------------------------------
    # DataLoaders  (shuffle=False when using DistributedSampler)
    # ------------------------------------------------------------------
    train_loader = DataLoaderConstructor(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None and config.dataloader.shuffle),
    )
    valid_loader = DataLoaderConstructor(
        valid_dataset,
        sampler=valid_sampler,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=OmegaConf.select(config, "train.weight_decay", default=1e-4),
    )

    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    kl_weight = OmegaConf.select(config, "train.kl_weight", default=None)
    if kl_weight is None:
        kl_weight = OmegaConf.select(config, "agent.kl_weight", default=None)
    if kl_weight is None:
        kl_weight = OmegaConf.select(config, "agent.instantiate_config.kl_weight", default=10.0)
    kl_weight = float(kl_weight)

    if is_main_process(rank):
        Logger.log_info(f"KL weight: {colored(kl_weight, 'red')}")

    # ------------------------------------------------------------------
    # Optional evaluator (rank-0 only)
    # ------------------------------------------------------------------
    evaluator = None
    if is_main_process(rank) and getattr(config.benchmark, "evaluator_instantiate_config", None) is not None:
        try:
            evaluator = instantiate(
                config=config.benchmark.evaluator_instantiate_config,
                task_name=config.task_name,
            )
        except Exception as e:
            Logger.log_warning(f"Failed to instantiate evaluator: {e}")

    best_success = -1.0
    best_val_loss = float("inf")
    clip_grad_value = OmegaConf.select(config, "train.clip_grad_value", default=10.0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for cur_epoch in range(config.train.num_epochs):

        # Inform the sampler of the current epoch so shuffling is correct.
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)

        model.train()
        loss_train = AverageMeter()
        recon_train = AverageMeter()
        kl_train = AverageMeter()
        weighted_kl_train = AverageMeter()
        pad_bce_train = AverageMeter()

        train_pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {cur_epoch+1}",
            leave=False,
            disable=not is_main_process(rank),   # only rank-0 shows the bar
        )

        for cur_iter, batch in enumerate(train_pbar):
            iteration_info: Dict[str, Any] = {}

            endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

            endoscope_image  = endoscope_image.to(device, non_blocking=True)
            wrist_l          = wrist_l.to(device, non_blocking=True)
            wrist_r          = wrist_r.to(device, non_blocking=True)
            state            = state.to(device, non_blocking=True)
            action_chunk     = action_chunk.to(device, non_blocking=True)
            action_is_pad    = action_is_pad.to(device, non_blocking=True)

            actions_norm = (action_chunk - raw_model.action_mean) / raw_model.action_std

            preds = model(
                endoscope_image,
                wrist_l,
                wrist_r,
                state,
                instruction_text,
                action_chunk=action_chunk,
                action_is_pad=action_is_pad,
            )

            loss_result = call(
                config.benchmark.loss_func,
                preds,
                actions_norm,
                kl_weight=kl_weight,
                is_pad=action_is_pad,
                include_is_pad_loss=True
            )

            if isinstance(loss_result, tuple):
                loss, loss_dict = loss_result
                if isinstance(loss_dict, dict) and is_main_process(rank):
                    for k, v in loss_dict.items():
                        iteration_info[f"train_iteration/{k}"] = _to_item(v)
            else:
                loss = loss_result
                loss_dict = {}

            # Average loss across all ranks before backward so gradients are
            # consistent (DDP already all-reduces gradients, but this keeps
            # the logged scalar accurate).
            loss_scalar = reduce_loss(loss.detach().clone(), world_size).item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if clip_grad_value > 0.0:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

            optimizer.step()
            loss_train.update(loss_scalar)

            recon_value = loss_dict.get("recon", None)
            if recon_value is not None:
                recon_scalar = reduce_loss(recon_value.detach().clone(), world_size).item()
                recon_train.update(recon_scalar)

            kl_value = loss_dict.get("kl", None)
            if kl_value is not None:
                kl_scalar = reduce_loss(kl_value.detach().clone(), world_size).item()
                kl_train.update(kl_scalar)
                weighted_kl_train.update(kl_weight * kl_scalar)

            pad_value = loss_dict.get("pad_bce", None)
            if pad_value is not None:
                pad_scalar = reduce_loss(pad_value.detach().clone(), world_size).item()
                pad_bce_train.update(pad_scalar)

            if is_main_process(rank):
                iteration_info.update(
                    {
                        "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
                        "train_iteration/epoch": cur_epoch,
                        "train_iteration/loss": loss_scalar,
                        "train_iteration/learning_rate": scheduler.get_last_lr()[0],
                    }
                )

        scheduler.step()

        if is_main_process(rank):
            Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg:.6f}")
            train_breakdown = (
                f"[train] epoch={cur_epoch}, recon={meter_value_or_na(recon_train)}, "
                f"kl={meter_value_or_na(kl_train)}, "
                f"weighted_kl={meter_value_or_na(weighted_kl_train)}, "
                f"pad_bce={meter_value_or_na(pad_bce_train)}"
            )
            Logger.log_info(train_breakdown)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
            (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs

        if periodic_validation or last_epoch:
            if valid_sampler is not None:
                valid_sampler.set_epoch(cur_epoch)

            model.eval()
            loss_val = AverageMeter()
            recon_val = AverageMeter()
            kl_val = AverageMeter()
            weighted_kl_val = AverageMeter()
            pad_bce_val = AverageMeter()

            valid_pbar = tqdm(
                valid_loader,
                desc=f"Validation Epoch {cur_epoch+1}",
                leave=False,
                disable=not is_main_process(rank),
            )

            with torch.no_grad():
                for batch in valid_pbar:
                    endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

                    endoscope_image  = endoscope_image.to(device, non_blocking=True)
                    wrist_l          = wrist_l.to(device, non_blocking=True)
                    wrist_r          = wrist_r.to(device, non_blocking=True)
                    state            = state.to(device, non_blocking=True)
                    action_chunk     = action_chunk.to(device, non_blocking=True)
                    action_is_pad    = action_is_pad.to(device, non_blocking=True)

                    actions_norm = (action_chunk - raw_model.action_mean) / raw_model.action_std

                    preds = model(
                        endoscope_image,
                        wrist_l,
                        wrist_r,
                        state,
                        instruction_text,
                        action_chunk=action_chunk,
                        action_is_pad=action_is_pad,
                    )

                    loss_result = call(
                        config.benchmark.loss_func,
                        preds,
                        actions_norm,
                        kl_weight=kl_weight,
                        is_pad=action_is_pad,
                        include_is_pad_loss=True
                    )

                    if isinstance(loss_result, tuple):
                        val_loss, val_loss_dict = loss_result
                    else:
                        val_loss = loss_result
                        val_loss_dict = {}
                    val_loss_scalar = reduce_loss(val_loss.detach().clone(), world_size).item()
                    loss_val.update(val_loss_scalar, action_chunk.shape[0])

                    val_recon = val_loss_dict.get("recon", None)
                    if val_recon is not None:
                        val_recon_scalar = reduce_loss(val_recon.detach().clone(), world_size).item()
                        recon_val.update(val_recon_scalar, action_chunk.shape[0])

                    val_kl = val_loss_dict.get("kl", None)
                    if val_kl is not None:
                        val_kl_scalar = reduce_loss(val_kl.detach().clone(), world_size).item()
                        kl_val.update(val_kl_scalar, action_chunk.shape[0])
                        weighted_kl_val.update(kl_weight * val_kl_scalar, action_chunk.shape[0])

                    val_pad = val_loss_dict.get("pad_bce", None)
                    if val_pad is not None:
                        val_pad_scalar = reduce_loss(val_pad.detach().clone(), world_size).item()
                        pad_bce_val.update(val_pad_scalar, action_chunk.shape[0])

            if is_main_process(rank):
                Logger.log_info(f"[validation] epoch={cur_epoch}, val_loss={loss_val.avg:.6f}")
                val_breakdown = (
                    f"[validation] epoch={cur_epoch}, recon={meter_value_or_na(recon_val)}, "
                    f"kl={meter_value_or_na(kl_val)}, "
                    f"weighted_kl={meter_value_or_na(weighted_kl_val)}, "
                    f"pad_bce={meter_value_or_na(pad_bce_val)}"
                )
                Logger.log_info(val_breakdown)

                avg_success = 0.0
                saved = False

                if (
                    evaluator is not None
                    and config.evaluation.save_best_model
                    and avg_success > best_success
                ):
                    best_success = avg_success
                    saved = True

                if config.evaluation.save_best_model and loss_val.avg < best_val_loss:
                    best_val_loss = loss_val.avg
                    if evaluator is None:
                        saved = True

                if saved:
                    model_path = os.path.join(local_run_output_dir, "best_model.pth")
                    # Save raw model weights (not the DDP wrapper).
                    torch.save(raw_model.state_dict(), model_path)
                    Logger.log_info(f"Save best model to {colored(model_path, 'red')}")

                    with open(os.path.join(local_run_output_dir, "best_model.json"), "w") as f:
                        json.dump(
                            {
                                "epoch": cur_epoch,
                                "val_loss": loss_val.avg,
                                "avg_success": avg_success,
                            },
                            f,
                            indent=4,
                        )

                torch.save(
                    raw_model.state_dict(),
                    os.path.join(local_run_output_dir, "last_model.pth"),
                )

        # Barrier keeps ranks in sync at the end of every epoch.
        if world_size > 1:
            dist.barrier()

    if is_main_process(rank):
        Logger.log_ok("Training Finished!")

    cleanup_ddp()


if __name__ == "__main__":
    main()
