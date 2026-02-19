#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
autonomous_surgery/tools/train_representation_policy_new.py

Representation ACT training script that follows the same training process as
autonomous_surgery/tools/act_policy_normal.py, while preserving representation-specific
normalization stats (including depth geometry stats).
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
from typing import Optional, Tuple, Any, Dict

import hydra
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from autonomous_surgery.helpers.common import Logger, WandBLogger, set_seed
from autonomous_surgery.helpers.pytorch import AverageMeter

# def _unpack_batch(
#     batch,
# ) -> Tuple[
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     Any,
#     torch.Tensor,
#     Optional[torch.Tensor],
#     Any,
# ]:
#     """Handle both 7-field and 8-field batch layouts."""
#     if not isinstance(batch, (tuple, list)):
#         raise ValueError(f"Batch must be tuple/list, got {type(batch)}")

#     (endoscope_image,
#     wrist_l,
#     wrist_r,
#     state,
#     action_chunk,
#     is_pad,
#     instruction_text) = batch

#     return images, point_clouds, depth, robot_states, raw_states, actions, is_pad, texts


def _to_item(v):
    if torch.is_tensor(v):
        return v.item()
    return v


def _compute_and_set_norm_stats(model, train_dataset, config) -> Tuple[int, int]:
    Logger.print_seperator()
    Logger.log_info(colored(" [Auto-Norm] Calculating dataset statistics...", "cyan"))

    no_depth = not config.use_depth

    stats_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
    )

    all_actions = []
    all_qpos = []
    all_geom_feats = [] if not no_depth else None

    max_samples_for_stats = 10000
    collected_samples = 0

    model.eval()
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

            if not no_depth:
                # Make sure `depth` is properly defined in your batch
                geom = model.representation_encoder.depth_encoder(depth)  # [B, N, Dg]
                all_geom_feats.append(geom)

            collected_samples += actions.shape[0]
            if collected_samples >= max_samples_for_stats:
                break

    # --- Action stats ---
    all_actions = torch.cat(all_actions, dim=0)
    flat_actions = all_actions.reshape(-1, all_actions.shape[-1])
    action_mean = flat_actions.mean(dim=0)
    action_std = torch.clip(flat_actions.std(dim=0), min=1e-5)

    # --- Qpos stats ---
    all_qpos = torch.cat(all_qpos, dim=0)
    flat_qpos = all_qpos.reshape(-1, all_qpos.shape[-1])
    qpos_mean = flat_qpos.mean(dim=0)
    qpos_std = torch.clip(flat_qpos.std(dim=0), min=1e-5)

    # --- Depth stats (optional) ---
    if not no_depth:
        flat_depth_geom = torch.cat(all_geom_feats, dim=0).reshape(
            -1, model.representation_encoder._get_depth_emb_dim()
        )
        depth_geom_mean = flat_depth_geom.mean(dim=0)
        depth_geom_std = torch.clip(flat_depth_geom.std(dim=0), min=1e-5)
    else:
        depth_geom_mean = None
        depth_geom_std = None

    Logger.log_info(f" [Auto-Norm] Action Mean: {action_mean[:3].tolist()}...")
    Logger.log_info(f" [Auto-Norm] Action Std:  {action_std[:3].tolist()}...")

    if hasattr(model, "set_norm_stats"):
        model.set_norm_stats(
            action_mean.to(config.device),
            action_std.to(config.device),
            depth_geom_mean.to(config.device) if depth_geom_mean is not None else None,
            depth_geom_std.to(config.device) if depth_geom_std is not None else None,
            qpos_mean.to(config.device),
            qpos_std.to(config.device),
        )
        Logger.log_info(colored(" [Auto-Norm] Statistics injected into model successfully!", "green"))
    else:
        Logger.log_warning(colored(" [Warning] Model missing 'set_norm_stats'. Auto-Norm skipped.", "yellow"))

    Logger.print_seperator()
    return int(qpos_mean.shape[0]), int(action_mean.shape[0])


@hydra.main(version_base=None, config_path="../config", config_name="train_representation_policy")
def main(config):

    set_seed(config.seed)

    train_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        chunk_size=config.agent.instantiate_config.chunk_size,
        split="train",
    )
    valid_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        chunk_size=config.agent.instantiate_config.chunk_size,
        split="val",
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    DataLoaderConstructor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )

    # Instantiate once so we can compute depth-geom stats from its depth encoder.
    sample_batch = next(iter(DataLoaderConstructor(train_dataset, shuffle=False)))
    _, _, _, sample_robot_state, sample_action, _, _ = sample_batch

    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=sample_robot_state.size(-1),
        action_dim=sample_action.size(-1),
    ).to(config.device)

    robot_state_dim, action_dim = _compute_and_set_norm_stats(model, train_dataset, config)
    Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
    Logger.log_info(f'Action dim: {colored(action_dim, "red")}')

    train_loader = DataLoaderConstructor(train_dataset, shuffle=config.dataloader.shuffle)
    valid_loader = DataLoaderConstructor(valid_dataset, shuffle=False)

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
    Logger.log_info(f"KL weight: {colored(kl_weight, 'red')}")

    evaluator: Optional[Evaluator] = None
    if getattr(config.benchmark, "evaluator_instantiate_config", None) is not None:
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

    for cur_epoch in range(config.train.num_epochs):
        epoch_logging_info: Dict[str, Any] = {"epoch_step": cur_epoch + 1}

        model.train()
        loss_train = AverageMeter()

        train_pbar = tqdm(
            train_loader,
            desc=f"Training for Epoch {cur_epoch+1}",
            leave=False
        )

        for cur_iter, batch in enumerate(train_pbar):
            iteration_info: Dict[str, Any] = {}

            endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch

            endoscope_image = endoscope_image.to(config.device)
            wrist_l = wrist_l.to(config.device)
            wrist_r = wrist_r.to(config.device)
            state = state.to(config.device)
            action_chunk = action_chunk.to(config.device)
            action_is_pad = action_is_pad.to(config.device)

            actions_norm = (action_chunk - model.action_mean) / model.action_std
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
                is_pad=action_is_pad
            )

            if isinstance(loss_result, tuple):
                loss, loss_dict = loss_result
                if isinstance(loss_dict, dict):
                    for k, v in loss_dict.items():
                        iteration_info[f"train_iteration/{k}"] = _to_item(v)
            else:
                loss = loss_result

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if clip_grad_value > 0.0:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

            optimizer.step()
            loss_train.update(loss.item())

            iteration_info.update(
                {
                    "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
                    "train_iteration/epoch": cur_epoch,
                    "train_iteration/loss": loss.item(),
                    "train_iteration/learning_rate": scheduler.get_last_lr()[0],
                }
            )

        scheduler.step()
        epoch_logging_info["train_epoch/epoch_loss"] = loss_train.avg
        Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg:.6f}")

        periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
            (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs

        if periodic_validation or last_epoch:
            model.eval()
            loss_val = AverageMeter()

            valid_pbar = tqdm(
                valid_loader,
                desc=f"Validation for Epoch {cur_epoch+1}",
                leave=False
            )

            with torch.no_grad():
                for batch in valid_pbar:
                    endoscope_image, wrist_l, wrist_r, state, action_chunk, action_is_pad, instruction_text = batch


                    endoscope_image = endoscope_image.to(config.device)
                    wrist_l = wrist_l.to(config.device)
                    wrist_r = wrist_r.to(config.device)
                    state = state.to(config.device)
                    action_chunk = action_chunk.to(config.device)
                    action_is_pad = action_is_pad.to(config.device)

                    actions_norm = (action_chunk - model.action_mean) / model.action_std
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
                        is_pad=action_is_pad
                    )

                    val_loss_val = loss_result[0] if isinstance(loss_result, tuple) else loss_result
                    loss_val.update(val_loss_val.item(), action_chunk.shape[0])

            epoch_logging_info.update(
                {
                    "validation/epoch": cur_epoch,
                    "validation/loss": loss_val.avg,
                }
            )
            Logger.log_info(f"[validation] epoch={cur_epoch}, val_loss={loss_val.avg:.6f}")

            avg_success = 0.0
            avg_rewards = 0.0

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
                torch.save(model.state_dict(), model_path)
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

            torch.save(model.state_dict(), os.path.join(local_run_output_dir, "last_model.pth"))

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()
