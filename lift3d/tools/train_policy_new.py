import functools
import json
import os
import pathlib
from typing import Tuple, Any, Optional

import hydra
import torch
from tqdm import tqdm
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored

import matplotlib.pyplot as plt  # <<< ADDED

from lift3d.envs import Evaluator
from lift3d.helpers.common import Logger, WandBLogger, set_seed
from lift3d.helpers.pytorch import AverageMeter, log_params_to_file


def save_loss_plot(train_losses, val_losses, val_epochs, output_dir):
    plt.figure(figsize=(8, 5))

    # Train loss: every epoch
    plt.plot(
        range(len(train_losses)),
        train_losses,
        label="Train Loss",
    )

    # Validation loss: sparse epochs
    if len(val_losses) > 0:
        plt.plot(
            val_epochs,
            val_losses,
            linestyle="-",
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def _unpack_batch(batch) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
    torch.Tensor,
    Optional[torch.Tensor],
    Any,
]:
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Batch must be tuple/list, got {type(batch)}")

    if len(batch) == 7:
        images, point_clouds, depth, robot_states, raw_states, actions, texts = batch
        is_pad = None
    elif len(batch) == 8:
        images, point_clouds, depth, robot_states, raw_states, actions, is_pad, texts = batch
    else:
        raise ValueError(f"Unexpected batch size {len(batch)}; expected 7 or 8.")

    return images, point_clouds, depth, robot_states, raw_states, actions, is_pad, texts


def log_training_info(config):
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with following config:'
    )
    Logger.log_info(f'Task: {colored(config.task_name, "green")}')
    Logger.log_info(f'Dataset directory: {colored(config.dataset_dir, "green")}')
    Logger.log_info(f'Image size: {colored(config.image_size, "green")}')
    Logger.log_info(
        f'WandB: Project {colored(config.wandb.project, "green")}; '
        f'Group {colored(config.wandb.group, "green")}; '
        f'Name {colored(config.wandb.name, "green")}; '
        f'Notes {colored(config.wandb.notes, "green")}; '
        f'Mode {colored(config.wandb.mode, "green")}'
    )
    Logger.print_seperator()


def init_wandb_logger(config):
    wandb_logger = WandBLogger(
        config=config.wandb,
        hyperparameters=OmegaConf.to_container(config, resolve=True),
    )
    wandb_logger.run.define_metric("train_interation/*", step_metric="iteration_step")
    wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
    wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")
    return wandb_logger


def get_dataloaders(config):
    train_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="train",
    )
    valid_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="validation",
    )

    DataLoaderConstuctor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )

    return (
        DataLoaderConstuctor(train_dataset),
        DataLoaderConstuctor(valid_dataset),
        train_dataset,
        valid_dataset,
    )

def set_model_normalization_stats(model, train_dataset, config):

    # Use a temporary loader (no shuffle) to scan data
    stats_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers
    )

    all_actions = []
    all_qpos = []
    all_geom_feats = []
    max_samples_for_stats = 10000 # Limit samples to avoid OOM
    collected_samples = 0

    for batch in tqdm(stats_loader, desc="Computing Stats"):
        _, _, depth, robot_states, _, actions, _, _ = _unpack_batch(batch)
        all_actions.append(actions)
        all_qpos.append(robot_states)
        geom = model.representation_encoder.depth_encoder(depth)   # [B, N, 8]
        all_geom_feats.append(geom)
        
        collected_samples += actions.shape[0]
        if collected_samples >= max_samples_for_stats:
            break
    
    # Concatenate and Flatten
    # actions: [N, T, D] -> [N*T, D]
    all_actions = torch.cat(all_actions, dim=0)
    flat_actions = all_actions.reshape(-1, all_actions.shape[-1])
    
    # qpos: [N, D] -> [N, D]
    all_qpos = torch.cat(all_qpos, dim=0)
    flat_qpos = all_qpos.reshape(-1, all_qpos.shape[-1])

    # depth geometry
    flat_depth_geom = torch.cat(all_geom_feats, dim=0).reshape(-1, model.representation_encoder._get_depth_emb_dim())
    
    # Compute Stats
    action_mean = flat_actions.mean(dim=0)
    action_std = flat_actions.std(dim=0)
    qpos_mean = flat_qpos.mean(dim=0)
    qpos_std = flat_qpos.std(dim=0)
    depth_geom_mean = flat_depth_geom.mean(dim=0)
    depth_geom_std  = flat_depth_geom.std(dim=0).clamp(min=1e-5)
    
    # Clip small std to prevent division by zero or explosion
    action_std = torch.clip(action_std, min=1e-5)
    qpos_std = torch.clip(qpos_std, min=1e-5)

    # --- INJECT STATS INTO MODEL ---
    # This requires the model class (ACTPolicy) to have a set_norm_stats method.
    if hasattr(model, "set_norm_stats"):
        model.set_norm_stats(
            action_mean.to(config.device), 
            action_std.to(config.device),
            depth_geom_mean.to(config.device),
            depth_geom_std.to(config.device),
            qpos_mean.to(config.device), 
            qpos_std.to(config.device)
        )
        Logger.log_info(colored(" [Auto-Norm] Statistics injected into model successfully!", "green"))
    else:
        Logger.log_warning(colored(" [Warning] Model missing 'set_norm_stats'. Auto-Norm skipped.", "yellow"))


@hydra.main(version_base=None, config_path="../config", config_name="train_representation_policy")
def main(config):
    log_training_info(config)
    set_seed(config.seed)

    # wandb_logger = init_wandb_logger(config)
    train_loader, valid_loader, train_dataset, _ = get_dataloaders(config)

    _, _, _, sample_robot_state, _, sample_action, _, _ = next(iter(train_loader))
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=sample_robot_state.size(-1),
        action_dim=sample_action.size(-1),
    ).to(config.device)
    set_model_normalization_stats(model, train_dataset, config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
    )

    scheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    ckpt_dir = local_run_output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    last_ckpt_path = ckpt_dir / "last.pth"
    save_every_epochs = config.train.save_every_epochs

    # <<< ADDED
    train_loss_history = []
    val_loss_history = []
    val_epochs = []
    plot_every_n_epochs = 1

    best_val_loss = float("inf")

    for cur_epoch in range(config.train.num_epochs):
        model.train()
        loss_train = AverageMeter()

        for cur_iter, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {cur_epoch}")):
            if cur_iter == 10:
                break
            images, _, depths, robot_states, _, actions, actions_is_pad, texts = _unpack_batch(batch)

            images = images.to(config.device)
            depths = depths.to(config.device)
            robot_states = robot_states.to(config.device)
            actions = actions.to(config.device)
            actions_is_pad = actions_is_pad.to(config.device)

            preds = model(images, depths, robot_states, texts, actions, actions_is_pad)

            actions_norm = (actions - model.action_mean) / model.action_std
            loss, loss_dict = call(config.benchmark.loss_func, preds, actions_norm, actions_is_pad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.update(loss.item())

        scheduler.step()

        train_loss_history.append(loss_train.avg)
        Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg:.6f}")

        # --- validation ---
        if (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0:
            model.eval()
            loss_val = AverageMeter()

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Valid epoch {cur_epoch}"):
                    images, _, depths, robot_states, _, actions, _, texts = _unpack_batch(batch)
                    images = images.to(config.device)
                    depths = depths.to(config.device)
                    robot_states = robot_states.to(config.device)
                    actions = actions.to(config.device)

                    preds = model(images, depths, robot_states, texts)
                    loss, loss_dict = call(config.benchmark.loss_func, preds, actions)
                    loss_val.update(loss.item(), actions.size(0))

            val_loss_history.append(loss_val.avg)
            val_epochs.append(cur_epoch)
            Logger.log_info(f"[val] epoch={cur_epoch}, loss={loss_val.avg:.6f}")

            if loss_val.avg < best_val_loss:
                best_val_loss = loss_val.avg
                torch.save(model.state_dict(), local_run_output_dir / "best_model.pth")

        # <<< RE-PLOT EVERY N EPOCHS
        if (cur_epoch + 1) % plot_every_n_epochs == 0:
            save_loss_plot(train_loss_history, val_loss_history, val_epochs, local_run_output_dir)

        # --- save last checkpoint ---
        torch.save(
            {
                "epoch": cur_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            last_ckpt_path,
        )

        # --- save periodic checkpoint ---
        if (cur_epoch + 1) % save_every_epochs == 0:
            torch.save(
                {
                    "epoch": cur_epoch,
                    "model": model.state_dict(),
                },
                ckpt_dir / f"epoch_{cur_epoch:04d}.pth",
            )

    # <<< FINAL SAVE
    save_loss_plot(train_loss_history, val_loss_history, local_run_output_dir)

    with open(local_run_output_dir / "loss_history.json", "w") as f:
        json.dump(
            {
                "train": train_loss_history,
                "validation": val_loss_history,
            },
            f,
            indent=4,
        )

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()


# import functools
# import json
# import os
# import pathlib
# from typing import Tuple, Any, Optional

# import hydra
# import torch
# from tqdm import tqdm
# from hydra.utils import call, instantiate
# from omegaconf import OmegaConf
# from termcolor import colored
# import matplotlib.pyplot as plt

# from lift3d.envs import Evaluator
# from lift3d.helpers.common import Logger, WandBLogger, set_seed
# from lift3d.helpers.pytorch import AverageMeter, log_params_to_file

# def save_loss_plot(
#     train_losses,
#     val_losses,
#     output_dir,
# ):

#     plt.figure(figsize=(8, 5))
#     plt.plot(train_losses, label="Train Loss")

#     if len(val_losses) > 0:
#         val_epochs = range(
#             len(train_losses) - len(val_losses),
#             len(train_losses),
#         )
#         plt.plot(val_epochs, val_losses, label="Validation Loss")

#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training & Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     out_path = output_dir / "loss_curve.png"
#     plt.savefig(out_path, dpi=150)
#     plt.close()

# def _unpack_batch(
#         batch
#     ) -> Tuple[
#         torch.Tensor,               # images
#         torch.Tensor,               # pointclouds
#         torch.Tensor,               # depth
#         torch.Tensor,               # robot_states
#         Any,                        # raw_states
#         torch.Tensor,               # actions
#         Optional[torch.Tensor],     # is_pad
#         Any                         # texts
#     ]:
#     """
#     Robust batch unpacking handling both 6-tuple (old) and 7-tuple (with is_pad).
#     Returns:
#       images, point_clouds, depth, robot_states, raw_states, actions, is_pad, texts
#     """
#     if not isinstance(batch, (tuple, list)):
#         raise ValueError(f"Batch must be tuple/list, got {type(batch)}")

#     if len(batch) == 7:
#         images, point_clouds, depth, robot_states, raw_states, actions, texts = batch
#         is_pad = None
#     elif len(batch) == 8:
#         images, point_clouds, depth, robot_states, raw_states, actions, is_pad, texts = batch
#     else:
#         raise ValueError(f"Unexpected batch size {len(batch)}; expected 7 or 8.")

#     return images, point_clouds, depth, robot_states, raw_states, actions, is_pad, texts

# def log_training_info(config):
#     #############################
#     # log important information #
#     #############################
#     Logger.log_info(
#         f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with following config:'
#     )
#     Logger.log_info(f'Task: {colored(config.task_name, "green")}')
#     Logger.log_info(f'Dataset directory: {colored(config.dataset_dir, "green")}')
#     Logger.log_info(f'Image size: {colored(config.image_size, "green")}')
#     Logger.log_info(
#         f'WandB: Project {colored(config.wandb.project, "green")}; '
#         f'Group {colored(config.wandb.group, "green")}; '
#         f'Name {colored(config.wandb.name, "green")}; '
#         f'Notes {colored(config.wandb.notes, "green")}; '
#         f'Mode {colored(config.wandb.mode, "green")}'
#     )
#     Logger.log_info(
#         f'Agent: {colored(config.agent.name, color="green")}\n{json.dumps(OmegaConf.to_container(config.agent, resolve=True), indent=4)}'
#     )
#     Logger.log_info(
#         f'Benchmark: {colored(config.benchmark.name, color="green")}\n{json.dumps(OmegaConf.to_container(config.benchmark, resolve=True), indent=4)}'
#     )
#     Logger.print_seperator()

# def init_wandb_logger(config):
#     wandb_logger = WandBLogger(
#         config=config.wandb,
#         hyperparameters=OmegaConf.to_container(config, resolve=True),
#     )
#     wandb_logger.run.define_metric("train_interation/*", step_metric="iteration_step")
#     wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
#     wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")

#     return wandb_logger

# def get_dataloaders(config):
#     ##########################
#     # datasets and evaluator #
#     ##########################
#     train_dataset = instantiate(
#         config=config.benchmark.dataset_instantiate_config,
#         data_dir=config.dataset_dir,
#         split="train",
#     )
#     valid_dataset = instantiate(
#         config=config.benchmark.dataset_instantiate_config,
#         data_dir=config.dataset_dir,
#         split="validation",
#     )

#     ###############
#     # dataloaders #
#     ###############
#     DataLoaderConstuctor = functools.partial(
#         torch.utils.data.DataLoader,
#         batch_size=config.dataloader.batch_size,
#         num_workers=config.dataloader.num_workers,
#         shuffle=config.dataloader.shuffle,
#         pin_memory=config.dataloader.pin_memory,
#         drop_last=config.dataloader.drop_last,
#     )
#     train_loader = DataLoaderConstuctor(train_dataset)
#     valid_loader = DataLoaderConstuctor(valid_dataset)

#     return train_loader, valid_loader, train_dataset, valid_dataset

# def set_model_normalization_stats(model, train_dataset, config):

#     # Use a temporary loader (no shuffle) to scan data
#     stats_loader = torch.utils.data.DataLoader(
#         train_dataset, 
#         batch_size=32, 
#         shuffle=False, 
#         num_workers=config.dataloader.num_workers
#     )

#     all_actions = []
#     all_qpos = []
#     all_geom_feats = []
#     max_samples_for_stats = 10000 # Limit samples to avoid OOM
#     collected_samples = 0

#     for batch in tqdm(stats_loader, desc="Computing Stats"):
#         _, _, depth, robot_states, _, actions, _, _ = _unpack_batch(batch)
#         all_actions.append(actions)
#         all_qpos.append(robot_states)
#         geom = model.representation_encoder.depth_encoder(depth)   # [B, N, 8]
#         all_geom_feats.append(geom)
        
#         collected_samples += actions.shape[0]
#         if collected_samples >= max_samples_for_stats:
#             break
    
#     # Concatenate and Flatten
#     # actions: [N, T, D] -> [N*T, D]
#     all_actions = torch.cat(all_actions, dim=0)
#     flat_actions = all_actions.reshape(-1, all_actions.shape[-1])
    
#     # qpos: [N, D] -> [N, D]
#     all_qpos = torch.cat(all_qpos, dim=0)
#     flat_qpos = all_qpos.reshape(-1, all_qpos.shape[-1])

#     # depth geometry
#     flat_depth_geom = torch.cat(all_geom_feats, dim=0).reshape(-1, model.representation_encoder._get_depth_emb_dim())
    
#     # Compute Stats
#     action_mean = flat_actions.mean(dim=0)
#     action_std = flat_actions.std(dim=0)
#     qpos_mean = flat_qpos.mean(dim=0)
#     qpos_std = flat_qpos.std(dim=0)
#     depth_geom_mean = flat_depth_geom.mean(dim=0)
#     depth_geom_std  = flat_depth_geom.std(dim=0).clamp(min=1e-5)
    
#     # Clip small std to prevent division by zero or explosion
#     action_std = torch.clip(action_std, min=1e-5)
#     qpos_std = torch.clip(qpos_std, min=1e-5)

#     # --- INJECT STATS INTO MODEL ---
#     # This requires the model class (ACTPolicy) to have a set_norm_stats method.
#     if hasattr(model, "set_norm_stats"):
#         model.set_norm_stats(
#             action_mean.to(config.device), 
#             action_std.to(config.device),
#             depth_geom_mean.to(config.device),
#             depth_geom_std.to(config.device),
#             qpos_mean.to(config.device), 
#             qpos_std.to(config.device)
#         )
#         Logger.log_info(colored(" [Auto-Norm] Statistics injected into model successfully!", "green"))
#     else:
#         Logger.log_warning(colored(" [Warning] Model missing 'set_norm_stats'. Auto-Norm skipped.", "yellow"))


# @hydra.main(version_base=None, config_path="../config", config_name="train_representation_policy")
# def main(config):
#     log_training_info(config)
#     set_seed(config.seed)
#     wandb_logger = init_wandb_logger(config)
#     train_loader, valid_loader, train_dataset, _ = get_dataloaders(config)

#     # instantiate model
#     _, _, _, sample_robot_state, _, sample_action, _, _ = next(iter(train_loader))
#     robot_state_dim = sample_robot_state.size(-1)
#     action_dim = sample_action.size(-1)
#     Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
#     Logger.log_info(f'Action dim: {colored(action_dim, "red")}')
#     model = instantiate(
#         config=config.agent.instantiate_config,
#         robot_state_dim=robot_state_dim,
#         action_dim=action_dim,
#     )
#     model = model.to(config.device)
#     set_model_normalization_stats(model, train_dataset, config)

#     # instantiate optimizer
#     optimizer: torch.optim.Optimizer = torch.optim.Adam(
#         params=model.parameters(),
#         lr=config.train.learning_rate,
#     )
#     local_run_output_dir = pathlib.Path(
#         hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
#     )
#     log_params_to_file(model, local_run_output_dir / "train_model_params.txt", True)
#     log_params_to_file(
#         model, local_run_output_dir / "train_model_params_freeze.txt", False
#     )
#     ckpt_dir = local_run_output_dir / "checkpoints"
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     last_ckpt_path = ckpt_dir / "last.pth"
#     best_ckpt_path = ckpt_dir / "best.pth"

#     # instantiate learning rate scheduler
#     scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
#         config=config.train.scheduler_instantiate_config,
#         optimizer=optimizer,
#     )

#     ############
#     # Training #
#     ############
#     # <<< changed: best metric depends on whether evaluator exists
#     best_success = -1.0
#     best_val_loss = float("inf")

#     max_success, max_rewards = 0.0, 0.0

#     for cur_epoch in range(config.train.num_epochs):
#         epoch_logging_info = {"epoch_step": cur_epoch + 1}

#         # --- train ---
#         model.train()
#         loss_train = AverageMeter()
#         for cur_iter, (
#             images,
#             point_clouds,
#             depths,
#             robot_states,
#             raw_states,
#             actions,
#             actions_is_pad,
#             texts,
#         ) in enumerate(tqdm(train_loader, desc=f"Train epoch: {cur_epoch}")):
#             iteration_info = {}

#             images = images.to(config.device)
#             point_clouds = point_clouds.to(config.device)
#             depths = depths.to(config.device)
#             robot_states = robot_states.to(config.device)
#             actions = actions.to(config.device, non_blocking=True)
#             actions_is_pad = actions_is_pad.to(config.device)

#             preds = model(images, depths, robot_states, texts, actions, actions_is_pad)

#             # we must compute the loss in a normalized space for training stability. our
#             # predicted actions are normalized, so we should normalize our ground truth
#             # actions as well
#             actions_norm = (actions - model.action_mean) / model.action_std
#             loss_result = call(config.benchmark.loss_func, preds, actions_norm, actions_is_pad)

#             if isinstance(loss_result, tuple):
#                 loss = loss_result[0]
#                 loss_dict = loss_result[1]
#                 for key, value in loss_dict.items():
#                     iteration_info[f"train_interation/{key}"] = value
#             else:
#                 loss = loss_result
                
#             optimizer.zero_grad()
#             loss.backward()

#             if config.train.clip_grad_value > 0.0:
#                 torch.nn.utils.clip_grad_value_(
#                     model.parameters(), config.train.clip_grad_value
#                 )

#             optimizer.step()
#             loss_train.update(loss.item())

#             iteration_info.update(
#                 {
#                     "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
#                     "train_interation/epoch": cur_epoch,
#                     "train_interation/loss": loss.item(),
#                     "train_interation/learning_rate": scheduler.get_last_lr()[0],
#                 }
#             )
#             wandb_logger.log(iteration_info)

#         scheduler.step()

#         epoch_logging_info["train_epoch/epoch_loss"] = loss_train.avg # type: ignore
#         Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg}")

#         # --- validation ---
#         periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
#             (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
#         )
#         last_epoch = (cur_epoch + 1) == config.train.num_epochs

#         if periodic_validation or last_epoch:
#             model.eval()

#             loss_val = AverageMeter()
#             for (
#                 images,
#                 point_clouds,
#                 robot_states,
#                 raw_states,
#                 actions,
#                 texts,
#             ) in tqdm(valid_loader, desc=f"Valid epoch: {cur_epoch}"):
#                 images = images.to(config.device)
#                 point_clouds = point_clouds.to(config.device)
#                 robot_states = robot_states.to(config.device)
#                 actions = actions.to(config.device, non_blocking=True)

#                 with torch.no_grad():
#                     preds = model(images, point_clouds, robot_states, texts)

#                 loss_result = call(config.benchmark.loss_func, preds, actions)
#                 if isinstance(loss_result, tuple):
#                     loss_val.update(loss_result[0].item(), actions.shape[0])
#                     loss_dict = loss_result[1]
#                     for key, value in loss_dict.items():
#                         epoch_logging_info[f"validation/{key}"] = value
#                 else:
#                     loss_val.update(loss_result.item(), actions.shape[0])

#             epoch_logging_info["validation/epoch"] = cur_epoch
#             epoch_logging_info["validation/loss"] = loss_val.avg # type: ignore

#             Logger.log_info(
#                 f"[validation] epoch={cur_epoch}, "
#                 f"validation_loss={loss_val.avg}, "
#             )

#             # save the best checkpoint
#             if loss_val.avg < best_val_loss:
#                 best_val_loss = loss_val.avg

#                 torch.save(
#                     {
#                         "epoch": cur_epoch,
#                         "model": model.state_dict(),
#                         "optimizer": optimizer.state_dict(),
#                         "scheduler": scheduler.state_dict(),
#                         "best_val_loss": best_val_loss,
#                         "config": OmegaConf.to_container(config, resolve=True),
#                     },
#                     best_ckpt_path,
#                 )

#                 Logger.log_info(
#                     colored(
#                         f"[checkpoint] Saved new best model to {best_ckpt_path}",
#                         "green",
#                     )
#                 )

#         # save the last checkpoint at every epoch
#         torch.save(
#             {
#                 "epoch": cur_epoch,
#                 "model": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "scheduler": scheduler.state_dict(),
#                 "config": OmegaConf.to_container(config, resolve=True),
#             },
#             last_ckpt_path,
#         )

#         # optionally, save every N epochs
#         if (cur_epoch + 1) % config.train.save_every_epochs == 0:
#             epoch_ckpt_path = ckpt_dir / f"epoch_{cur_epoch:04d}.pth"
#             torch.save(
#                 {
#                     "epoch": cur_epoch,
#                     "model": model.state_dict(),
#                 },
#                 epoch_ckpt_path,
#             )

#     Logger.log_ok("Training Finished!")


# if __name__ == "__main__":
#     main()