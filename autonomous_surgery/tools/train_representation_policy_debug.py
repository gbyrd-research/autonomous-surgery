import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from autonomous_surgery.dataset.pusht_dataset import PushTDataset
from autonomous_surgery.models.cnn.cnn_policy import CNNMLPPolicy
import autonomous_surgery.tools.utils.utils as train_utils

train_cfg = OmegaConf.create({
    "batch_size": 32,
    "num_workers": 8,
    "num_epochs": 100,
    "checkpoint_dir": "./checkpoints",
    "save_every": 10,
})

policy_cfg = OmegaConf.create({
    "lr": 1e-5,
    "lr_backbone": -1,
    "backbone": "resnet18",
    "num_queries": 1,
    "camera_names": ["3rd_person"],
    "state_dim": 2
})

wandb_cfg = OmegaConf.create({
    "enabled": True,
    "project": "SURPASS",
    "name": "train_representation_policy_debug",
    "mode": "online",
})

def save_checkpoint(
    path,
    epoch,
    policy,
    optimizer,
    train_history,
    test_history,
    train_cfg,
    policy_cfg,
    best_test_loss=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_history": train_history,
        "test_history": test_history,
        "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
        "policy_cfg": OmegaConf.to_container(policy_cfg, resolve=True),
        "best_test_loss": best_test_loss,
    }
    torch.save(checkpoint, path)

train_ds = PushTDataset(
    split="train",
    repo_id="lerobot/pusht",
    root="/home/grayson/surpass/autonomous-surgery/.hf/lerobot/pusht",
)
test_ds = PushTDataset(
    split="test",
    repo_id="lerobot/pusht",
    root="/home/grayson/surpass/autonomous-surgery/.hf/lerobot/pusht",
)

train_dataloader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=8,
)
test_dataloader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32,
    shuffle=True,
    num_workers=8,
)

policy = CNNMLPPolicy(policy_cfg)
criterion = nn.L1Loss()  # L1 loss only
optimizer = optim.Adam(policy.parameters(), lr=policy_cfg.lr)

num_epochs = train_cfg.num_epochs
policy.cuda()

checkpoint_dir = Path(train_cfg.checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

wandb_run = None
if wandb_cfg.enabled:
    wandb_run = wandb.init(
        project=wandb_cfg.project,
        name=wandb_cfg.name,
        mode=wandb_cfg.mode,
        config={
            "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
            "policy_cfg": OmegaConf.to_container(policy_cfg, resolve=True),
        },
    )

train_history = []
test_history = []
best_test_loss = float("inf")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch}")

    # training
    policy.train()
    optimizer.zero_grad()

    epoch_train_outputs = []
    for batch_idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        forward_dict = train_utils.forward_pass(data, policy)

        loss = forward_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        detached = train_utils.detach_dict(forward_dict)
        train_history.append(detached)
        epoch_train_outputs.append(detached)

    epoch_summary = train_utils.compute_dict_mean(epoch_train_outputs)
    epoch_train_loss = epoch_summary["loss"]

    summary_string = ""
    for k, v in epoch_summary.items():
        summary_string += f"{k}: {v.item():.5f} "
    print("Train Loss:")
    print(summary_string)

    # evaluation
    policy.eval()
    epoch_test_outputs = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            forward_dict = train_utils.forward_pass(data, policy)
            detached = train_utils.detach_dict(forward_dict)
            test_history.append(detached)
            epoch_test_outputs.append(detached)

    epoch_summary = train_utils.compute_dict_mean(epoch_test_outputs)
    epoch_test_loss = epoch_summary["loss"]

    summary_string = ""
    for k, v in epoch_summary.items():
        summary_string += f"{k}: {v.item():.5f} "
    print("Test Loss:")
    print(summary_string)

    # save latest checkpoint every epoch
    save_checkpoint(
        checkpoint_dir / "latest.pt",
        epoch=epoch,
        policy=policy,
        optimizer=optimizer,
        train_history=train_history,
        test_history=test_history,
        train_cfg=train_cfg,
        policy_cfg=policy_cfg,
        best_test_loss=best_test_loss,
    )

    if wandb_run is not None and epoch >= 3:
        log_data = {
            "epoch": epoch,
            "train/loss": epoch_train_loss.item(),
            "test/loss": epoch_test_loss.item(),
            "best_test_loss": best_test_loss,
        }
        wandb_run.log(log_data)

if wandb_run is not None:
    wandb_run.finish()