from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def resolve_path(path_str: str | None, original_cwd: Path) -> Path | None:
    if not path_str:
        return None

    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = original_cwd / path
    return path


def get_input_output_features(dataset_cfg: DictConfig, dataset_metadata: LeRobotDatasetMetadata):
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    allowed_input_keys = dataset_cfg.get("input_feature_keys")
    if allowed_input_keys:
        allowed_input_keys = set(allowed_input_keys)
        input_features = {
            key: ft
            for key, ft in features.items()
            if key not in output_features and key in allowed_input_keys
        }
    else:
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

    return input_features, output_features


def build_delta_timestamps(dataset_cfg: DictConfig, dataset_metadata: LeRobotDatasetMetadata, act_cfg: ACTConfig):
    delta_timestamps = {}
    for action_key in dataset_cfg.delta_timestamp_action_keys:
        delta_timestamps[action_key] = [i / dataset_metadata.fps for i in act_cfg.action_delta_indices]
    return delta_timestamps


def build_dataset(dataset_cfg, delta_timestamps, episodes, root):
    dataset_kwargs = {
        "repo_id": dataset_cfg.repo_id,
        "delta_timestamps": delta_timestamps,
        "episodes": episodes,
    }
    if root is not None:
        dataset_kwargs["root"] = str(root)
    return LeRobotDataset(**dataset_kwargs)


def tensor_to_metric_dict(prefix: str, values: torch.Tensor) -> dict:
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(values.tolist())}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    original_cwd = Path(get_original_cwd())
    output_directory = Path(HydraConfig.get().runtime.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(config))

    device = torch.device(config.device)
    dataset_cfg = config.dataset

    dataset_root = resolve_path(dataset_cfg.root, original_cwd)
    checkpoint_dir = resolve_path(config.training.resume_from_checkpoint, original_cwd)

    dataset_metadata = LeRobotDatasetMetadata(dataset_cfg.repo_id)
    input_features, output_features = get_input_output_features(dataset_cfg, dataset_metadata)

    act_kwargs = OmegaConf.to_container(config.act, resolve=True)
    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        **act_kwargs,
    )

    if checkpoint_dir is not None:
        policy = ACTPolicy.from_pretrained(checkpoint_dir)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint_dir,
        )
    else:
        policy = ACTPolicy(policy_cfg)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg,
            dataset_stats=dataset_metadata.stats,
        )

    # Freeze vision backbone
    if config.training.freeze_backbone:
        for name, param in policy.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

    policy.to(device)

    # ---------------------------
    # W&B init
    # ---------------------------
    wandb.init(
        project=config.get("wandb", {}).get("project", "lerobot-act"),
        entity=config.get("wandb", {}).get("entity", None),
        name=config.get("wandb", {}).get("name", None),
        config=OmegaConf.to_container(config, resolve=True),
        dir=str(output_directory),
    )

    wandb.watch(policy, log="gradients", log_freq=100)

    delta_timestamps = build_delta_timestamps(dataset_cfg, dataset_metadata, policy.config)

    train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
    val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]

    train_dataset = build_dataset(dataset_cfg, delta_timestamps, list(range(train_ep_start, train_ep_end)), dataset_root)
    val_dataset = build_dataset(dataset_cfg, delta_timestamps, list(range(val_ep_start, val_ep_end)), dataset_root)

    # Build optimizer only from trainable params (if backbone is frozen, do not
    # include these parameters in the optimizer)
    optimizer = torch.optim.Adam(
        (p for p in policy.parameters() if p.requires_grad),
        lr=config.training.learning_rate,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory and device.type != "cpu",
        drop_last=config.dataloader.drop_last,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        pin_memory=config.dataloader.pin_memory and device.type != "cpu",
        drop_last=config.dataloader.drop_last,
    )

    for epoch in range(config.training.epochs):
        policy.train()

        train_loss_total = 0.0
        train_l1_total = 0.0
        train_kl_total = 0.0
        train_dist = None

        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            if dataset_cfg.dataset_action_key != "action":
                batch["action"] = batch.pop(dataset_cfg.dataset_action_key)

            raw_action = batch["action"]
            proc_batch = preprocessor(batch)

            policy.train()
            loss, loss_dict = policy.forward(proc_batch)
            action = postprocessor(policy.predict_action_chunk(proc_batch))

            if train_dist is None:
                train_dist = torch.zeros_like(raw_action).cpu()

            train_dist += (action - raw_action).abs().cpu()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_total += loss.item()
            train_l1_total += float(loss_dict["l1_loss"])
            train_kl_total += float(loss_dict["kld_loss"])

        # validation
        policy.eval()
        val_loss_total = 0.0
        val_l1_total = 0.0
        val_kl_total = 0.0
        val_dist = None

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch}"):
                if dataset_cfg.dataset_action_key != "action":
                    batch["action"] = batch.pop(dataset_cfg.dataset_action_key)

                raw_action = batch["action"]
                proc_batch = preprocessor(batch)

                policy.train()
                loss, loss_dict = policy.forward(proc_batch)
                action = postprocessor(policy.predict_action_chunk(proc_batch))

                if val_dist is None:
                    val_dist = torch.zeros_like(raw_action).cpu()

                val_loss_total += loss.item()
                val_l1_total += float(loss_dict["l1_loss"])
                val_kl_total += float(loss_dict["kld_loss"])
                val_dist += (action - raw_action).abs().cpu()

        # averages
        train_loss = train_loss_total / len(train_loader)
        train_l1 = train_l1_total / len(train_loader)
        train_kl = train_kl_total / len(train_loader)

        val_loss = val_loss_total / len(val_loader)
        val_l1 = val_l1_total / len(val_loader)
        val_kl = val_kl_total / len(val_loader)

        train_dist = (train_dist / len(train_loader))
        val_dist = (val_dist / len(val_loader))

        train_dist_first = train_dist[:, 0, :].mean(dim=0)
        val_dist_first = val_dist[:, 0, :].mean(dim=0)

        # ---------------------------
        # W&B logging
        # ---------------------------
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/l1": train_l1,
            "train/kl": train_kl,
            "val/loss": val_loss,
            "val/l1": val_l1,
            "val/kl": val_kl,
        }

        metrics.update(tensor_to_metric_dict("train/dist_first", train_dist_first))
        metrics.update(tensor_to_metric_dict("val/dist_first", val_dist_first))

        wandb.log(metrics, step=epoch)

        print(metrics)

        policy.save_pretrained(output_directory)

    wandb.finish()


if __name__ == "__main__":
    main()