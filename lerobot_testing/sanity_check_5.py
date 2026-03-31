from pathlib import Path
import csv

import hydra
import torch
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


def append_log_row(log_path: Path, row_dict: dict):
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


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


def build_dataset(
    dataset_cfg: DictConfig,
    delta_timestamps: dict,
    episodes: list[int],
    root: Path | None,
):
    dataset_kwargs = {
        "repo_id": dataset_cfg.repo_id,
        "delta_timestamps": delta_timestamps,
        "episodes": episodes,
    }
    if root is not None:
        dataset_kwargs["root"] = str(root)

    return LeRobotDataset(**dataset_kwargs)


def add_vector_metrics(row: dict, prefix: str, values: torch.Tensor):
    for idx, value in enumerate(values.tolist()):
        row[f"{prefix}_{idx}"] = float(value)


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

    policy.train()
    policy.to(device)

    delta_timestamps = build_delta_timestamps(dataset_cfg, dataset_metadata, policy.config)

    train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
    val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]
    train_episodes = list(range(train_ep_start, train_ep_end))
    val_episodes = list(range(val_ep_start, val_ep_end))

    train_dataset = build_dataset(dataset_cfg, delta_timestamps, train_episodes, dataset_root)
    val_dataset = build_dataset(dataset_cfg, delta_timestamps, val_episodes, dataset_root)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.training.learning_rate)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory and device.type != "cpu",
        drop_last=config.dataloader.drop_last,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        pin_memory=config.dataloader.pin_memory and device.type != "cpu",
        drop_last=config.dataloader.drop_last,
    )

    log_path = output_directory / "training_log.csv"

    for epoch in range(config.training.epochs):
        train_loss_total = 0.0
        train_loss_l1 = 0.0
        train_loss_kl = 0.0
        train_dist = None

        policy.train()
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"train {epoch}"):
            dataset_action_key = dataset_cfg.dataset_action_key
            if dataset_action_key != "action":
                batch["action"] = batch[dataset_action_key]
                del batch[dataset_action_key]

            raw_action = batch["action"]
            proc_batch = preprocessor(batch)

            policy.train()
            loss, loss_dict = policy.forward(proc_batch)
            action = policy.predict_action_chunk(proc_batch)
            action_pred = postprocessor(action)

            if train_dist is None:
                train_dist = torch.zeros_like(raw_action).cpu()

            train_dist += (action_pred - raw_action).abs().cpu()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_total += loss.detach().item()
            train_loss_l1 += float(loss_dict["l1_loss"])
            train_loss_kl += float(loss_dict["kld_loss"])

        with torch.no_grad():
            val_loss_total = 0.0
            val_loss_l1 = 0.0
            val_loss_kl = 0.0
            val_dist = None

            for batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"val {epoch}"):
                dataset_action_key = dataset_cfg.dataset_action_key
                if dataset_action_key != "action":
                    batch["action"] = batch[dataset_action_key]
                    del batch[dataset_action_key]

                raw_action = batch["action"]
                proc_batch = preprocessor(batch)

                policy.train()
                loss, val_loss_dict = policy.forward(proc_batch)

                policy.eval()
                action = policy.predict_action_chunk(proc_batch)
                action = postprocessor(action)

                if val_dist is None:
                    val_dist = torch.zeros_like(raw_action).cpu()

                val_loss_total += loss.item()
                val_loss_l1 += float(val_loss_dict["l1_loss"])
                val_loss_kl += float(val_loss_dict["kld_loss"])
                val_dist += (action - raw_action).abs().cpu()

        train_dist = train_dist / len(train_dataloader)
        train_dist_first_timestamp = train_dist[:, 0, :].mean(dim=0)
        train_dist_total = train_dist.mean(dim=(0, 1))

        val_dist = val_dist / len(val_dataloader)
        val_dist_first_timestamp = val_dist[:, 0, :].mean(dim=0)
        val_dist_total = val_dist.mean(dim=(0, 1))

        train_loss_mean = train_loss_total / len(train_dataloader)
        train_l1_mean = train_loss_l1 / len(train_dataloader)
        train_kl_mean = train_loss_kl / len(train_dataloader)

        val_loss_mean = val_loss_total / len(val_dataloader)
        val_l1_mean = val_loss_l1 / len(val_dataloader)
        val_kl_mean = val_loss_kl / len(val_dataloader)

        print(
            f"Epoch: {epoch} - "
            f"Train Loss: {train_loss_mean} L1: {train_l1_mean} KL: {train_kl_mean} - "
            f"Val Loss: {val_loss_mean} L1: {val_l1_mean} KL: {val_kl_mean}"
        )
        print(f"Train Dist First Timestamp: {train_dist_first_timestamp}")
        print(f"Train Dist Total: {train_dist_total}")
        print(f"Val Dist First Timestamp: {val_dist_first_timestamp}")
        print(f"Val Dist Total: {val_dist_total}")

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss_mean,
            "train_l1": train_l1_mean,
            "train_kl": train_kl_mean,
            "val_loss": val_loss_mean,
            "val_l1": val_l1_mean,
            "val_kl": val_kl_mean,
        }
        add_vector_metrics(log_row, "train_dist_first_timestamp", train_dist_first_timestamp)
        add_vector_metrics(log_row, "train_dist_total", train_dist_total)
        add_vector_metrics(log_row, "val_dist_first_timestamp", val_dist_first_timestamp)
        add_vector_metrics(log_row, "val_dist_total", val_dist_total)
        append_log_row(log_path, log_row)

        policy.save_pretrained(output_directory)
        preprocessor.save_pretrained(output_directory)
        postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
