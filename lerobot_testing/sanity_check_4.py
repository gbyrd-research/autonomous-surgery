from pathlib import Path
import csv

import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from lerobot.policies.factory import make_pre_post_processors

GRASP = 1
PUSHT = 2

ACT = 1
DIFFUSION = 2


def append_log_row(log_path, row_dict):
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    dataset = PUSHT
    model = ACT

    if dataset == GRASP:
        output_directory = Path("outputs/train/grasp_0_hybrid_relative_actions")
        output_directory.mkdir(parents=True, exist_ok=True)

        import json
        with open("/home/grayson/surpass/autonomous-surgery/.hf_home/lerobot/surpass/grasp_only/meta/stats_original.json", "r") as file:
            temp = json.load(file)
            action_mean = torch.Tensor(temp["action_hybrid_relative"]["mean"])
            action_std = torch.Tensor(temp["action_hybrid_relative"]["std"])

        device = torch.device("cuda")

        ckpt_dir = Path("/home/grayson/surpass/autonomous-surgery/lerobot_testing/outputs/train/grasp_0_hybrid_relative_actions")
        # ckpt_dir = None

        dataset_metadata = LeRobotDatasetMetadata("surpass/grasp_only")
        features = dataset_to_policy_features(dataset_metadata.features)
        input_feature_keys = {
            "observation.images.endoscope.left",
            "observation.images.wrist.left",
            "observation.images.wrist.right",
            "observation.state",
        }
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features and key in input_feature_keys}

        cfg = ACTConfig(input_features=input_features, output_features=output_features, )

        if ckpt_dir is not None:
            policy = ACTPolicy.from_pretrained(ckpt_dir)
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=ckpt_dir
            )
        else:
            policy = ACTPolicy(cfg)
            preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

        policy.train()
        policy.to(device)

        delta_timestamps = {
            "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
            "action_hybrid_relative": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        }

        train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
        val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]
        train_episodes = list(range(train_ep_start, train_ep_end))
        val_episodes = list(range(val_ep_start, val_ep_end))
        dataset_repo_id = "surpass/grasp_only"
        dataset_root = "/home/grayson/surpass/autonomous-surgery/.hf_home/lerobot/surpass/grasp_only"

        train_dataset = LeRobotDataset(
            repo_id=dataset_repo_id,
            root=dataset_root,
            delta_timestamps=delta_timestamps,
            episodes=train_episodes,
        )
        val_dataset = LeRobotDataset(
            repo_id=dataset_repo_id,
            root=dataset_root,
            delta_timestamps=delta_timestamps,
            episodes=val_episodes,
        )

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=8,
            batch_size=8,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=8,
            batch_size=8,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )

    elif dataset == PUSHT:

        device = torch.device("cuda")

        ckpt_dir = None

        dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

        if model == ACT:

            output_directory = Path("outputs/train/pusht/act")
            output_directory.mkdir(parents=True, exist_ok=True)

            cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size=10, vision_backbone="resnet152", n_action_steps=10, pretrained_backbone_weights='ResNet152_Weights.IMAGENET1K_V1')

            if ckpt_dir is not None:
                policy = ACTPolicy.from_pretrained(ckpt_dir)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy.config,
                    pretrained_path=ckpt_dir
                )
            else:
                policy = ACTPolicy(cfg)
                preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

            delta_timestamps = {
                "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
            }

        elif model == DIFFUSION:

            output_directory = Path("outputs/train/pusht/diffusion")
            output_directory.mkdir(parents=True, exist_ok=True)

            # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
            # we'll just use the defaults and so no arguments other than input/output features need to be passed.
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

            # We can now instantiate our policy with this config and the dataset stats.
            if ckpt_dir is not None:
                policy = DiffusionPolicy.from_pretrained(ckpt_dir)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy.config,
                    pretrained_path=ckpt_dir
                )
            else:
                policy = DiffusionPolicy(cfg)
                preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

            # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
            delta_timestamps = {
                # Load the previous image and state at -0.1 seconds before current frame,
                # then load current image and state corresponding to 0.0 second.
                "observation.image": [-0.1, 0.0],
                "observation.state": [-0.1, 0.0],
                # Load the previous action (-0.1), the next action to be executed (0.0),
                # and 14 future actions with a 0.1 seconds spacing. All these actions will be
                # used to supervise the policy.
                "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            }

        policy.train()
        policy.to(device)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
            
        

        train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
        val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]
        train_episodes = list(range(train_ep_start, train_ep_end))
        val_episodes = list(range(val_ep_start, val_ep_end))

        train_dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps, episodes=train_episodes)
        val_dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps, episodes=val_episodes)

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=4,
            batch_size=64,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=64,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )
    else:
        raise NotImplementedError()

    log_path = output_directory / "training_log.csv"

    for epoch in range(100):
        train_loss_total = 0.0
        train_loss_l1 = 0.0
        train_loss_kl = 0.0
        train_dist = None

        policy.train()
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            if dataset == GRASP:
                batch["action"] = batch["action_hybrid_relative"]
                del batch["action_hybrid_relative"]

            raw_action = batch["action"]
            batch = preprocessor(batch)

            policy.train()
            loss, loss_dict = policy.forward(batch)
            action = policy.predict_action_chunk(batch)
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

            for batch in tqdm(val_dataloader, total=len(val_dataloader)):
                if dataset == GRASP:
                    batch["action"] = batch["action_hybrid_relative"]
                    del batch["action_hybrid_relative"]

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

        # Average train distances
        train_dist = train_dist / len(train_dataloader)
        train_dist_first_timestamp = train_dist[:, 0, :].mean(dim=0)   # shape [2]
        train_dist_total = train_dist.mean(dim=(0, 1))                 # shape [2]

        # Average val distances
        val_dist = val_dist / len(val_dataloader)
        val_dist_first_timestamp = val_dist[:, 0, :].mean(dim=0)       # shape [2]
        val_dist_total = val_dist.mean(dim=(0, 1))                     # shape [2]

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

        append_log_row(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "train_l1": train_l1_mean,
                "train_kl": train_kl_mean,
                "train_dist_first_timestamp_0": float(train_dist_first_timestamp[0]),
                "train_dist_first_timestamp_1": float(train_dist_first_timestamp[1]),
                "train_dist_total_0": float(train_dist_total[0]),
                "train_dist_total_1": float(train_dist_total[1]),
                "val_loss": val_loss_mean,
                "val_l1": val_l1_mean,
                "val_kl": val_kl_mean,
                "val_dist_first_timestamp_0": float(val_dist_first_timestamp[0]),
                "val_dist_first_timestamp_1": float(val_dist_first_timestamp[1]),
                "val_dist_total_0": float(val_dist_total[0]),
                "val_dist_total_1": float(val_dist_total[1]),
            },
        )

        policy.save_pretrained(output_directory)
        preprocessor.save_pretrained(output_directory)
        postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()