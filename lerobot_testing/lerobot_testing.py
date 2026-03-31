"""This script demonstrates how to train Diffusion Policy on the PushT environment."""

from pathlib import Path

import torch

from tqdm import tqdm
from lerobot.utils.constants import ACTION

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

GRASP = "grasp_v0"
PUSHT = "pusht"

def main():

    dataset_type = PUSHT

    # Select your device
    device = torch.device("cuda")

    # Create a directory to store the training checkpoint.
    if dataset_type == GRASP:
        output_directory = Path("outputs/train/grasp_v0")
        output_directory.mkdir(parents=True, exist_ok=True)
        dataset_metadata = LeRobotDatasetMetadata("surpass/grasp_only")
        features = dataset_to_policy_features(dataset_metadata.features)
        input_feature_keys = {
            "observation.images.endoscope.left", 
            "observation.images.wrist.left", 
            "observation.images.wrist.right", 
            "observation.state"
        }
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features and key in input_feature_keys}
    elif dataset_type == PUSHT:
        output_directory = Path("outputs/train/pusht")
        output_directory.mkdir(parents=True, exist_ok=True)
        dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # create model
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)

    # We can now instantiate our policy with this config and the dataset stats.
    policy.train()
    policy.to(device)

    # HACK: We must do this to ensure that action_hybrid_relative is included in the
    # preprocessed batch
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    if dataset_type == GRASP:
        delta_timestamps = {
            ACTION: [t / dataset_metadata.fps for t in range(cfg.n_action_steps)],
            "action_hybrid_relative": [t / dataset_metadata.fps for t in range(cfg.n_action_steps)]
        }
    elif dataset_type == PUSHT:
        delta_timestamps = {
            ACTION: [t / dataset_metadata.fps for t in range(cfg.n_action_steps)]
        }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
    val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]
    train_episodes = list(range(train_ep_start, train_ep_end))
    val_episodes = list(range(val_ep_start, val_ep_end))
    if dataset_type == PUSHT:
        dataset_repo_id = "lerobot/pusht"
        dataset_root = "/home/grayson/surpass/autonomous-surgery/.hf_home/lerobot/pusht"
    elif dataset_type == GRASP:
        dataset_repo_id = "surpass/grasp_only"
        dataset_root = "/home/grayson/surpass/autonomous-surgery/.hf_home/lerobot/surpass/grasp_only"
    train_dataset = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root, delta_timestamps=delta_timestamps, episodes=train_episodes)
    val_dataset = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root, delta_timestamps=delta_timestamps, episodes=val_episodes)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=8,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=8,
        batch_size=8,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss_total = train_loss_l1 = train_loss_kl = 0
        policy.train()
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # our actual "action" feature is under the "action_hybrid_relative" key, thus
            # we will reset this here
            if dataset_type == GRASP:
                batch["action"] = batch["action_hybrid_relative"]
                del batch["action_hybrid_relative"]
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_total += loss.detach().item()
            train_loss_l1 += loss_dict["l1_loss"]
            train_loss_kl += loss_dict["kld_loss"]
        
        avg_dist = torch.zeros_like(batch["action"]).cpu()
        policy.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                # our actual "action" feature is under the "action_hybrid_relative" key, thus
                # we will reset this here
                if dataset_type == GRASP:
                    batch["action"] = batch["action_hybrid_relative"]
                    del batch["action_hybrid_relative"]
                initial_raw = batch["action"]
                batch = preprocessor(batch)
                action = policy.predict_action_chunk(batch)
                action_post = postprocessor(action)
                avg_dist += (action_post - initial_raw).abs()
            avg_dist = (avg_dist / len(test_dataloader)).mean()

        print(f"Epoch: {epoch} - Train Loss: {train_loss_total / len(train_dataloader)} | {train_loss_l1 / len(train_dataloader)} | {train_loss_kl / len(train_dataloader)} - Test Dist: {avg_dist}")

        # Save a policy checkpoint.
        if epoch % 10 == 0:
            policy.save_pretrained(output_directory)
            preprocessor.save_pretrained(output_directory)
            postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
