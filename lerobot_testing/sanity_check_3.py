# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to train Diffusion Policy on the PushT environment."""

from pathlib import Path

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


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/grasp_0_abs_actions")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
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

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        # "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        # "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    # delta_timestamps = {
    #     # Load the previous image and state at -0.1 seconds before current frame,
    #     # then load current image and state corresponding to 0.0 second.
    #     "observation.image": [-0.1, 0.0],
    #     "observation.state": [-0.1, 0.0],
    #     # Load the previous action (-0.1), the next action to be executed (0.0),
    #     # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    #     # used to supervise the policy.
    #     "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    # }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
    val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]
    train_episodes = list(range(train_ep_start, train_ep_end))
    val_episodes = list(range(val_ep_start, val_ep_end))
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
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=8,
        batch_size=8,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    for epoch in range(100):
        train_loss_total = train_loss_l1 = train_loss_kl = 0
        policy.train()
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_total += loss.detach().item()
        
        with torch.no_grad():
            # policy.eval()
            total_val_loss = 0

            for batch in tqdm(val_dataloader, total=len(val_dataloader)):
                raw_action = batch["action"].to(device)
                proc_batch = preprocessor(batch)
                loss, _ = policy.forward(proc_batch)
                total_val_loss += loss

        print(f"Epoch: {epoch} - Train Loss: {train_loss_total / len(train_dataloader)} - Test Loss: {total_val_loss / len(val_dataloader)}")


    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()