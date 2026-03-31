import torch
from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

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

# Path where you saved
ckpt_dir = Path("/home/grayson/surpass/autonomous-surgery/lerobot_testing/outputs/train/grasp_v0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load policy (this restores config automatically)
policy = ACTPolicy.from_pretrained(ckpt_dir)
policy.to(device)
policy.eval()

# Load preprocessors
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=ckpt_dir
)

# Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
# which can differ for inputs, outputs and rewards (if there are some).
dataset_metadata = LeRobotDatasetMetadata("surpass/grasp_only")
features = dataset_to_policy_features(dataset_metadata.features)
input_feature_keys = {
    "observation.images.endoscope.left", 
    "observation.images.wrist.left", 
    "observation.images.wrist.right", 
    "observation.state"
}
output_feature_keys = {
    "action_hybrid_relative"
}
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features and key in input_feature_keys}
cfg = ACTConfig(input_features=input_features, output_features=output_features)
delta_timestamps = {
    ACTION: [t / dataset_metadata.fps for t in range(cfg.n_action_steps)],
    "action_hybrid_relative": [t / dataset_metadata.fps for t in range(cfg.n_action_steps)]
}
train_ep_start, train_ep_end = [int(i) for i in dataset_metadata.info["splits"]["train"].split(":")]
val_ep_start, val_ep_end = [int(i) for i in dataset_metadata.info["splits"]["val"].split(":")]
train_episodes = list(range(train_ep_start, train_ep_end))
val_episodes = list(range(val_ep_start, val_ep_end))
dataset_repo_id = "surpass/grasp_only"
dataset_root = "/home/grayson/surpass/autonomous-surgery/.hf_home/lerobot/surpass/grasp_only"
train_dataset = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root, delta_timestamps=delta_timestamps, episodes=train_episodes)
val_dataset = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root, delta_timestamps=delta_timestamps, episodes=val_episodes)

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

# for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
#     # our actual "action" feature is under the "action_hybrid_relative" key, thus
#     # we will reset this here
#     batch["action"] = batch["action_hybrid_relative"]
#     del batch["action_hybrid_relative"]
#     batch = preprocessor(batch)
#     actions = policy.predict_action_chunk(batch)
#     pass

for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    # our actual "action" feature is under the "action_hybrid_relative" key, thus
    # we will reset this here
    gt_action_before_processing = batch["action"]
    gt_action_hyb_rel_before_processing = batch["action_hybrid_relative"]
    batch["action"] = batch["action_hybrid_relative"]
    batch_processed = preprocessor(batch)
    actions = policy.predict_action_chunk(batch_processed)
    actions_post = postprocessor(actions)
    pass

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

def get_desired_features_from_batch(batch):
    pass

def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    log_freq = 1

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
    output_feature_keys = {
        "action_hybrid_relative"
    }
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features and key in input_feature_keys}

    # HACK: It seems like we can't use the action_hybrid_relative key as input when
    # creating the ACTPolicy model because Lerobot hardcodes the action key to be "action".
    # since our action_hybrid_relative and action features are the exact same shape and type,
    # we can just keep the action keyword and later on rename the key in the batch from "action_hybrid_relative" to 
    # "action"
    del output_features["action_hybrid_relative"]

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)

    # We can now instantiate our policy with this config and the dataset stats.
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        ACTION: [t / dataset_metadata.fps for t in range(cfg.n_action_steps)],
        "action_hybrid_relative": [t / dataset_metadata.fps for t in range(cfg.n_action_steps)]
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
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # our actual "action" feature is under the "action_hybrid_relative" key, thus
            # we will reset this here
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

        
        test_loss_total = test_loss_l1 = test_loss_kl = 0
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # our actual "action" feature is under the "action_hybrid_relative" key, thus
            # we will reset this here
            batch["action"] = batch["action_hybrid_relative"]
            del batch["action_hybrid_relative"]
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            test_loss_total += loss.detach().item()
            test_loss_l1 += loss_dict["l1_loss"]
            test_loss_kl += loss_dict["kld_loss"]


        print(f"Epoch: {epoch} - Train Loss: {train_loss_total / len(train_dataloader)} | {train_loss_l1 / len(train_dataloader)} | {train_loss_kl / len(train_dataloader)} - Test Loss: {test_loss_total / len(test_dataloader)} | {test_loss_l1 / len(test_dataloader)} | {test_loss_kl / len(test_dataloader)}")

        # Save a policy checkpoint.
        if epoch % 10 == 0:
            policy.save_pretrained(output_directory)
            preprocessor.save_pretrained(output_directory)
            postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()







# import torch

# from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
# from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
# from lerobot.policies.act.modeling_act import ACTPolicy
# from lerobot.policies.factory import make_pre_post_processors
# from lerobot.policies.utils import build_inference_frame, make_robot_action
# from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

# MAX_EPISODES = 5
# MAX_STEPS_PER_EPISODE = 20


# def main():
#     device = torch.device("mps")  # or "cuda" or "cpu"
#     model_id = "<user>/robot_learning_tutorial_act"
#     model = ACTPolicy.from_pretrained(model_id)

#     dataset_id = "lerobot/svla_so101_pickplace"
#     # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
#     dataset_metadata = LeRobotDatasetMetadata(dataset_id)
#     preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

#     # # find ports using lerobot-find-port
#     follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

#     # # the robot ids are used the load the right calibration files
#     follower_id = ...  # something like "follower_so100"

#     # Robot and environment configuration
#     # Camera keys must match the name and resolutions of the ones used for training!
#     # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
#     camera_config = {
#         "side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
#         "up": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
#     }

#     robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
#     robot = SO100Follower(robot_cfg)
#     robot.connect()

#     for _ in range(MAX_EPISODES):
#         for _ in range(MAX_STEPS_PER_EPISODE):
#             obs = robot.get_observation()
#             obs_frame = build_inference_frame(
#                 observation=obs, ds_features=dataset_metadata.features, device=device
#             )

#             obs = preprocess(obs_frame)

#             action = model.select_action(obs)
#             action = postprocess(action)

#             action = make_robot_action(action, dataset_metadata.features)

#             robot.send_action(action)

#         print("Episode finished! Starting new episode...")


# if __name__ == "__main__":
#     main()