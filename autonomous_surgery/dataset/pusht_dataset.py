from __future__ import annotations

from pathlib import Path
import random
import os
import torch

from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

random.seed(42)

hf_home = os.environ.get("HF_HOME")


def parse_split_range(s: str) -> list[int]:
    start, end = map(int, s.split(":"))
    return list(range(start, end))


class PushTDataset(Dataset):
    """
    Sequence-style dataset wrapper for LeRobot PushT, formatted to match
    multi-camera ACT-style datasets.

    Each sample corresponds to a single starting timestep and returns a
    fixed-length future window.

    Returns:
        (
            image,                  # [C, H, W] current top-down image
            dummy_wrist_l,          # [3, 480, 640] placeholder (no wrist cam in PushT)
            dummy_wrist_r,          # [3, 480, 640] placeholder (no wrist cam in PushT)
            initial_state,          # [state_dim] current state at the first timestep
            relative_action_chunk,  # [T, action_dim] actions relative to state at each timestep
            action_is_pad,          # [T] boolean mask
            instruction_text        # str placeholder
        )
    """

    def __init__(
        self,
        repo_id: str = "lerobot/pusht",
        root: str | Path = Path(str(hf_home)).joinpath("lerobot").joinpath("pusht"),
        tolerance_s: float = 1e-4,
        split: str = "train",
        chunk_size: int = 16,
    ):
        super().__init__()

        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=repo_id,
            root=root,
        )

        info = dataset_metadata.info
        available_splits = set(info["splits"].keys())
        assert split in available_splits, (
            f"Invalid split name, {split}. Must be one of {available_splits}."
        )

        episodes = parse_split_range(info["splits"][split])
        fps = dataset_metadata.fps
        timestamps = [i / fps for i in range(chunk_size)]

        # Load the FULL dataset to avoid LeRobot's episode-index bug on nonzero episode subsets
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            tolerance_s=tolerance_s,
            delta_timestamps={
                "action": timestamps,
                "observation.state": timestamps,
            },
        )

        self.split = split
        self.episodes = episodes

        # Build frame bounds for this split using global episode indices
        episode_from = self.dataset.episode_data_index["from"]
        episode_to = self.dataset.episode_data_index["to"]

        self.start_idx = int(episode_from[episodes[0]].item())
        self.end_idx = int(episode_to[episodes[-1]].item())

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        global_idx = idx + self.start_idx
        sample = self.dataset[global_idx]

        image = sample["observation.image"]
        state_chunk = sample["observation.state"]   # [T, state_dim]
        action_chunk = sample["action"]             # [T, action_dim]
        action_is_pad = sample.get(
            "action_is_pad",
            torch.zeros(action_chunk.shape[0], dtype=torch.bool),
        )

        relative_action_chunk = action_chunk - state_chunk
        initial_state = state_chunk[0]

        dummy_wrist_r = torch.zeros((3, 480, 640))
        dummy_wrist_l = torch.zeros((3, 480, 640))

        instruction_text = "placeholder"

        return (
            image,
            dummy_wrist_l,
            dummy_wrist_r,
            initial_state,
            relative_action_chunk,
            action_is_pad,
            instruction_text,
        )


if __name__ == "__main__":
    pusht_ds = PushTDataset(split="test")
    print("PushT length:", len(pusht_ds))

    sample = pusht_ds[0]
    print("image shape:", sample[0].shape if hasattr(sample[0], "shape") else type(sample[0]))
    print("left wrist shape:", sample[1].shape if hasattr(sample[1], "shape") else type(sample[1]))
    print("right wrist shape:", sample[2].shape if hasattr(sample[2], "shape") else type(sample[2]))
    print("state shape:", sample[3].shape if hasattr(sample[3], "shape") else type(sample[3]))
    print("relative action shape:", sample[4].shape if hasattr(sample[4], "shape") else type(sample[4]))
    print("action_is_pad shape:", sample[5].shape if hasattr(sample[5], "shape") else type(sample[5]))