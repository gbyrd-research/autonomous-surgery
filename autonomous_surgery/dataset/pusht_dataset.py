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
    Sequence-style dataset wrapper for LeRobot PushT.

    Each sample corresponds to one timestep and returns:
      - a history of images ending at the current timestep
      - a future chunk of states/actions starting at the current timestep

    Returns:
        (
            images,             # dict: {"3rd_person": [H, C, H_img, W_img]}
            initial_state,      # [state_dim]
            action_chunk,       # [T, action_dim]
            action_is_pad,      # [T] bool
            instruction_text,   # str
        )

    Notes:
      - H = image_history
      - T = chunk_size
      - image history is ordered oldest -> newest
    """

    def __init__(
        self,
        repo_id: str = "lerobot/pusht",
        root: str | Path = Path(str(hf_home)).joinpath("lerobot").joinpath("pusht"),
        tolerance_s: float = 1e-4,
        split: str = "train",
        chunk_size: int = 16,
        image_history: int = 4,
    ):
        super().__init__()

        if image_history < 1:
            raise ValueError(f"image_history must be >= 1, got {image_history}")

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

        # Future window for actions/states starting at current timestep.
        future_timestamps = [i / fps for i in range(chunk_size)]

        # Past image history ending at current timestep.
        # Example for image_history=4: [-3/fps, -2/fps, -1/fps, 0]
        image_timestamps = [-(image_history - 1 - i) / fps for i in range(image_history)]

        # Load the FULL dataset to avoid LeRobot's episode-index bug on nonzero episode subsets
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            tolerance_s=tolerance_s,
            delta_timestamps={
                "observation.image": image_timestamps,
                "observation.state": future_timestamps,
                "action": future_timestamps,
            },
        )

        self.split = split
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.image_history = image_history
        self.fps = fps

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

        # [image_history, C, H, W]
        image_history = sample["observation.image"]

        # [T, state_dim]
        state_chunk = sample["observation.state"]

        # [T, action_dim]
        action_chunk = sample["action"]

        action_is_pad = sample.get(
            "action_is_pad",
            torch.zeros(action_chunk.shape[0], dtype=torch.bool),
        )

        relative_action_chunk = action_chunk - state_chunk
        initial_state = state_chunk[0]
        instruction_text = "placeholder"

        images = {
            "3rd_person": image_history[-1]
        }
        for idx in range(image_history.shape[0]-1):
            images[f"3rd_person_{idx}"] = image_history[idx]

        return (
            images,
            initial_state,
            relative_action_chunk,
            action_is_pad,
            instruction_text,
        )


if __name__ == "__main__":
    pusht_ds = PushTDataset(split="test", chunk_size=16, image_history=4)
    print("PushT length:", len(pusht_ds))

    sample = pusht_ds[0]

    images, initial_state, action_chunk, action_is_pad, instruction_text = sample

    print("images type:", type(images))
    print("3rd_person shape:", images["3rd_person"].shape)
    print("initial_state shape:", initial_state.shape)
    print("action_chunk shape:", action_chunk.shape)
    print("action_is_pad shape:", action_is_pad.shape)
    print("instruction_text:", instruction_text)