from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def parse_split_range(s: str) -> list[int]:
    # supports "start:end" (end exclusive)
    start, end = map(int, s.split(":"))
    return list(range(start, end))


class AlohaStaticCupsOpenDataset(Dataset):
    """
    Frame-level dataset wrapper for:
        lerobot/aloha_static_cups_open

    Each sample corresponds to a single timestep.

    Returns:
        (
            cam_high,
            cam_left_wrist,
            cam_right_wrist,
            state,
            action_chunk,
        )
    """

    def __init__(
        self,
        repo_id: str = "lerobot/aloha_static_cups_open",
        root: str | Path = Path(str(hf_home)).joinpath("lerobot").joinpath("aloha_static_cups_open"),
        tolerance_s: float = 1e-4,
        split: str = "train",
        chunk_size: int = 16,
    ):
        super().__init__()

        valid_splits = {"train"}
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split name, {split}. Must be one of {valid_splits}."
            )

        self.repo_id = repo_id
        self.root = root
        self.split = split
        self.chunk_size = chunk_size

        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=repo_id,
            root=root,
        )

        info = dataset_metadata.info
        fps = dataset_metadata.fps

        episodes = parse_split_range(info["splits"][split])

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            tolerance_s=tolerance_s,
            delta_timestamps={
                "action": [i / fps for i in range(chunk_size)],
            },
        )

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        cam_high = sample["observation.images.cam_high"]
        cam_left_wrist = sample["observation.images.cam_left_wrist"]
        cam_right_wrist = sample["observation.images.cam_right_wrist"]
        state = sample["observation.state"]
        action_chunk = sample["action"]

        return (
            cam_high,
            cam_left_wrist,
            cam_right_wrist,
            state,
            action_chunk,
        )


if __name__ == "__main__":
    ds = AlohaStaticCupsOpenDataset(split="train")
    print("Dataset length:", len(ds))

    sample = ds[0]
    print(type(sample))
    print("cam_high shape:", sample[0].shape if hasattr(sample[0], "shape") else type(sample[0]))
    print("cam_left_wrist shape:", sample[1].shape if hasattr(sample[1], "shape") else type(sample[1]))
    print("cam_right_wrist shape:", sample[2].shape if hasattr(sample[2], "shape") else type(sample[2]))
    print("state shape:", sample[3].shape if hasattr(sample[3], "shape") else type(sample[3]))
    print("action shape:", sample[4].shape if hasattr(sample[4], "shape") else type(sample[4]))