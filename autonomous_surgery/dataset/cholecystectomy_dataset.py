from typing import List
import time

import json
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

random.seed(42)

class CholecystectomyDataset(Dataset):
    """
    Frame-level dataset similar to MetaWorldDataset.

    Each sample corresponds to a single timestep.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        tolerance_s: float,
        split: str,
        chunk_size: int = 16,
    ):
        super().__init__()
        valid_splits = {"train", "val"}
        assert split in valid_splits, f"Invalid split name, {split}. Must be one of {valid_splits}."

        # self.split = split
        # self.chunk_size = chunk_size

        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=repo_id, root=root
        )

        # get train, val splits
        fps = dataset_metadata.fps
        # num_eps = dataset_metadata.total_episodes
        # val_prop = 0.2                          # validation represents approx 20% of dataset
        # N_val_eps = int(val_prop * num_eps)

        # ep_idxs = list(range(num_eps))
        # val_ep_idxs = random.sample(ep_idxs, N_val_eps)
        # train_ep_idxs = list(set(ep_idxs) - set(val_ep_idxs))

        # ep_idxs = train_ep_idxs if split=="train" else val_ep_idxs

        # self.dataset = LeRobotDataset(
        #     repo_id=repo_id,
        #     root=root,
        #     tolerance_s=tolerance_s,
        #     episodes=ep_idxs,
        #     delta_timestamps={
        #         "action": [i / fps for i in range(chunk_size)],  # fetch full chunk
        #     }
        # )

        def parse_split_range(s: str) -> list[int]:
            # supports "start:end" (end exclusive)
            start, end = map(int, s.split(":"))
            return list(range(start, end))

        info = dataset_metadata.info

        episodes = parse_split_range(info["splits"][split])

        self.dataset = LeRobotDataset(
            repo_id="surpass/cholecystectomy",
            root=root,
            episodes=episodes,
            tolerance_s=1e-4,
        )
        

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        endoscope_image = sample["observation.images.endoscope.left"]
        wrist_r = sample["observation.images.wrist.right"]
        wrist_l = sample["observation.images.wrist.left"]
        state = sample["observation.state"]
        action_chunk = sample["action"]
        # action_is_pad = sample["action_is_pad"]
        instruction_text = sample["instruction.text"]

        return (
            endoscope_image,
            wrist_l,
            wrist_r,
            state,
            action_chunk,
            # action_is_pad,
            instruction_text,
        )
    
class CholecystectomyACTDataset(Dataset):
    """
    Sequence dataset for ACT training.
    Mirrors MetaWorldDatasetACT behavior.

    Returns fixed-length windows:
        obs:     [T, state_dim]
        action:  [T, action_dim]
        images:  Dict[str, [T, C, H, W]]
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        tolerance_s: float,
        split: str,
        chunk_size: int = 16,
    ):
        super().__init__()
        valid_splits = {"train", "test"}
        assert split in valid_splits, f"Invalid split name, {split}. Must be one of {valid_splits}."

        # self.split = split
        # self.chunk_size = chunk_size

        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=repo_id, root=root
        )

        # get train, val splits
        fps = dataset_metadata.fps
        # num_eps = dataset_metadata.total_episodes
        # val_prop = 0.2                          # validation represents approx 20% of dataset
        # N_val_eps = int(val_prop * num_eps)

        # ep_idxs = list(range(num_eps))
        # val_ep_idxs = random.sample(ep_idxs, N_val_eps)
        # train_ep_idxs = list(set(ep_idxs) - set(val_ep_idxs))

        # ep_idxs = train_ep_idxs if split=="train" else val_ep_idxs

        # self.dataset = LeRobotDataset(
        #     repo_id=repo_id,
        #     root=root,
        #     tolerance_s=tolerance_s,
        #     episodes=ep_idxs,
        #     delta_timestamps={
        #         "action": [i / fps for i in range(chunk_size)],  # fetch full chunk
        #     }
        # )

        def parse_split_range(s: str) -> list[int]:
            # supports "start:end" (end exclusive)
            start, end = map(int, s.split(":"))
            return list(range(start, end))

        info = dataset_metadata.info

        episodes = parse_split_range(info["splits"][split])

        self.dataset = LeRobotDataset(
            repo_id="surpass/cholecystectomy",
            root=root,
            episodes=episodes,
            tolerance_s=1e-4,
            delta_timestamps={
                "action": [i / info["fps"] for i in range(chunk_size)],
            },
        )
        

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        endoscope_image = sample["observation.images.endoscope.left"]
        wrist_r = sample["observation.images.wrist.right"]
        wrist_l = sample["observation.images.wrist.left"]
        state = sample["observation.state"]
        action_chunk = sample["action"]
        action_is_pad = sample["action_is_pad"]
        instruction_text = sample["instruction.text"]

        return (
            endoscope_image,
            wrist_l,
            wrist_r,
            state,
            action_chunk,
            action_is_pad,
            instruction_text,
        )

    
if __name__=="__main__":
    act_dataset = CholecystectomyACTDataset(
        repo_id="surpass/cholecystectomy", 
        root="/home/gbyrd/SURPASS/.hf/lerobot/surpass/cholecystectomy",
        tolerance_s=1e-4,
        split="train"
    )
    act_dataset_cs_1 = CholecystectomyACTDataset(
        repo_id="surpass/cholecystectomy", 
        root="/home/gbyrd/SURPASS/.hf/lerobot/surpass/cholecystectomy",
        tolerance_s=1e-4,
        split="train",
        chunk_size=1
    )
    dataset = CholecystectomyDataset(
        repo_id="surpass/cholecystectomy", 
        root="/home/gbyrd/SURPASS/.hf/lerobot/surpass/cholecystectomy",
        tolerance_s=1e-4,
        split="train"
    )
    
    N_index = 1000

    start_act = time.time()
    for i in tqdm(range(N_index)):
        act_dataset[i]
    print(f"Time ACT at {N_index} calls: {time.time()-start_act}")

    start_act = time.time()
    for i in tqdm(range(N_index)):
        act_dataset_cs_1[i]
    print(f"Time ACT cs 1 at {N_index} calls: {time.time()-start_act}")

    start = time.time()
    for i in tqdm(range(N_index)):
        dataset[i]
    print(f"Time normal dataset at {N_index} calls: {time.time()-start}")
