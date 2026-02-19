from typing import List

import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

class CholecystectomyDataset(Dataset):
    """
    Frame-level dataset similar to MetaWorldDataset.

    Each sample corresponds to a single timestep.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        image_keys=None,
    ):
        super().__init__()

        self.dataset = LeRobotDataset.from_pretrained(dataset_root)

        if image_keys is None:
            image_keys = [
                "observation.images.endoscope.left",
                "observation.images.endoscope.right",
            ]

        self.image_keys = image_keys

        # Get episode indices for split
        self.episode_ids = self.dataset.get_split(split)

        # Build global frame index (episode_id, frame_id)
        self.index = []
        for ep_id in self.episode_ids:
            ep_len = self.dataset.get_episode_length(ep_id)
            for t in range(ep_len):
                self.index.append((ep_id, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_id, t = self.index[idx]
        frame = self.dataset.get_frame(ep_id, t)

        state = torch.tensor(frame["observation.state"], dtype=torch.float32)
        action = torch.tensor(frame["action"], dtype=torch.float32)

        images = {}
        for k in self.image_keys:
            if k in frame:
                img = torch.from_numpy(frame[k]).permute(2, 0, 1).float() / 255.0
                images[k] = img

        return {
            "obs": state,
            "action": action,
            "images": images,
        }
    
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
        valid_splits = {"train", "val"}
        assert split in valid_splits, f"Invalid split name, {split}. Must be one of {valid_splits}."

        self.split = split
        self.chunk_size = chunk_size

        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=repo_id, root=root
        )

        # get train, val splits
        random.seed(42)
        num_eps = dataset_metadata.total_episodes
        val_prop = 0.2                          # validation represents approx 20% of dataset
        N_val_eps = int(val_prop * num_eps)

        ep_idxs = list(range(num_eps))
        val_ep_idxs = random.sample(ep_idxs, N_val_eps)
        train_ep_idxs = list(set(ep_idxs) - set(val_ep_idxs))

        ep_idxs = train_ep_idxs if split=="train" else val_ep_idxs

        self.dataset = LeRobotDataset(
            repo_id=repo_id, root=root, tolerance_s=tolerance_s, episodes=ep_idxs
        )
        

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        endoscope_image = sample["observation.images.endoscope.left"]
        wrist_r = sample["observation.images.wrist.right"]
        wrist_l = sample["observation.images.wrist.left"]
        state = sample["observation.state"]
        instruction_text = sample["instruction.text"]

        ep_idx = sample["episode_index"]
        ep_end = self.dataset.episode_data_index["to"][ep_idx]

        max_valid = ep_end - idx
        valid_steps = min(self.chunk_size, max_valid)

        # Collect valid actions
        actions = [
            self.dataset[i]["action"]
            for i in range(idx, idx + valid_steps)
        ]
        action_chunk = torch.stack(actions, dim=0)  # [valid_steps, action_dim]

        action_dim = action_chunk.shape[-1]

        # Pad if necessary
        if valid_steps < self.chunk_size:
            pad_size = self.chunk_size - valid_steps
            pad_tensor = torch.zeros(pad_size, action_dim, dtype=action_chunk.dtype)
            action_chunk = torch.cat([action_chunk, pad_tensor], dim=0)

        # Padding mask (1 = valid, 0 = padded)
        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        is_pad[valid_steps:] = True

        return (
            endoscope_image,
            wrist_l,
            wrist_r,
            state,
            action_chunk,
            is_pad,
            instruction_text,
        )

    
if __name__=="__main__":
    act_dataset = CholecystectomyACTDataset(
        repo_id="surpass/cholecystectomy_dummy", 
        root="/home/gbyrd/SURPASS/.hf/lerobot/surpass/cholecystectomy_dummy",
        tolerance_s=1e-4,
        split="train"
    )
    entry = act_dataset[108]
    pass