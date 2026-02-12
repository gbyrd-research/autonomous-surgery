import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
        dataset_root: str | Path,
        split: str = "train",
        window_size: int = 16,
        image_keys=None,
    ):
        super().__init__()

        self.dataset = LeRobotDataset(dataset_root)
        self.window_size = window_size

        if image_keys is None:
            image_keys = [
                "observation.images.endoscope.left",
                "observation.images.endoscope.right",
            ]

        self.image_keys = image_keys

        self.episode_ids = self.dataset.get_split(split)

        # Build index of valid (episode_id, start_t)
        self.index = []

        for ep_id in self.episode_ids:
            ep_len = self.dataset.get_episode_length(ep_id)

            for start in range(0, ep_len - window_size + 1):
                self.index.append((ep_id, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_id, start = self.index[idx]
        end = start + self.window_size

        states = []
        actions = []
        images = {k: [] for k in self.image_keys}

        for t in range(start, end):
            frame = self.dataset.get_frame(ep_id, t)

            states.append(frame["observation.state"])
            actions.append(frame["action"])

            for k in self.image_keys:
                if k in frame:
                    img = torch.from_numpy(frame[k]).permute(2, 0, 1).float() / 255.0
                    images[k].append(img)

        states = torch.tensor(np.stack(states), dtype=torch.float32)
        actions = torch.tensor(np.stack(actions), dtype=torch.float32)

        for k in images:
            if len(images[k]) > 0:
                images[k] = torch.stack(images[k])
            else:
                images[k] = None

        return {
            "obs": states,
            "action": actions,
            "images": images,
        }
    
if __name__=="__main__":
    act_dataset = CholecystectomyACTDataset(dataset_root="jchen396/openh_test")
    entry = act_dataset[0]
    pass