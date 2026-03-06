from typing import Optional

from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
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
        chunk_size: int = 16
    ):
        super().__init__()
        valid_splits = {"train", "test"}
        assert split in valid_splits, f"Invalid split name, {split}. Must be one of {valid_splits}."

        self.split = split

        dataset_metadata = LeRobotDatasetMetadata(
            repo_id=repo_id, root=root
        )

        # get train, val splits
        def parse_split_range(s: str) -> list[int]:
            # supports "start:end" (end exclusive)
            start, end = map(int, s.split(":"))
            return list(range(start, end))

        info = dataset_metadata.info

        episodes = parse_split_range(info["splits"][split])

        # calculate the timestamps based on the desired action chunk size
        timestamps = [i / info["fps"] for i in range(chunk_size)]

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            tolerance_s=tolerance_s,
            delta_timestamps = {
                "action_hybrid_relative": timestamps,
            }  # type: ignore
        )

        self.episode_start_idx = self.dataset.episode_data_index["from"][episodes[0]].item()
        self.episode_end_idx = self.dataset.episode_data_index["to"][episodes[-1]].item()

    def __len__(self):
        return self.episode_end_idx-self.episode_start_idx

    def __getitem__(self, idx):

        idx = idx + self.episode_start_idx

        sample = self.dataset[idx]

        endoscope_image = sample["observation.images.endoscope.left"]
        wrist_r = sample["observation.images.wrist.right"]
        wrist_l = sample["observation.images.wrist.left"]
        state = sample["observation.state"]
        action_chunk = sample["action_hybrid_relative"]
        action_is_pad = sample["action_hybrid_relative_is_pad"]
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

class CholecystectomyACTDataset_GraspOnly_PSM1_Only(CholecystectomyACTDataset):
    """
    Filters dataset to produce only samples from the "grasp" action and only
    return actions corresponding to PSM1.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        tolerance_s: float,
        split: str,
        chunk_size: int = 16
    ):
        super().__init__(
            repo_id,
            root,
            tolerance_s,
            split,
            chunk_size
        )
        self.chunk_size = chunk_size

        # get a list of indices where the "anchor" frame's "instruction.text" value
        # was "grasp"
        indices = [
            i for i, text in enumerate(self.dataset.hf_dataset["instruction.text"])
            if text == "grasp"
        ]

        indices_set = set(indices)

        self.global_idxs_to_chunk_len = dict()

        # if another type of instruction.text appears in the middle of the
        # trajectory, we will want to revise the action_is_pad and pad
        # the future timesteps from this point. we will precompute the chunk length of
        # each global index
        for idx in tqdm(indices):
            for i in range(1, self.chunk_size + 1):
                if idx + i not in indices_set:
                    # Keep the anchor step as valid supervision.
                    # `i` here means the first invalid offset (1-based).
                    self.global_idxs_to_chunk_len[idx] = i
                    break
            if idx not in self.global_idxs_to_chunk_len:
                self.global_idxs_to_chunk_len[idx] = self.chunk_size

        # get train, test, val split
        seed = 42
        rng = random.Random(seed)

        indices = indices.copy()
        rng.shuffle(indices)

        p1, p2, p3 = 0.7, 0.15, 0.15

        n = len(indices)
        n1 = int(p1 * n)
        n2 = int(p2 * n)

        train_idxs = indices[:n1]
        val_idxs = indices[n1:n1+n2]
        test_idxs = indices[n1+n2:]

        if self.split == "train":
            self.idxs = train_idxs
        elif self.split == "test":
            self.idxs = test_idxs
        else:
            raise NotImplementedError(f"Invalid split type: {self.split}")

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):

        # get the index from the randomly sampled list of indices given by the
        # dataset split
        global_idx = self.idxs[idx]

        sample = self.dataset[global_idx]

        endoscope_image = sample["observation.images.endoscope.left"]
        wrist_r = sample["observation.images.wrist.right"]
        wrist_l = sample["observation.images.wrist.left"]
        state = sample["observation.state"]
        action_chunk = sample["action_hybrid_relative"]
        action_is_pad = sample["action_hybrid_relative_is_pad"].clone().to(dtype=torch.bool)
        instruction_text = sample["instruction.text"]

        # modify action_is_pad based on precomputed chunk length if needed
        action_is_pad[self.global_idxs_to_chunk_len[global_idx]:] = True

        return (
            endoscope_image,
            wrist_l,
            wrist_r,
            state[:8],           # only provide state for the PSM1
            action_chunk[:, :8],    # only provide action chunk for the PSM1
            action_is_pad,
            instruction_text,
        )

class Debug(CholecystectomyACTDataset_GraspOnly_PSM1_Only):
    """
    Filters dataset to produce only samples from the "grasp" action and only
    return actions corresponding to PSM1.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        tolerance_s: float,
        split: str,
        chunk_size: int = 16
    ):
        super().__init__(
            repo_id,
            root,
            tolerance_s,
            split,
            chunk_size
        )

        self.samples = list()
        for i in range(128):
            self.samples.append(self.dataset[self.global_idxs_to_chunk_len[i]])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        # get the index from the randomly sampled list of indices given by the
        # dataset split
        global_idx = self.idxs[idx]

        sample = self.samples[idx]

        endoscope_image = sample["observation.images.endoscope.left"]
        wrist_r = sample["observation.images.wrist.right"]
        wrist_l = sample["observation.images.wrist.left"]
        state = sample["observation.state"]
        action_chunk = sample["action_hybrid_relative"]
        action_is_pad = sample["action_hybrid_relative_is_pad"].clone().to(dtype=torch.bool)
        instruction_text = sample["instruction.text"]

        # modify action_is_pad based on precomputed chunk length if needed
        action_is_pad[self.global_idxs_to_chunk_len[global_idx]:] = True

        return (
            endoscope_image,
            wrist_l,
            wrist_r,
            state[:8],           # only provide state for the PSM1
            action_chunk[:, :8],    # only provide action chunk for the PSM1
            action_is_pad,
            instruction_text,
        )  

if __name__=="__main__":
    act_dataset = CholecystectomyACTDataset(
        repo_id="surpass/cholecystectomy_accelerated", 
        root="/home/byrdgb1/surpass/.hf/lerobot/surpass/cholecystectomy_accelerated",
        tolerance_s=1e-4,
        split="test"
    )

    act_dataset[0]

    print("Done")

    # start_act = time.time()
    # for i in tqdm(range(N_index)):
    #     act_dataset[i]
    # print(f"Time ACT at {N_index} calls: {time.time()-start_act}")

    # start_act = time.time()
    # for i in tqdm(range(N_index)):
    #     act_dataset_cs_1[i]
    # print(f"Time ACT cs 1 at {N_index} calls: {time.time()-start_act}")

    # start = time.time()
    # for i in tqdm(range(N_index)):
    #     dataset[i]
    # print(f"Time normal dataset at {N_index} calls: {time.time()-start}")
