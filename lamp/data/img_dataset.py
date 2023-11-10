import os
import cv2
from torch.utils.data import Dataset
from einops import rearrange
import random
import torch
import matplotlib.pyplot as plt
import copy
import re

# custom comparator for file names
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

class LAMPImageDataset(Dataset):
    def __init__(
            self,
            video_root: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.width = width
        self.height = height
        self.channels = 3
        self.traj_length = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        self.root = video_root
        self.traj_folder_paths = sorted([os.path.join(self.root, traj_folder) for traj_folder in os.listdir(self.root)], key=alphanum_key)

        # lang labels
        self.prompt = []
        for traj_folder_path in self.traj_folder_paths:
            with open(os.path.join(traj_folder_path, "lang.txt"), 'r') as file:
                self.prompt.append(file.readline().strip())
        self.prompt_ids = [] # tokenized prompts (handled in training script)

    def __len__(self):
        return len(self.traj_folder_paths)

    def __getitem__(self, idx):
        # traj directory handling (single view)
        traj_folder_path = self.traj_folder_paths[idx]
        cam0_folder_path = os.path.join(traj_folder_path, "images0")
        img_paths = sorted([os.path.join(cam0_folder_path, file) for file in os.listdir(cam0_folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))], key=alphanum_key)
        start_idx = random.randint(0, len(img_paths)-self.traj_length*self.sample_frame_rate-1)
        sample_idx = list(range(start_idx, len(img_paths), self.sample_frame_rate))[:self.traj_length]

        # traj imgs (single view)
        img_t = torch.zeros((len(sample_idx), self.height, self.width, self.channels), dtype=torch.uint8)
        for i, val in enumerate(sample_idx):
            img_path = copy.copy(img_paths[val])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.width, self.height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t[i] = torch.from_numpy(img)
        
        # img manipulation 
        img_t = img_t.permute(0, 3, 1, 2)
        if random.uniform(0, 1) > 0.5:
            img_t = torch.flip(img_t, dims=[3])
        item = {
            "pixel_values": (img_t / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[idx]
        }
        return item
