import decord
decord.bridge.set_bridge('torch')
import os
import cv2
from torch.utils.data import Dataset
from einops import rearrange
import random
import torch

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
        self.root = video_root
        self.img_folder_paths = []
        self.prompt = []
        # for img_folder in os.listdir(self.root):
        #     self.img_folder_paths.append(os.path.join(self.root, img_folder))
        #     self.prompt.append(root.split('/')[-1].replace('_', ' '))
        self.img_folder_paths.append(os.path.join(self.root, "images0"))
        self.prompt.append(self.root.split('/')[-1].replace('_', ' '))
        self.prompt_ids = []

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return len(self.img_folder_paths)

    def __getitem__(self, idx):
        img_folder_path = self.img_folder_paths[idx]
        img_paths = sorted([os.path.join(img_folder_path, file) for file in os.listdir(img_folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))])

        start_idx = random.randint(0, len(img_paths)-self.n_sample_frames*self.sample_frame_rate-1)
        sample_idx = list(range(start_idx, len(img_paths), self.sample_frame_rate))[:self.n_sample_frames]

        img_t = torch.zeros((len(img_paths), self.height, self.width, 3), dtype=torch.uint8)
        for i, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t[i] = torch.from_numpy(img)
        img_t = img_t[sample_idx]
        img_t = rearrange(img_t, "f h w c -> f c h w")
        if random.uniform(0, 1) > 0.5:
            img_t = torch.flip(img_t, dims=[3])
        
        item = {
            "pixel_values": (img_t / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[idx]
        }
        return item
