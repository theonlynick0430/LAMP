import os
import cv2
from torch.utils.data import Dataset
from einops import rearrange
import random
import torch
import matplotlib.pyplot as plt
import copy
import re
from tqdm import tqdm
import json

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
            root: str,
            width: int = 512,
            height: int = 512,
            traj_length: int = 8,
            sample_frame_freq: int = 1,
            preload_images: bool = True,  # Option to preload images
            # video_length: int = 16
    ):
        self.width = width
        self.height = height
        self.channels = 3
        self.traj_length = traj_length
        self.sample_frame_freq = sample_frame_freq
        self.preload_images = preload_images
        # self.video_length = video_length

        self.root = root
        self.load_data()

    def load_data(self):
        with open(self.root, 'r') as file:
            self.data_info = json.load(file)

        self.data_info = self.data_info[:20]
        self.traj_folder_paths = []
        self.img_paths = []
        self.preloaded_images = {} if self.preload_images else None

        for info in tqdm(self.data_info, desc="Loading dataset"):
            cam0_folder_path = info['images0']
            traj_img_paths = sorted([os.path.join(cam0_folder_path, file) for file in os.listdir(cam0_folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))], key=alphanum_key)
            self.traj_folder_paths.append(cam0_folder_path)
            self.img_paths.append(traj_img_paths)

            if self.preload_images:
                for img_path in tqdm(traj_img_paths, desc="Preloading images", leave=False):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (self.width, self.height))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.transpose(2, 0, 1)  # Change from HWC to CHW format
                    self.preloaded_images[img_path] = img

        # lang labels
        self.prompt = [open(info['lang'], 'r').readline().strip() for info in self.data_info]
        self.prompt_ids = []  # tokenized prompts (handled in training script)

    def __len__(self):
        return len(self.traj_folder_paths)

    def __getitem__(self, idx):
        traj_img_paths = self.img_paths[idx]
        start_idx = random.randint(0, len(traj_img_paths) - self.traj_length * self.sample_frame_freq - 1)
        sample_idx = list(range(start_idx, len(traj_img_paths), self.sample_frame_freq))[:self.traj_length]

        img_t = torch.zeros((len(sample_idx), self.channels, self.height, self.width), dtype=torch.uint8)

        for i, val in enumerate(sample_idx):
            img_path = traj_img_paths[val]
            img = self.preloaded_images[img_path] if self.preload_images else cv2.imread(img_path)
            if not self.preload_images:
                img = cv2.resize(img, (self.width, self.height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t[i] = torch.from_numpy(img)

        # img manipulation 
        img_t = img_t.permute(0, 1, 2, 3)
        if random.uniform(0, 1) > 0.5:
            img_t = torch.flip(img_t, dims=[3])
        item = {
            "pixel_values": (img_t / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[idx]
        }
        return item

class ValidationDataset(Dataset):
    def __init__(self, validation_data):
        self.root = validation_data['root']
        self.width = validation_data['width']
        self.height = validation_data['height']
        self.video_length = validation_data['video_length']
        self.num_inference_steps = validation_data['num_inference_steps']
        self.guidance_scale = validation_data['guidance_scale']
        self.use_inv_latent = validation_data['use_inv_latent']
        self.num_inv_steps = validation_data['num_inv_steps']

        self.data = self.load_data(self.root)

    def load_data(self, json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)

        processed_data = []
        for item in data:
            img_folder_path = item['images0']
            prompt_path = item['lang']
            img_path = os.path.join(img_folder_path, "im_0.jpg")  # Assuming 'im_0.jpg' is the image to be used

            with open(prompt_path, 'r') as lang_file:
                prompt = lang_file.readline().strip()

            processed_data.append((img_path, prompt))
        
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, prompt = self.data[idx]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # Change from HWC to CHW format
        img = torch.Tensor(img) / 127.5 - 1.0

        return {"image": img, "prompt": prompt}