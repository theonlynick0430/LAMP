from lamp.util import save_videos_grid
import argparse
from omegaconf import OmegaConf
import os
from lamp.data.img_dataset import alphanum_key
import cv2
import torch

def main(
    output_dir: str,
    root: str,
    video_length: int = 16,
    width: int = 320,
    height: int = 240,
    **kwargs,
):
    traj_folder_paths = sorted([os.path.join(root, traj_folder) for traj_folder in os.listdir(root)], key=alphanum_key)
    img_t = torch.zeros((len(traj_folder_paths), video_length, height, width, 3), dtype=torch.uint8)
    prompts = []
    for i, traj_folder_path in enumerate(traj_folder_paths):
        with open(os.path.join(traj_folder_path, "lang.txt"), 'r') as file:
            prompts.append(file.readline().strip())
        for j in range(video_length):
            img_path = os.path.join(os.path.join(traj_folder_path, "images0"), f"im_{j}.jpg")
            img = cv2.imread(img_path)
            img = cv2.resize(img, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t[i, j, :] = torch.from_numpy(img)
    img_t = img_t.permute((0, 4, 1, 2, 3))/255.0
    save_videos_grid(img_t, f"{output_dir}/gt/trajs.gif")
    with open(f"{output_dir}/gt/prompts.txt", 'w') as file:
        for i, prompt in enumerate(prompts):
            file.write(f"traj{i} prompt: " + prompt + "\n") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config)["output_dir"], **OmegaConf.load(args.config)["validation_data"])