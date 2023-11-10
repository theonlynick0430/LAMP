import argparse
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from diffusers.utils import check_min_version
from lamp.data.dataset import LAMPDataset
from lamp.data.img_dataset import LAMPImageDataset

# import os
# os.environ['LD_LIBRARY_PATH'] = '/data/anaconda3/'
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 4,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    # Get the training dataset
    train_dataset = LAMPImageDataset(**train_data)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # subsitution for tokenizer
    for prompt in train_dataset.prompt:
        train_dataset.prompt_ids.append(prompt)

    weight_dtype = torch.float32
    for i in range(100): # 100 epochs
        print(f"Epoch: {i}")
        for j, batch in enumerate(train_dataloader):
            print(f"Batch {j}")
            pixel_values = batch["pixel_values"].to(weight_dtype)
            prompt_id = batch["prompt_ids"]
            print(pixel_values.shape)
            print(prompt_id)


if __name__ == "__main__":
    print("Entered main function")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()
    print
    main(**OmegaConf.load(args.config))
