from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.lost_and_found import list_image_label_pairs


class LostAndFoundSegDataset(Dataset):
    """
    Simple semantic segmentation dataset for Lost & Found.

    Returns:
      image: 3xHxW float tensor in [0,1]
      label: HxW long tensor with class IDs (from *_gtCoarse_labelIds.png)
    """

    def __init__(self, split: str = "train", transform=None):
        assert split in ["train", "test"]
        self.split = split
        self.transform = transform
        self.pairs = list_image_label_pairs(split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) 

        label = Image.open(label_path)
        label_np = np.array(label, dtype=np.int64)  
        label_tensor = torch.from_numpy(label_np)

        if self.transform is not None:
            img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

        return img_tensor, label_tensor


if __name__ == "__main__":
    ds = LostAndFoundSegDataset(split="train")
    print("Train size:", len(ds))
    img, label = ds[0]
    print("Image shape:", img.shape)   
    print("Label shape:", label.shape)  