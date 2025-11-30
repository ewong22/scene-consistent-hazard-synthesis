import os
from pathlib import Path
from typing import Optional

import cv2
import torch
import numpy as np

from src.datasets.lost_and_found import list_image_label_pairs

DEPTH_ROOT = Path("data/processed/depth")


def load_midas(device: str = "cpu"):
    """
    Load a MiDaS / DPT model from torch.hub.
    Using DPT_Large 
    """
    model_type = "DPT_Large"  # or "DPT_Hybrid" / "MiDaS_small"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if "DPT" in model_type:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform


def compute_and_save_depth_for_image(
    img_path: Path,
    out_path: Path,
    midas,
    transform,
    device: str = "cpu",
):
    """
    Compute depth for a single image and save as a 16-bit PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transformed = transform(img_rgb)
    if isinstance(transformed, dict):
        img_input = transformed["image"].to(device)
    else:
        img_input = transformed.to(device)

    with torch.no_grad():
        prediction = midas(img_input)

        h, w = img_rgb.shape[:2]
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),  
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()  

    depth = prediction.cpu().numpy()

    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)

    depth_16 = (depth_norm * 65535.0).astype(np.uint16)

    cv2.imwrite(str(out_path), depth_16)


def process_split(
    split: str = "train",
    max_images: Optional[int] = None,
    device: str = "cpu",
):
    """
    Compute depth maps for all images in a split (train/test),
    optionally limited to first max_images for quick tests.
    """
    print(f"Loading MiDaS for device={device} ...")
    midas, transform = load_midas(device=device)

    pairs = list_image_label_pairs(split)
    print(f"{split}: {len(pairs)} image/label pairs found")

    if max_images is not None:
        pairs = pairs[:max_images]

    for i, (img_path, _) in enumerate(pairs):
        rel = img_path.relative_to("data/raw/lostandfound/leftImg8bit")
        out_path = DEPTH_ROOT / rel
        out_path = Path(str(out_path).replace("_leftImg8bit.png", "_depth.png"))

        print(f"[{i+1}/{len(pairs)}] {img_path} -> {out_path}")
        compute_and_save_depth_for_image(
            img_path, out_path, midas, transform, device=device
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="limit for quick testing (None = all)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    process_split(split=args.split, max_images=args.max_images, device=args.device)