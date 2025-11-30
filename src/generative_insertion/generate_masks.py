from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from src.datasets.lost_and_found import list_image_label_pairs

MASK_ROOT = Path("data/processed/masks")


def make_simple_road_mask(img: np.ndarray) -> np.ndarray:
    """
    Make a simple rectangular mask in the lower-middle of the image.
    This is a placeholder; later we'll replace this with VLM-guided masks.

    img: H x W x 3 RGB/BGR image
    returns: H x W uint8 mask with values 0 or 255
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    top = int(h * 0.6)
    bottom = int(h * 0.9)

    rect_width = int(w * 0.15)
    rect_height = bottom - top

    center_x = w // 2
    left = center_x - rect_width // 2
    right = center_x + rect_width // 2

    left = max(0, left)
    right = min(w, right)

    mask[top:bottom, left:right] = 255

    return mask


def generate_masks_for_split(split: str = "train", max_images: Optional[int] = None):
    """
    For each image in the split, generate a simple insertion mask and save it.

    Output path mirrors the image structure:
      data/processed/masks/<split>/<seq>/<name>_insertmask.png
    """
    pairs = list_image_label_pairs(split)
    print(f"{split}: {len(pairs)} image/label pairs found")

    if max_images is not None:
        pairs = pairs[:max_images]

    for i, (img_path, _) in enumerate(pairs):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        mask = make_simple_road_mask(img_bgr)

        rel = img_path.relative_to("data/raw/lostandfound/leftImg8bit")
        out_path = MASK_ROOT / rel
        out_path = Path(str(out_path).replace("_leftImg8bit.png", "_insertmask.png"))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{len(pairs)}] {img_path} -> {out_path}")
        cv2.imwrite(str(out_path), mask)


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
    args = parser.parse_args()

    generate_masks_for_split(split=args.split, max_images=args.max_images)