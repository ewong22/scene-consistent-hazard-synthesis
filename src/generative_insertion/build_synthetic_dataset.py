from pathlib import Path
from typing import Optional, List, Dict
import random
import json

import cv2
import numpy as np

from src.datasets.lost_and_found import list_image_label_pairs

IMG_ROOT = Path("data/raw/lostandfound/leftImg8bit")
DEPTH_ROOT = Path("data/processed/depth")
MASK_ROOT = Path("data/processed/masks")
SYNTH_ROOT = Path("data/synthetic/lostandfound")

DEBRIS_PROMPTS = [
    "a worn black tire lying on the asphalt",
    "a cardboard box on the road",
    "a fallen traffic cone on the street",
    "a small metal object on the road surface",
    "a piece of wooden debris on the asphalt",
]


def choose_prompt() -> str:
    return random.choice(DEBRIS_PROMPTS)


def find_depth_path(img_path: Path) -> Path:
    rel = img_path.relative_to(IMG_ROOT)
    depth_path = DEPTH_ROOT / rel
    depth_path = Path(str(depth_path).replace("_leftImg8bit.png", "_depth.png"))
    return depth_path


def find_mask_path(img_path: Path) -> Path:
    rel = img_path.relative_to(IMG_ROOT)
    mask_path = MASK_ROOT / rel
    mask_path = Path(str(mask_path).replace("_leftImg8bit.png", "_insertmask.png"))
    return mask_path


def fake_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Placeholder "inpainting": just overlays a colored rectangle where the mask is white.
    Later, replace this with a real diffusion inpainting call.
    """
    synth = image_bgr.copy()

    color = (0, 140, 255)
    synth[mask == 255] = color

    return synth


def build_synthetic_for_split(split: str = "train", max_images: Optional[int] = None):
    """
    For each image in split:
      - load image, depth, mask
      - choose a text prompt
      - run fake_inpaint
      - save synthetic image + metadata
    """
    pairs = list_image_label_pairs(split)
    print(f"{split}: {len(pairs)} image/label pairs found")

    if max_images is not None:
        pairs = pairs[:max_images]

    for i, (img_path, _) in enumerate(pairs):
        depth_path = find_depth_path(img_path)
        mask_path = find_mask_path(img_path)

        if not depth_path.exists():
            print(f"[WARN] missing depth: {depth_path}")
            continue
        if not mask_path.exists():
            print(f"[WARN] missing mask: {mask_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] failed to read mask: {mask_path}")
            continue

        prompt = choose_prompt()

        # TODO: later replace fake_inpaint with real diffusion inpainting
        synth_bgr = fake_inpaint(img_bgr, mask)

        rel = img_path.relative_to(IMG_ROOT)
        out_img_path = SYNTH_ROOT / "images" / split / rel
        out_img_path = Path(str(out_img_path).replace("_leftImg8bit.png", "_synth.png"))

        out_meta_path = SYNTH_ROOT / "meta" / split / rel
        out_meta_path = Path(str(out_meta_path).replace("_leftImg8bit.png", "_synth.json"))

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{len(pairs)}] {img_path} -> {out_img_path}")
        cv2.imwrite(str(out_img_path), synth_bgr)

        meta: Dict[str, str] = {
            "original_image": str(img_path),
            "depth_map": str(depth_path),
            "mask": str(mask_path),
            "synthetic_image": str(out_img_path),
            "prompt": prompt,
            "split": split,
        }
        with open(out_meta_path, "w") as f:
            json.dump(meta, f, indent=2)


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

    build_synthetic_for_split(split=args.split, max_images=args.max_images)