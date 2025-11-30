from pathlib import Path
from typing import List, Tuple
from PIL import Image

# Root paths
DATA_ROOT = Path("data/raw/lostandfound")
IMG_ROOT = DATA_ROOT / "leftImg8bit"
GT_ROOT = DATA_ROOT / "gtCoarse"


def list_image_label_pairs(split: str = "train") -> List[Tuple[Path, Path]]:
    """
    Return (image_path, label_path) pairs for a given split: 'train' or 'test'.

    Layout assumed:
      leftImg8bit/<split>/<seq>/<name>_leftImg8bit.png
      gtCoarse/<split>/<seq>/<name>_gtCoarse_labelIds.png
    """
    img_split_dir = IMG_ROOT / split
    gt_split_dir = GT_ROOT / split

    if not img_split_dir.exists():
        raise FileNotFoundError(f"Image split dir not found: {img_split_dir}")
    if not gt_split_dir.exists():
        raise FileNotFoundError(f"GT split dir not found: {gt_split_dir}")

    pairs: List[Tuple[Path, Path]] = []

    for img_path in img_split_dir.rglob("*_leftImg8bit.png"):

        stem = img_path.name.replace("_leftImg8bit.png", "")


        seq_dir = img_path.parent.name

        gt_filename = f"{stem}_gtCoarse_labelIds.png"
        gt_path = gt_split_dir / seq_dir / gt_filename

        if gt_path.exists():
            pairs.append((img_path, gt_path))
        else:
            print(f"[WARN] Missing GT for {img_path} -> {gt_path}")

    return pairs


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_label(path: Path) -> Image.Image:

    return Image.open(path)


if __name__ == "__main__":
    for split in ["train", "test"]:
        pairs = list_image_label_pairs(split)
        print(f"{split}: found {len(pairs)} image/label pairs")
        if pairs:
            img_path, gt_path = pairs[0]
            print(f"  example image: {img_path}")
            print(f"  example label: {gt_path}")