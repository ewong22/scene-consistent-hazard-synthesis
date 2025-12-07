from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

# Root paths (Adjust relative to where you run the script)
DATA_ROOT = Path("data/raw/lostandfound")
IMG_ROOT = DATA_ROOT / "leftImg8bit"
GT_ROOT = DATA_ROOT / "gtCoarse"

def get_laf_id_map() -> Dict[int, int]:
    """
    Creates a mapping from Original ID -> Train ID based on laf_table.pdf.
    
    Train IDs:
    0   -> Background / Non-Hazard (ignore or bg)
    1   -> Free Space (Road)
    2   -> Hazard (The target class)
    255 -> Ignore/Unlabeled
    """
    mapping = {}
    
    # Defaults: Set everything to 255 (ignore) first
    for i in range(256):
        mapping[i] = 255

    # --- Mappings from laf_table.pdf  ---
    
    # ID 1: Free space -> TrainID 1
    mapping[1] = 1
    
    # ID 0: Unlabeled/Ego Vehicle -> TrainID 255
    mapping[0] = 255
    
    # Random Non-Hazards -> TrainID 0 (Background/Negative)
    # IDs: 31 (Marker pole), 33 (Post red), 34 (Post stand), 
    # 36 (Timber small), 37 (Timber squared), 38 (Wheel cap), 39 (Wood thin)
    non_hazards = [31, 33, 34, 36, 37, 38, 39] 
    for original_id in non_hazards:
        mapping[original_id] = 0

    # Hazards -> TrainID 2
    # Standard objects (2-7), Random hazards (8-11, 12-21, 30, 32, 35),
    # Emotional hazards (22-29), Humans (40-42)
    # Basically, we map the known hazard ranges:
    hazard_ids = (
        list(range(2, 31)) +  # 2-30
        [32, 35] +            # Specifics
        list(range(40, 44))   # Humans 40-43
    )
    
    # Remove any that were explicitly set as non-hazards (just in case of overlap)
    for hid in hazard_ids:
        if hid not in non_hazards:
            mapping[hid] = 2
            
    return mapping

def list_image_label_pairs(split: str = "train") -> List[Tuple[Path, Path]]:
    """
    Return (image_path, label_path) pairs for a given split.
    """
    img_split_dir = IMG_ROOT / split
    gt_split_dir = GT_ROOT / split

    if not img_split_dir.exists():
        raise FileNotFoundError(f"Image split dir not found: {img_split_dir}")
    # Note: Sometimes test split might not have GT if it's a challenge server format
    # But for LostAndFound validation/test splits usually do.

    pairs: List[Tuple[Path, Path]] = []

    for img_path in img_split_dir.rglob("*_leftImg8bit.png"):
        stem = img_path.name.replace("_leftImg8bit.png", "")
        seq_dir = img_path.parent.name
        
        # We look for labelIds. 
        # NOTE: If have _instanceIds.png, prefer that for Mask R-CNN!
        gt_filename = f"{stem}_gtCoarse_labelIds.png"
        gt_path = gt_split_dir / seq_dir / gt_filename

        if gt_path.exists():
            pairs.append((img_path, gt_path))
        else:
            pass 

    return pairs

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_label(path: Path, convert_to_train_id: bool = True) -> np.ndarray:
    """
    Loads label and optionally converts raw IDs to Train IDs.
    Returns a numpy array.
    """
    label_img = Image.open(path)
    label_arr = np.array(label_img, dtype=np.uint8)

    if convert_to_train_id:
        id_map = get_laf_id_map()
        # Vectorized mapping using numpy array indexing
        # Create a lookup table
        lut = np.zeros(256, dtype=np.uint8)
        for k, v in id_map.items():
            lut[k] = v
        
        # Apply lookup
        label_arr = lut[label_arr]

    return label_arr

if __name__ == "__main__":
    for split in ["train", "test"]:
        try:
            pairs = list_image_label_pairs(split)
            print(f"--- {split.upper()} ---")
            print(f"Found {len(pairs)} image/label pairs")
            
            if pairs:
                img_path, gt_path = pairs[0]
                print(f"Example Image: {img_path}")
                print(f"Example GT:    {gt_path}")
                
                # Verify Loading
                lbl_raw = load_label(gt_path, convert_to_train_id=False)
                lbl_mapped = load_label(gt_path, convert_to_train_id=True)
                
                print(f"Raw IDs present in image: {np.unique(lbl_raw)}")
                print(f"Mapped TrainIDs present:  {np.unique(lbl_mapped)}")
                print("(Expected: 1=Road, 2=Hazard, 0=NonHazard, 255=Ignore)")
                
        except FileNotFoundError as e:
            print(f"Skipping {split}: {e}")