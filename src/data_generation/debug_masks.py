import numpy as np
from PIL import Image
from pathlib import Path

# Path to your generated masks
MASK_DIR = Path("data/synthetic_model_c/masks")

def inspect_masks():
    masks = list(MASK_DIR.glob("*.png"))
    if not masks:
        print("No masks found in data/synthetic_model_c/masks!")
        return

    print(f"Found {len(masks)} masks. Checking the first 3...\n")

    for mask_path in masks[:3]:
        # Load mask
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img)

        print(f"--- Checking {mask_path.name} ---")
        print(f"  Shape: {mask_np.shape}")
        print(f"  Dtype: {mask_np.dtype}")
        print(f"  Unique Values: {np.unique(mask_np)}")
        
        # Check if ID 2 exists
        if 2 in np.unique(mask_np):
            count = np.sum(mask_np == 2)
            print(f"  ✅ SUCCESS: Found {count} pixels with ID 2 (Hazard).")
        else:
            print(f"  ❌ FAILURE: ID 2 is MISSING from this mask.")
        print("-" * 30)

if __name__ == "__main__":
    inspect_masks()