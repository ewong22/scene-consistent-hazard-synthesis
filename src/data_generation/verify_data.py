import os
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/synthetic_model_c")
IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

def verify():
    # Get list of generated images
    images = list(IMG_DIR.glob("*.png"))
    if not images:
        print("No images found!")
        return

    # Pick 5 random samples
    samples = random.sample(images, min(5, len(images)))

    for img_path in samples:
        # Find corresponding mask
        mask_name = img_path.name.replace(".png", "_gtCoarse_labelIds.png")
        mask_path = MASK_DIR / mask_name
        
        if not mask_path.exists():
            print(f"Missing mask for {img_path.name}")
            continue

        # Load
        img = cv2.imread(str(img_path))
        mask = np.array(Image.open(mask_path))

        # Create overlay
        # ID 2 is the hazard (Green Crate / Traffic Cone / etc)
        # We will paint ID 2 pixels as RED
        overlay = img.copy()
        overlay[mask == 2] = [0, 0, 255]  # BGR format -> Red

        # Blend
        alpha = 0.5
        vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Save visualization
        out_name = f"verify_{img_path.name}"
        cv2.imwrite(out_name, vis)
        print(f"Saved {out_name} - Check this file!")

if __name__ == "__main__":
    verify()