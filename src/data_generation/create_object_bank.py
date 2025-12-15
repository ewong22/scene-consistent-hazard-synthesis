import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Use the ID map to know what is a hazard (ID 2)
from src.datasets.lost_and_found import get_laf_id_map

INPUT_IMG_DIR = Path("data/raw/lostandfound/leftImg8bit/train")
INPUT_GT_DIR = Path("data/raw/lostandfound/gtCoarse/train")
BANK_DIR = Path("data/object_bank")
BANK_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("--- Creating Object Bank for Model B (Copy-Paste) ---")
    
    # Get standard ID mapping
    id_map = get_laf_id_map()
    
    # List pairs
    # (We re-use logic from your dataset file essentially)
    all_images = list(INPUT_IMG_DIR.rglob("*.png"))
    
    count = 0
    
    for img_path in tqdm(all_images):
        try:
            # Find Label
            filename = img_path.name
            gt_filename = filename.replace("leftImg8bit.png", "gtCoarse_labelIds.png")
            city_dir = img_path.parent.name
            gt_path = INPUT_GT_DIR / city_dir / gt_filename
            
            if not gt_path.exists():
                continue
                
            # Load Data
            image = cv2.imread(str(img_path)) # BGR
            label = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            
            # Map raw IDs to TrainIDs
            # We want to extract anything that maps to TrainID 2 (Hazard)
            # Efficient mapping:
            lut = np.zeros(256, dtype=np.uint8)
            for k, v in id_map.items():
                lut[k] = v
            train_id_map = lut[label]
            
            # Create a binary mask of just hazards
            hazard_mask = (train_id_map == 2).astype(np.uint8)
            
            # Find individual objects (Contours)
            contours, _ = cv2.findContours(hazard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, cnt in enumerate(contours):
                # Filter tiny noise
                if cv2.contourArea(cnt) < 500: # Min 500 pixels to be a useful cutout
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Extract the cutout
                roi = image[y:y+h, x:x+w]
                mask_roi = hazard_mask[y:y+h, x:x+w]
                
                # Refine mask to be exact to the contour (remove neighbors in the box)
                # Create a clean mask for just this contour
                clean_mask = np.zeros_like(hazard_mask)
                cv2.drawContours(clean_mask, [cnt], -1, 1, thickness=-1)
                clean_mask_roi = clean_mask[y:y+h, x:x+w]
                
                # Make transparent background (RGBA)
                b, g, r = cv2.split(roi)
                alpha = (clean_mask_roi * 255).astype(np.uint8)
                rgba = cv2.merge([b, g, r, alpha])
                
                # Save
                obj_name = f"hazard_{count}.png"
                cv2.imwrite(str(BANK_DIR / obj_name), rgba)
                count += 1
                
        except Exception as e:
            print(f"Skipping {img_path.name}: {e}")
            
    print(f"\nSuccessfully extracted {count} objects to {BANK_DIR}")

if __name__ == "__main__":
    main()