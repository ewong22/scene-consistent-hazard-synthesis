import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Reuse your VLM because Model B also needs to know WHERE to paste!
# (Or we can use random placement if you want "Naive Copy-Paste", 
# but "Scene-Consistent" usually implies using the VLM for placement at least).
from src.data_generation.vlm_module import LocalVLM

# Reuse ID map for saving masks
from src.datasets.lost_and_found import get_laf_id_map

INPUT_DIR = Path("data/raw/lostandfound/leftImg8bit/train")
GT_DIR = Path("data/raw/lostandfound/gtCoarse/train")
BANK_DIR = Path("data/object_bank")
OUTPUT_DIR = Path("data/synthetic_model_b")

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

def parse_box(vlm_text, w, h):
    import re
    nums = [int(s) for s in re.findall(r'\d+', vlm_text)]
    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return None

def main():
    print("--- Starting Pipeline Model B (Copy-Paste) ---")
    
    # 1. Load Resources
    object_paths = list(BANK_DIR.glob("*.png"))
    if not object_paths:
        print("Error: Object bank is empty! Run create_object_bank.py first.")
        return
        
    vlm = LocalVLM() # Using VLM only for location detection
    print("VLM Loaded for location detection.")
    
    all_images = list(INPUT_DIR.rglob("*.png"))
    
    for img_path in tqdm(all_images):
        try:
            filename = img_path.name
            stem = img_path.stem 
            city_dir = img_path.parent.name
            
            # Find original mask
            gt_filename = filename.replace("leftImg8bit.png", "gtCoarse_labelIds.png")
            gt_path = GT_DIR / city_dir / gt_filename
            if not gt_path.exists(): continue

            # 2. Get Location
            # We only need 1 variation for Model B usually
            loc_text, _ = vlm.analyze_scene(str(img_path), num_variations=1)
            
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            box = parse_box(loc_text, W, H)
            
            if not box: continue
            x1, y1, x2, y2 = box
            target_w = x2 - x1
            target_h = y2 - y1
            
            if target_w < 10 or target_h < 10: continue

            # 3. Pick Random Object & Paste
            obj_path = random.choice(object_paths)
            obj_img = Image.open(obj_path).convert("RGBA")
            
            # Resize object to fit the box
            # Maintain aspect ratio to avoid squashing
            obj_ratio = obj_img.width / obj_img.height
            box_ratio = target_w / target_h
            
            if obj_ratio > box_ratio:
                # Width constrained
                new_w = target_w
                new_h = int(new_w / obj_ratio)
            else:
                # Height constrained
                new_h = target_h
                new_w = int(new_h * obj_ratio)
                
            obj_resized = obj_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center in box
            paste_x = x1 + (target_w - new_w) // 2
            paste_y = y1 + (target_h - new_h) // 2
            
            # Paste onto Image
            final_img = image.copy()
            final_img.paste(obj_resized, (paste_x, paste_y), obj_resized)
            
            # Paste onto Mask
            # Load original mask
            orig_mask = Image.open(gt_path)
            final_mask_np = np.array(orig_mask).copy()
            
            # We need the alpha channel of the pasted object to update the mask exactly
            alpha = np.array(obj_resized)[:, :, 3] # Get alpha channel
            
            # Create a 2D boolean mask where alpha > 0 (visible pixels)
            obj_visible = alpha > 128
            
            # Define ROI in the main mask
            roi = final_mask_np[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
            
            # Update only visible pixels to ID 2 (Hazard)
            roi[obj_visible] = 2 
            final_mask_np[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = roi
            
            # 4. Save
            final_img.save(OUTPUT_DIR / "images" / f"{stem}_cp.png")
            Image.fromarray(final_mask_np).save(OUTPUT_DIR / "masks" / f"{stem}_cp_gtCoarse_labelIds.png")
            
        except Exception as e:
            print(f"Error on {img_path.name}: {e}")
            continue

if __name__ == "__main__":
    main()