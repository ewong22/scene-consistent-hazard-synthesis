import os
import cv2
import random
import numpy as np
from pathlib import Path
from PIL import Image
import re


# Import your modules
from src.data_generation.vlm_module import LocalVLM

# Configuration
# We use the same 'Hannover' image you used for previous tests
TEST_IMG_PATH = Path("data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000007_000170_leftImg8bit.png")
GT_DIR = Path("data/raw/lostandfound/gtCoarse/train")
BANK_DIR = Path("data/object_bank")

def parse_box(vlm_text, w, h):
    nums = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", vlm_text)]
    
    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        
        # If max value is <= 1.0, we assume they are normalized and scale them up
        if max(x1, y1, x2, y2) <= 1.0:
            print(f"[DEBUG] Detected normalized coordinates: {nums[:4]}")
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
        else:
            # They are already pixels, just convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return None

def main():
    print("--- Testing Model B (Copy-Paste) Single Example ---")

    # 1. Check Object Bank
    object_paths = list(BANK_DIR.glob("*.png"))
    if not object_paths:
        print("ERROR: Object bank is empty! You must run 'python -m src.data_generation.create_object_bank' first.")
        return
    print(f"Object Bank: Found {len(object_paths)} objects.")

    # 2. Check Test Image
    if not TEST_IMG_PATH.exists():
        print(f"ERROR: Could not find test image at {TEST_IMG_PATH}")
        return

    # 3. Initialize VLM (Only need it for location)
    print("Loading VLM to find a safe pasting spot...")
    vlm = LocalVLM()
    
    # 4. Run Analysis
    print(f"Analyzing {TEST_IMG_PATH.name}...")
    # We ask for 1 location
    loc_text, _ = vlm.analyze_scene(str(TEST_IMG_PATH), num_variations=1)
    print(f"VLM Suggested Location: {loc_text}")

    # 5. Parse Location
    image = Image.open(TEST_IMG_PATH).convert("RGB")
    W, H = image.size
    box = parse_box(loc_text, W, H)

    if not box:
        print("ERROR: Could not parse VLM output. Try running again.")
        return

    x1, y1, x2, y2 = box
    target_w = x2 - x1
    target_h = y2 - y1
    print(f"Target Box Width: {target_w}, Height: {target_h}")

    if target_w < 10 or target_h < 10:
        print("ERROR: Target box is too small!")
        return

    # 6. Pick Random Object & Paste
    obj_path = random.choice(object_paths)
    print(f"Selected Object to Paste: {obj_path.name}")
    
    obj_img = Image.open(obj_path).convert("RGBA")
    
    # Resize object to fit the box (maintaining aspect ratio)
    obj_ratio = obj_img.width / obj_img.height
    box_ratio = target_w / target_h
    
    if obj_ratio > box_ratio:
        new_w = target_w
        new_h = int(new_w / obj_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * obj_ratio)
        
    obj_resized = obj_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center in box
    paste_x = x1 + (target_w - new_w) // 2
    paste_y = y1 + (target_h - new_h) // 2
    
    # Paste onto Image
    final_img = image.copy()
    final_img.paste(obj_resized, (paste_x, paste_y), obj_resized)
    
    # 7. Update Mask
    # Find original mask path
    city_dir = TEST_IMG_PATH.parent.name
    gt_filename = TEST_IMG_PATH.name.replace("leftImg8bit.png", "gtCoarse_labelIds.png")
    gt_path = GT_DIR / city_dir / gt_filename
    
    if gt_path.exists():
        orig_mask = Image.open(gt_path)
        final_mask_np = np.array(orig_mask).copy()
        
        # Get alpha channel for precise masking
        alpha = np.array(obj_resized)[:, :, 3] 
        obj_visible = alpha > 128
        
        # Update mask
        roi = final_mask_np[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
        roi[obj_visible] = 2  # Set to Hazard ID
        final_mask_np[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = roi
        
        final_mask_img = Image.fromarray(final_mask_np)
        final_mask_img.save("test_model_b_mask.png")
        print("Saved updated mask to 'test_model_b_mask.png'")
    else:
        print("Warning: Original GT mask not found, skipping mask update.")

    # 8. Save Result
    final_img.save("test_model_b_result.png")
    print("\nSUCCESS! Saved result to 'test_model_b_result.png'")
    print("Check this image to see if the object was pasted correctly.")

if __name__ == "__main__":
    main()