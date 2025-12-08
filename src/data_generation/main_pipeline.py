import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import your modules
from src.data_generation.vlm_module import LocalVLM
from src.data_generation.depth_module import DepthEstimator
from src.data_generation.generator import HazardGenerator
from src.datasets.lost_and_found import get_laf_id_map 

# Configuration
INPUT_DIR = Path("data/raw/lostandfound/leftImg8bit/train")
GT_DIR = Path("data/raw/lostandfound/gtCoarse/train")
OUTPUT_DIR = Path("data/synthetic_model_c")

# Create output structure
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "depths").mkdir(parents=True, exist_ok=True)

def main():
    print(f"--- Starting Synthetic Pipeline (Model C) ---")
    
    # 1. Initialize Modules
    try:
        vlm = LocalVLM()
        depth_estimator = DepthEstimator()
        generator = HazardGenerator()
        # Get the ID map so we can respect original labels (Hazard=2)
        id_map = get_laf_id_map() 
        print("All AI Modules Loaded Successfully.\n")
    except Exception as e:
        print(f"CRITICAL ERROR Loading Models: {e}")
        return

    # 2. List all training images
    all_images = list(INPUT_DIR.rglob("*.png"))
    print(f"Found {len(all_images)} images to process.")

    # 3. Main Loop
    for i, img_path in enumerate(tqdm(all_images)):
        try:
            filename = img_path.name
            stem = img_path.stem 
            city_dir = img_path.parent.name
            
            # --- FIND CORRESPONDING GT ---
            # Image:  Hannover_000000_000000_leftImg8bit.png
            # Label:  Hannover_000000_000000_gtCoarse_labelIds.png
            gt_filename = filename.replace("leftImg8bit.png", "gtCoarse_labelIds.png")
            gt_path = GT_DIR / city_dir / gt_filename
            
            if not gt_path.exists():
                # If no GT exists, we can't safely use this image for training
                # (unless we assume everything else is background, which is risky)
                # Let's skip or warn. For LostAndFound, they should exist.
                # print(f"Skipping {filename}: GT not found.")
                continue

            # --- Step A: Depth Estimation ---
            depth_save_path = OUTPUT_DIR / "depths" / f"{stem}_depth.png"
            if depth_save_path.exists():
                depth_map_path = str(depth_save_path)
            else:
                depth_map = depth_estimator.estimate_depth(str(img_path))
                cv2.imwrite(str(depth_save_path), depth_map)
                depth_map_path = str(depth_save_path)

            # --- Step B: VLM Scene Analysis ---
            target_location, hazards_list = vlm.analyze_scene(str(img_path), num_variations=3)
            
            if not target_location or "None" in target_location:
                continue

            # --- Step C: Generation Loop ---
            for idx, hazard_desc in enumerate(hazards_list):
                var_name = f"{stem}_var{idx+1}"
                out_img_path = OUTPUT_DIR / "images" / f"{var_name}.png"
                out_mask_path = OUTPUT_DIR / "masks" / f"{var_name}_gtCoarse_labelIds.png" # Naming convention

                # 1. Generate the synthetic image
                final_image = generator.generate_hazard(
                    str(img_path),
                    depth_map_path,
                    target_location,
                    hazard_desc
                )
                final_image.save(out_img_path)

                # 2. MERGE MASKS (The Fix)
                # Load original label
                orig_label = Image.open(gt_path)
                final_label_np = np.array(orig_label).copy()
                
                # Parse the new box
                W, H = final_image.size
                coords = generator.parse_box(target_location, W, H)
                
                if coords:
                    x1, y1, x2, y2 = coords
                    
                    # Update the label array: Set the box area to ID 2 (Hazard)
                    # Note: We are writing to the "Raw ID" map. 
                    # In your dataset loader, ID 2 maps to TrainID 2, so this is consistent.
                    # (Standard ID 2 = "Crate", which is a hazard)
                    final_label_np[y1:y2, x1:x2] = 2 

                    # Save the merged label
                    final_label_img = Image.fromarray(final_label_np)
                    final_label_img.save(out_mask_path)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print("\n--- Pipeline Complete! ---")
    print(f"Data generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()