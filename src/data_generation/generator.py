import torch
import cv2
import numpy as np
import re
import os
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HazardGenerator:
    def __init__(self):
        print("Loading Generative Models (ControlNet + SD Inpainting)...")
        
        # 1. Load ControlNet (Depth Guidance)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        # 2. Load Stable Diffusion Inpainting (Base Model)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            controlnet=controlnet, 
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
            variant="fp16" 
        )

        # Optimize
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload() 
        print("Generator Loaded Successfully.")

    def parse_box(self, vlm_text, image_width, image_height):
        """
        Robustly parses coordinates from VLM text.
        Handles:
          - Integers: [100, 200, 300, 400]
          - Floats:   [0.1, 0.2, 0.5, 0.6] (Normalizes to image size)
        """
        if not vlm_text:
            return None

        # Clean string: remove brackets, commas, and parentheses
        text = vlm_text.lower().replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").replace(",", " ")

        # Regex to find numbers (integers OR floats)
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        
        try:
            nums = [float(x) for x in matches]
        except Exception as e:
            print(f"[ERROR] parsing numbers in '{vlm_text}': {e}")
            return None

        # We need exactly 4 numbers for a box
        if len(nums) < 4:
            print(f"[WARN] VLM text did not contain 4 coordinates: '{vlm_text}'")
            return None

        # Heuristic: Take the first 4 numbers found
        x1, y1, x2, y2 = nums[:4]

        # AUTO-SCALE: If coordinates are small floats (0.0 - 1.0), scale them up
        if all(0.0 <= n <= 1.0 for n in [x1, y1, x2, y2]):
            x1 *= image_width
            x2 *= image_width
            y1 *= image_height
            y2 *= image_height

        # Convert to int and clamp (ensure values are within image bounds)
        x1 = int(max(0, min(x1, image_width)))
        y1 = int(max(0, min(y1, image_height)))
        x2 = int(max(0, min(x2, image_width)))
        y2 = int(max(0, min(y2, image_height)))

        # Validation: Ensure box has area (x2 must be > x1, y2 must be > y1)
        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] Invalid box dimensions parsed: {x1, y1, x2, y2} (Zero Area)")
            return None

        return (x1, y1, x2, y2)

    def generate_hazard(self, image_path, depth_map_path, vlm_box_text, prompt, output_path=None, debug=False):
        # 1. Load Data
        image = load_image(image_path).convert("RGB")
        depth_image = load_image(depth_map_path).convert("RGB") 
        W, H = image.size
        
        # 2. Create Mask
        coords = self.parse_box(vlm_box_text, W, H)
        
        if coords is None:
            print(f"[ERROR] Could not parse box '{vlm_box_text}'. Skipping generation.")
            # Return the original image (unmodified) to maintain pipeline flow
            return image 

        x1, y1, x2, y2 = coords
        
        if debug:
            print(f"[DEBUG] Box coordinates: {x1}, {y1}, {x2}, {y2}")

        # Create the binary mask (White=255 inside the box, Black=0 outside)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1) 
        mask_image = Image.fromarray(mask).convert("RGB")

        if debug:
            mask_image.save("debug_mask.png")

        # 3. Enhance Prompt
        enhanced_prompt = f"{prompt}"
        negative_prompt = "bad, deformed, ugly, bad anatomy"

        # 4. Resize inputs to 512x512 (Diffusion standard)
        proc_w, proc_h = 512, 512
        img_resized = image.resize((proc_w, proc_h))
        mask_resized = mask_image.resize((proc_w, proc_h))
        depth_resized = depth_image.resize((proc_w, proc_h))

        # 5. Run Diffusion
        with torch.no_grad():
            output = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=img_resized,
                mask_image=mask_resized,
                control_image=depth_resized,
                num_inference_steps=30,
                strength=1.0, # Full inpainting strength
                guidance_scale=10.0, # Increased for better prompt adherence (The final tuning fix)
                controlnet_conditioning_scale=0.1 # Low influence allows for new 3D objects
            ).images[0]

        # 6. Composite Back to Original Resolution
        output_resized = output.resize((W, H))
        
        orig_np = np.array(image)
        new_np = np.array(output_resized)
        mask_np = np.array(mask_image) / 255.0 # Scale mask to 0.0 or 1.0
        
        # Blend: Output only inside the mask, keep original outside
        final_np = (new_np * mask_np + orig_np * (1 - mask_np)).astype(np.uint8)
        final_image = Image.fromarray(final_np)
        
        if output_path:
            final_image.save(output_path)
            
        return final_image

if __name__ == "__main__":
    from src.data_generation.depth_module import DepthEstimator

    # --- STANDALONE TEST BLOCK (Ensures Depth Map MATCHES Image) ---
    img_path = "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000007_000170_leftImg8bit.png" 
    
    if not os.path.exists(img_path):
        print("ERROR: Image path not found. Please verify file structure.")
    else:
        # 1. Generate a FRESH, MATCHING depth map
        print("Generating matching depth map...")
        depth_est = DepthEstimator()
        depth_map = depth_est.estimate_depth(img_path)
        
        matching_depth_path = "debug_depth_matching.png"
        cv2.imwrite(matching_depth_path, depth_map)

        # 2. Run Generator
        print("Running Generator...")
        gen = HazardGenerator()
        
        # Mock VLM Output (Center of the road, normalized coordinates)
        mock_vlm_box = "[0.24, 0.43, 0.5, 0.75]" 
        mock_prompt = 'Red plastic crate'

        gen.generate_hazard(
            img_path, 
            matching_depth_path, 
            mock_vlm_box, 
            mock_prompt, 
            output_path="debug_result.png", 
            debug=True
        )
        print("Done! Check debug_result.png")