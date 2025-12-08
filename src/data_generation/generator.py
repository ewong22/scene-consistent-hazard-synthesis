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
        
        # 1. Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        # 2. Load Stable Diffusion Inpainting
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
        # Extract numbers
        nums = [int(s) for s in re.findall(r'\d+', vlm_text)]
        if len(nums) >= 4:
            x1, y1, x2, y2 = nums[:4]
            # Clamp
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            return (x1, y1, x2, y2)
        else:
            print("[WARN] Could not parse coordinates. Returning None.")
            return None

    def generate_hazard(self, image_path, depth_map_path, vlm_box_text, prompt, output_path=None, debug=True):
        # 1. Load Data
        image = load_image(image_path).convert("RGB")
        depth_image = load_image(depth_map_path).convert("RGB") 
        W, H = image.size
        
        # 2. Create Mask
        coords = self.parse_box(vlm_box_text, W, H)
        if coords is None:
            # Fallback for debug: Center box
            cx, cy = W // 2, int(H * 0.75)
            coords = (cx-50, cy-50, cx+50, cy+50)
            print(f"[DEBUG] Using fallback box: {coords}")

        x1, y1, x2, y2 = coords
        print(f"[DEBUG] Box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"[DEBUG] Box Area: {(x2-x1)*(y2-y1)} pixels")

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1) 
        mask_image = Image.fromarray(mask).convert("RGB")

        if debug:
            mask_image.save("debug_mask.png")
            print("[DEBUG] Saved 'debug_mask.png'. CHECK THIS FILE! It should be black with a white box.")

        # 3. Enhance Prompt
        enhanced_prompt = f"{prompt}, lying on asphalt road, realistic, 4k, soft shadows, photograph"
        negative_prompt = "floating, cartoon, unrealistic, drawing, bad quality"

        # 4. Resize
        proc_w, proc_h = 512, 512
        img_resized = image.resize((proc_w, proc_h))
        mask_resized = mask_image.resize((proc_w, proc_h))
        depth_resized = depth_image.resize((proc_w, proc_h))

        if debug:
            img_resized.save("debug_input_resized.png")

        # 5. Run Diffusion (TEST MODE: Weak ControlNet)
        # We set scale to 0.3 to let the object 'override' the depth map slightly
        print(f"Generating '{prompt}' with ControlNet Scale 0.3...")
        
        output = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            image=img_resized,
            mask_image=mask_resized,
            control_image=depth_resized,
            num_inference_steps=30,
            strength=1.0,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.3 # Reduced for visibility test
        ).images[0]

        # 6. Composite Back
        output_resized = output.resize((W, H))
        
        orig_np = np.array(image)
        new_np = np.array(output_resized)
        mask_np = np.array(mask_image) / 255.0 
        
        # Hard Blend for Debugging (To verify pixels actually changed)
        # If this works, you'll see the generated square clearly
        final_np = (new_np * mask_np + orig_np * (1 - mask_np)).astype(np.uint8)
        final_image = Image.fromarray(final_np)
        
        if output_path:
            final_image.save(output_path)
            print(f"Saved to {output_path}")
            
        return final_image

if __name__ == "__main__":
    # Test Inputs
    img_path = "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000004_000210_leftImg8bit.png" 
    depth_path = "test_depth_map.png"
    
    # FIX: Shift X to 600 (Left side) and Y to 600 (Further up road)
    # This avoids the Mercedes star at the bottom center (X=1024, Y=1000)
    mock_vlm_box = "(600, 600) to (750, 750)" 
    
    # High contrast object
    mock_prompt = "a bright red traffic cone"

    if not os.path.exists(img_path):
        print("ERROR: Image path not found. Please fix path in __main__")
    else:
        gen = HazardGenerator()
        # Run with debug=True to see the mask again
        gen.generate_hazard(img_path, depth_path, mock_vlm_box, mock_prompt, output_path="debug_result.png", debug=True)