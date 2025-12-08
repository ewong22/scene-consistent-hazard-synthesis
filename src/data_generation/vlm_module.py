import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import re

# 1. Configuration
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

HAZARD_CATEGORIES = [
    # Standard Objects
    "Black Crate", "Stacked Black Crates", "Upright Black Crate",
    "Grey Crate", "Stacked Grey Crates", "Upright Grey Crate",
    "Blue Crate", "Small Blue Crate", "Green Crate", "Small Green Crate",
    # Random Hazards
    "Car Bumper", "Cardboard Box", "Car Exhaust Pipe", "Car Headlight", 
    "Euro Wooden Pallet", "Rearview Mirror", "Black Tire",
    "Bloated Plastic Bag", "Styrofoam Block",
    # Pylons
    "Orange Traffic Pylon", "Large Traffic Pylon", "White Traffic Pylon",
    # Emotional/Toys
    "Soccer Ball", "Bicycle", "Kid Dummy", 
    "Grey Bobby Car Toy", "Red Bobby Car Toy", "Yellow Bobby Car Toy",
    # Animals
    "Black Dog Replica", "White Dog Replica"
]

class LocalVLM:
    def __init__(self):
        print(f"Loading VLM: {MODEL_ID}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
        print("VLM Loaded Successfully (Qwen2-VL 4-bit).")

    def analyze_scene(self, image_path, num_variations=3):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # --- Task 1: Find valid insertion region ---
        prompt_loc = (
            "Analyze this road scene. Identify a small, EMPTY patch of asphalt road where I can place a new object. "
            "CRITICAL: The area must be completely clear. "
            "It must NOT overlap with any people, cars, existing boxes, balls, or other debris. "
            "Return only the bounding box coordinates [x1, y1, x2, y2]."
        )
        
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_loc}]}
        ]
        
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        # Retry logic handled by helper
        loc_result = self._generate_location(inputs, width, height)

        # --- Task 2: Generate LIST of Hazard Descriptions ---
        options_str = ", ".join(HAZARD_CATEGORIES)
        
        prompt_desc = (
            f"Review this list of hazards: [{options_str}]. "
            f"Select {num_variations} DISTINCT items that would look realistic in this specific scene. "
            "Provide a numbered list (1., 2., 3.) with a short visual description for each. "
            "Example format:\n1. A dirty grey crate\n2. A small black tire"
        )
        
        messages_desc = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_desc}]}
        ]
        
        text_input_desc = self.processor.apply_chat_template(messages_desc, tokenize=False, add_generation_prompt=True)
        image_inputs_desc, video_inputs_desc = process_vision_info(messages_desc)
        
        inputs_desc = self.processor(
            text=[text_input_desc], images=image_inputs_desc, videos=video_inputs_desc, padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids_desc = self.model.generate(
                **inputs_desc, 
                max_new_tokens=256,
                do_sample=True,      
                temperature=0.9,
                top_p=0.95           
            )
            
        desc_result = self.processor.batch_decode(generated_ids_desc, skip_special_tokens=True)[0]
        raw_text = desc_result.split("assistant")[-1].strip()
        
        hazards_list = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', raw_text, re.DOTALL)
        if not hazards_list:
            hazards_list = [raw_text]
        hazards_list = [h.strip() for h in hazards_list]

        return loc_result, hazards_list

    def _generate_location(self, inputs, width, height):
        """Helper to generate and validate location box."""
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        loc_text = output_text.split("assistant")[-1].strip()
        
        nums = [int(s) for s in re.findall(r'\d+', loc_text)]
        
        if len(nums) >= 4:
            x1, y1, x2, y2 = nums[:4]
            box_area = (x2 - x1) * (y2 - y1)
            img_area = width * height
            
            # REJECT if box covers > 50% of image (Stricter than before)
            if box_area > (img_area * 0.5):
                print(f"[WARN] VLM returned massive box ({loc_text}). Using fallback center crop.")
                cx, cy = width // 2, int(height * 0.75)
                w, h = 100, 100
                return f"({cx-w}, {cy-h}) to ({cx+w}, {cy+h})"
        
        return loc_text

if __name__ == "__main__":
    test_image_path = "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000012_000280_leftImg8bit.png" 
    
    if not os.path.exists(test_image_path):
        print(f"Error: Could not find image at {test_image_path}")
    else:
        vlm = LocalVLM()
        print("Analyzing scene...")
        location, detected_objects = vlm.analyze_scene(test_image_path, num_variations=5)
        
        print(f"\nTarget Location: {location}")
        print(f"Detected {len(detected_objects)} potential hazards:")
        for i, obj_desc in enumerate(detected_objects):
            print(f"  [Variation {i+1}]: {obj_desc}")

# "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000060_leftImg8bit.png" 