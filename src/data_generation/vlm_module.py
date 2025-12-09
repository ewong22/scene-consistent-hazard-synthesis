import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import re

# 1. Configuration
# We use the 7B Instruct model which is smarter than the 2B version
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

class LocalVLM:
    def __init__(self):
        print(f"Loading VLM: {MODEL_ID}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load in 4-bit to save memory (fits on standard GPUs)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa" # Standard attention (stable)
            )
            
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
            print("VLM Loaded Successfully (Qwen2-VL 4-bit).")
        except Exception as e:
            print(f"Error loading VLM: {e}")
            raise e

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

        # Generate Location
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        loc_text = output_text.split("assistant")[-1].strip()

        # --- Task 2: Generate Hazards ---
        prompt_desc = (
            f"Select {num_variations} distinct objects that would be realistic hazards on this road."
            "Provide a numbered list."
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
                temperature=0.7 
            )
            
        desc_result = self.processor.batch_decode(generated_ids_desc, skip_special_tokens=True)[0]
        raw_text = desc_result.split("assistant")[-1].strip()
        
        # Parse List
        hazards_list = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', raw_text, re.DOTALL)
        if not hazards_list:
            hazards_list = [raw_text]
        hazards_list = [h.strip() for h in hazards_list][:num_variations]

        return loc_text, hazards_list

if __name__ == "__main__":
    test_image_path = "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000007_000170_leftImg8bit.png" 
    
    if os.path.exists(test_image_path):
        vlm = LocalVLM()
        print("\n--- Analyzing Scene ---")
        torch.cuda.empty_cache()
        loc, hazards = vlm.analyze_scene(test_image_path)
        print(f"Location: {loc}")
        print(f"Hazards: {hazards}")

# "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000060_leftImg8bit.png" 