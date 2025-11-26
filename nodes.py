import torch
import os
import folder_paths
import numpy as np
from PIL import Image

# Attempt imports
try:
    from modelscope import snapshot_download, ZImagePipeline
except ImportError:
    print("Error: modelscope not installed. Please run: pip install modelscope")

class ZImageLoader:
    def __init__(self):
        self.pipe = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "compile_model": ("BOOLEAN", {"default": False}),
                "attention_backend": (["default", "flash", "flash_3"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "Z-Image-Turbo"

    def load_model(self, model_id, precision, cpu_offload, compile_model, attention_backend):
        print(f"--- Z-Image: Checking for model {model_id} ---")
        
        # 1. Download/Locate Model via ModelScope
        # We try to store it in ComfyUI/models/diffusers if possible, otherwise let modelscope handle cache
        try:
            local_dir = os.path.join(folder_paths.models_dir, "diffusers", "Z-Image-Turbo")
            if not os.path.exists(local_dir):
                print("--- Z-Image: Downloading model... this may take a while ---")
                model_dir = snapshot_download(model_id, local_dir=local_dir)
            else:
                model_dir = local_dir
                print(f"--- Z-Image: Found local model at {model_dir} ---")
        except Exception as e:
            print(f"Error downloading via modelscope: {e}. Trying default cache.")
            model_dir = model_id # Fallback to remote string

        # 2. Determine Dtype
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # 3. Load Pipeline
        print("--- Z-Image: Loading Pipeline ---")
        pipe = ZImagePipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        if not cpu_offload:
            pipe.to("cuda")

        # 4. Optional Backend Settings
        if attention_backend == "flash":
            try:
                pipe.transformer.set_attention_backend("flash")
            except Exception as e:
                print(f"Warning: Flash attention failed: {e}")
        elif attention_backend == "flash_3":
             try:
                pipe.transformer.set_attention_backend("_flash_3")
             except Exception as e:
                print(f"Warning: Flash attention 3 failed: {e}")

        # 5. Compilation
        if compile_model:
            print("--- Z-Image: Compiling model (First run will be slow) ---")
            pipe.transformer.compile()

        # 6. CPU Offloading
        if cpu_offload:
            pipe.enable_model_cpu_offload()

        return (pipe,)

class ZImageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ZIMAGE_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": "Young Chinese woman in red Hanfu, intricate embroidery..."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image-Turbo"

    def generate(self, pipe, prompt, width, height, steps, seed):
        
        # Set Generator
        generator = torch.Generator("cuda").manual_seed(seed)

        print(f"--- Z-Image: Generating ({width}x{height}) steps={steps} ---")
        
        # Inference
        # Note: guidance_scale is hardcoded to 0.0 as per model requirements
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0, 
            generator=generator,
        ).images[0]

        # Convert PIL to Tensor for ComfyUI
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        return (image_tensor,)
