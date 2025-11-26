import torch
import os
import folder_paths
import numpy as np
import inspect

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
                "compile_model": ("BOOLEAN", {"default": False}),
                "attention_backend": (["default", "flash", "flash_3"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "Z-Image-Turbo"

    def load_model(self, model_id, precision, compile_model, attention_backend):
        print(f"--- Z-Image: Checking for model {model_id} ---")
        
        # 1. Download Logic (ModelScope)
        try:
            local_dir = os.path.join(folder_paths.models_dir, "diffusers", "Z-Image-Turbo")
            if not os.path.exists(local_dir):
                print("--- Z-Image: Downloading model... ---")
                model_dir = snapshot_download(model_id, local_dir=local_dir)
            else:
                model_dir = local_dir
        except Exception as e:
            print(f"Fallback to remote load: {e}")
            model_dir = model_id

        # 2. Precision Logic
        if precision == "bf16": dtype = torch.bfloat16
        elif precision == "fp16": dtype = torch.float16
        else: dtype = torch.float32

        # 3. Load Pipeline
        pipe = ZImagePipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")

        # 4. Optimization Settings
        if attention_backend == "flash":
            try: pipe.transformer.set_attention_backend("flash")
            except: pass
        elif attention_backend == "flash_3":
             try: pipe.transformer.set_attention_backend("_flash_3")
             except: pass

        if compile_model:
            pipe.transformer.compile()

        return (pipe,)

class ZImageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ZIMAGE_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": "Young Chinese woman in red Hanfu..."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}), # Turbo defaults
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}), 
                "cfg_normalization": ("BOOLEAN", {"default": False}),
                "cfg_truncation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image-Turbo"

    def generate(self, pipe, prompt, width, height, steps, guidance_scale, seed, 
                 negative_prompt="", batch_size=1, max_sequence_length=512, 
                 cfg_normalization=False, cfg_truncation=1.0):
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        print(f"--- Z-Image: {width}x{height}, Steps: {steps}, Batch: {batch_size} ---")

        # Call pipeline with arguments
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,   # Maps to batch_size
            max_sequence_length=max_sequence_length, # Maps to Token Limit
            cfg_normalization=cfg_normalization, # Specific to Z-Image
            cfg_truncation=cfg_truncation,       # Specific to Z-Image
            generator=generator,
        )

        # Process Batch of Images
        images_list = []
        for img in output.images:
            image_np = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            images_list.append(image_tensor)

        return (torch.cat(images_list, dim=0),)
