from .nodes import ZImageLoader, ZImageSampler

NODE_CLASS_MAPPINGS = {
    "ZImageLoader": ZImageLoader,
    "ZImageSampler": ZImageSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoader": "Z-Image Turbo Loader (ModelScope)",
    "ZImageSampler": "Z-Image Turbo Sampler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
