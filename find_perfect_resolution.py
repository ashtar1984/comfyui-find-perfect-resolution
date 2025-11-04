# find_perfect_resolution_v0.1.py
# Version: 0.1
# Auteur: ashtar1984 + amélioration Grok
# Description: Calcule la résolution parfaite (divisible par N) en conservant le ratio,
#              et optionnellement upscale l'image avec méthode choisie (Lanczos par défaut).

import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class FindPerfectResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "desired_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 1
                }),
                "desired_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 1
                }),
                "divisible_by": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
            },
            "optional": {
                "upscale": ("BOOLEAN", {"default": False}),
                "upscale_method": (["lanczos", "bilinear", "bicubic", "nearest"], {
                    "default": "lanczos"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE")
    RETURN_NAMES = ("width", "height", "image_upscaled")
    FUNCTION = "calculate_resolution"
    CATEGORY = "utils"

    def calculate_resolution(self, image, desired_width, desired_height, divisible_by,
                            upscale=False, upscale_method="lanczos"):
        # --- Récupérer les dimensions originales ---
        _, orig_height, orig_width, _ = image.shape  # [B, H, W, C]

        aspect_ratio = orig_width / orig_height
        num_pixels = desired_width * desired_height

        # --- Calculer la nouvelle hauteur ---
        new_height_float = math.sqrt((num_pixels * orig_height) / orig_width)
        new_height = round(new_height_float / divisible_by) * divisible_by
        new_height = max(divisible_by, new_height)  # éviter taille nulle

        # --- Calculer la nouvelle largeur ---
        new_width_float = aspect_ratio * new_height_float
        new_width = round(new_width_float / divisible_by) * divisible_by
        new_width = max(divisible_by, new_width)

        # --- Préparer la sortie image (si demandée) ---
        image_upscaled = None
        if upscale:
            # Convertir en PIL pour resize
            pil_images = []
            for i in range(image.shape[0]):
                img_np = image[i].cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Méthode de resize
                method_map = {
                    "lanczos": Image.LANCZOS,
                    "bilinear": Image.BILINEAR,
                    "bicubic": Image.BICUBIC,
                    "nearest": Image.NEAREST,
                }
                resize_method = method_map.get(upscale_method, Image.LANCZOS)
                
                resized = pil_img.resize((new_width, new_height), resample=resize_method)
                pil_images.append(resized)

            # Retour au format ComfyUI [B, H, W, C] en float32 [0,1]
            stacked = np.stack([np.array(img).astype(np.float32) / 255.0 for img in pil_images])
            image_upscaled = torch.from_numpy(stacked).to(image.device)

        return (int(new_width), int(new_height), image_upscaled)
