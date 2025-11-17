# Version: 0.6.0
# Auteur: ashtar1984 + ChatGPT édition premium
# Nouveautés :
# - Affichage automatique de la résolution calculée sous les paramètres
# - Calcul auto si l’un des deux est 0 (ratio respecté)
# - Comportement identique, juste plus intelligent

import math
import torch
import numpy as np
from PIL import Image, ImageOps


class FindPerfectResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "desired_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "desired_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "upscale": ("BOOLEAN", {"default": False}),
                "upscale_method": (["lanczos", "bilinear", "bicubic", "nearest"], {"default": "lanczos"}),
                "small_image_mode": (["none", "crop", "pad"], {"default": "none"}),
                "pad_color": ("STRING", {"default": "#000000"}),
                "skip_if_smaller": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("width", "height", "IMAGE", "resolution_info")
    FUNCTION = "calculate"
    CATEGORY = "utils"

    # ----- Méthode principale -----
    def calculate(self, image, desired_width, desired_height, divisible_by,
                  upscale=False, upscale_method="lanczos",
                  small_image_mode="none", pad_color="#000000",
                  skip_if_smaller=True):

        # Dimensions originales (B, H, W, C)
        _, orig_h, orig_w, _ = image.shape
        aspect_ratio = orig_w / orig_h

        # ----- AUTO CALCUL W/H -----
        if desired_width == 0 and desired_height == 0:
            raise ValueError("desired_width et desired_height ne peuvent PAS être tous les deux à 0.")

        if desired_width == 0:
            desired_width = int(desired_height * aspect_ratio)

        if desired_height == 0:
            desired_height = int(desired_width / aspect_ratio)

        # Pixel target
        num_pixels = desired_width * desired_height

        # ----- Calcul résolution divisible -----
        h_float = math.sqrt((num_pixels * orig_h) / orig_w)
        new_h = round(h_float / divisible_by) * divisible_by
        new_h = max(divisible_by, new_h)

        new_w = round((aspect_ratio * h_float) / divisible_by) * divisible_by
        new_w = max(divisible_by, new_w)

        # TEXT info affichée dans le node
        resolution_info = f"{new_w}x{new_h}"

        # ----- Si pas d’upscale -----
        if not upscale:
            return (int(new_w), int(new_h), image, resolution_info)

        # ----- Skip si plus petit -----
        if skip_if_smaller and (orig_w < new_w or orig_h < new_h):
            return (int(new_w), int(new_h), image, resolution_info)

        # Méthode resize PIL
        method_map = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }
        resize_method = method_map.get(upscale_method, Image.LANCZOS)

        # ----- Upscale -----
        results = []
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Traitement small image mode
            if small_image_mode != "none" and (pil_img.width < new_w or pil_img.height < new_h):
                target_ar = new_w / new_h
                img_ar = pil_img.width / pil_img.height

                if small_image_mode == "crop":
                    if img_ar > target_ar:
                        tmp_h = new_h
                        tmp_w = int(tmp_h * img_ar)
                    else:
                        tmp_w = new_w
                        tmp_h = int(tmp_w / img_ar)
                    pil_img = pil_img.resize((tmp_w, tmp_h), resize_method)

                    left = (pil_img.width - new_w) // 2
                    top = (pil_img.height - new_h) // 2
                    pil_img = pil_img.crop((left, top, left + new_w, top + new_h))

                elif small_image_mode == "pad":
                    pil_img.thumbnail((new_w, new_h), resize_method)
                    bg = Image.new("RGB", (new_w, new_h), self._hex_to_rgb(pad_color))
                    offset = ((new_w - pil_img.width) // 2, (new_h - pil_img.height) // 2)
                    bg.paste(pil_img, offset)
                    pil_img = bg
            else:
                pil_img = pil_img.resize((new_w, new_h), resize_method)

            img_np = np.array(pil_img).astype(np.float32) / 255.0
            results.append(img_np)

        image_out = torch.from_numpy(np.stack(results)).to(image.device)

        return (int(new_w), int(new_h), image_out, resolution_info)

    # ----- Convertisseur hex -> RGB -----
    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # ----- Affichage auto dans le node -----
    def display(self, width, height, **kwargs):
        return f"{width}x{height}"
