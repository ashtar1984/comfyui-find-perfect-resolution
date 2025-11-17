# find_perfect_resolution.py
# Version 0.8.0
import math
import torch
import numpy as np
from PIL import Image

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
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("width", "height", "IMAGE", "resolution_info")
    FUNCTION = "calculate"
    CATEGORY = "utils"

    def calculate(self, image, desired_width, desired_height, divisible_by,
                  upscale=False, upscale_method="lanczos",
                  small_image_mode="none", pad_color="#000000",
                unique_id=None): 

        _, orig_h, orig_w, _ = image.shape
        aspect_ratio = orig_w / orig_h

        # --- Auto calcul si 0 ---
        if desired_width == 0 and desired_height == 0:
            raise ValueError("desired_width et desired_height ne peuvent PAS être tous les deux à 0.")
        if desired_width == 0:
            desired_width = int(desired_height * aspect_ratio)
        if desired_height == 0:
            desired_height = int(desired_width / aspect_ratio)

        # --- Calcul résolution divisible ---
        num_pixels = desired_width * desired_height
        h_float = math.sqrt((num_pixels * orig_h) / orig_w)
        new_h = max(divisible_by, round(h_float / divisible_by) * divisible_by)
        new_w = max(divisible_by, round((aspect_ratio * h_float) / divisible_by) * divisible_by)

        method_map = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }
        resize_method = method_map.get(upscale_method, Image.LANCZOS)

        results = []
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            is_upscale = new_w > orig_w or new_h > orig_h
            do_resize = not is_upscale or (is_upscale and upscale)

            if do_resize:
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

        # --- Progress UI comme Kijai ---
        if unique_id is not None:
            try:
                num_elements = image_out.numel()
                element_size = image_out.element_size()
                memory_size_mb = (num_elements * element_size) / (1024*1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td>"
                    f"<td><b>{image_out.shape[0]}</b> x <b>{image_out.shape[2]}</b> x <b>{image_out.shape[1]}</b> | {memory_size_mb:.2f}MB</td></tr>",
                    unique_id
                )
            except Exception:
                pass

        return int(new_w), int(new_h), image_out, f"{new_w}x{new_h}"

    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) if len(hex_color) == 6 else (0, 0, 0)

    # --- Affichage dans ComfyUI sous le node ---
    def display(self, width, height, resolution_info="", **kwargs):
        return resolution_info


