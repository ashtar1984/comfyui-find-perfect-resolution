import math

class FindPerfectResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "desired_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 1
                }),
                "desired_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 1
                }),
                "divisible_by": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate_resolution"
    CATEGORY = "utils"

    def calculate_resolution(self, image, desired_width, desired_height, divisible_by):
        # Get original image dimensions
        _, orig_height, orig_width, _ = image.shape  # Assuming [batch, height, width, channels]

        # Calculate original aspect ratio
        aspect_ratio = orig_width / orig_height  # a/b where a=width, b=height

        # Calculate number of pixels (c = a * b)
        num_pixels = desired_width * desired_height

        # Calculate new height: round(sqrt((c * b) / a) / divisible_by) * divisible_by
        new_height = round(math.sqrt((num_pixels * orig_height) / orig_width) / divisible_by) * divisible_by

        # Calculate new width: round(((a / b) * sqrt((c * b) / a)) / divisible_by) * divisible_by
        new_width = round((aspect_ratio * math.sqrt((num_pixels * orig_height) / orig_width)) / divisible_by) * divisible_by

        return (int(new_width), int(new_height))