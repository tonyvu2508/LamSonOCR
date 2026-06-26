"""Data augmentation transforms for OCR images."""
from PIL import Image, ImageFilter, ImageEnhance
import random


class OCRAugmentation:
    """Apply random augmentations to OCR text images."""

    def __init__(
        self,
        blur_prob: float = 0.2,
        noise_prob: float = 0.2,
        brightness_prob: float = 0.3,
        contrast_prob: float = 0.3,
        rotation_max: float = 2.0,
        rotation_prob: float = 0.2,
    ):
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.rotation_max = rotation_max
        self.rotation_prob = rotation_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        # Random rotation (slight)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.rotation_max, self.rotation_max)
            img = img.rotate(angle, fillcolor=255, expand=False)

        # Random brightness
        if random.random() < self.brightness_prob:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Random contrast
        if random.random() < self.contrast_prob:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Random blur
        if random.random() < self.blur_prob:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img
