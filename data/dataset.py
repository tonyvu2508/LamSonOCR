"""PyTorch Dataset for OCR text line images."""
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class OCRDataset(Dataset):
    """Dataset for text line recognition.

    Expects:
        root_dir/
            labels.csv  (columns: image, text)
            images/
                00001.png
                ...
    """

    def __init__(
        self,
        root_dir: Path,
        charset,
        img_height: int = 32,
        img_max_width: int = 256,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.charset = charset
        self.img_height = img_height
        self.img_max_width = img_max_width
        self.transform = transform

        # Load labels
        self.samples = []
        label_file = self.root_dir / "labels.csv"
        with open(label_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image": row["image"],
                    "text": row["text"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["image"]
        text = sample["text"]

        # Load and preprocess image
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        # Resize height to img_height, keep aspect ratio
        w, h = img.size
        new_w = max(1, int(w * self.img_height / h))
        new_w = min(new_w, self.img_max_width)
        img = img.resize((new_w, self.img_height), Image.BILINEAR)

        # Convert to tensor and normalize
        img_tensor = TF.to_tensor(img)  # (1, H, W), values in [0, 1]

        # Encode label
        encoded = self.charset.encode(text)

        return img_tensor, encoded, text


def ocr_collate_fn(batch):
    """Custom collate function that pads images to same width.

    Returns:
        images: (B, C, H, max_W) padded tensor
        labels: (sum_of_label_lengths,) flattened tensor
        label_lengths: (B,) tensor
        texts: list[str] original texts
    """
    images, labels, texts = zip(*batch)

    # Find max width
    max_w = max(img.shape[2] for img in images)

    # Pad images to max width
    padded_images = []
    for img in images:
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            # Pad with white background (1.0) on the right
            padded = torch.nn.functional.pad(img, (0, pad_w), value=1.0)
        else:
            padded = img
        padded_images.append(padded)

    images_tensor = torch.stack(padded_images)
    labels_flat = torch.IntTensor([idx for label in labels for idx in label])
    label_lengths = torch.IntTensor([len(label) for label in labels])

    return images_tensor, labels_flat, label_lengths, list(texts)
