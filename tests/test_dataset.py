import pytest
import torch
from pathlib import Path
import csv
import os
from PIL import Image


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small sample dataset for testing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    labels = []

    test_texts = ["123", "ABC", "R03 02月", "5"]
    for i, text in enumerate(test_texts):
        img = Image.new("L", (100, 32), color=200)
        img_path = img_dir / f"{i:05d}.png"
        img.save(img_path)
        labels.append({"image": str(img_path), "text": text})

    label_file = tmp_path / "labels.csv"
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(labels)

    return tmp_path


def test_dataset_length(sample_dataset):
    from data.dataset import OCRDataset
    from config.charset import Charset
    cs = Charset()
    ds = OCRDataset(root_dir=sample_dataset, charset=cs)
    assert len(ds) == 4


def test_dataset_item_types(sample_dataset):
    from data.dataset import OCRDataset
    from config.charset import Charset
    cs = Charset()
    ds = OCRDataset(root_dir=sample_dataset, charset=cs)
    img, label, text = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, list)
    assert isinstance(text, str)


def test_dataset_image_shape(sample_dataset):
    from data.dataset import OCRDataset
    from config.charset import Charset
    cs = Charset()
    ds = OCRDataset(root_dir=sample_dataset, charset=cs, img_height=32)
    img, _, _ = ds[0]
    assert img.shape[0] == 1  # Channels
    assert img.shape[1] == 32  # Height
    assert img.shape[2] > 0  # Width


def test_dataset_collate(sample_dataset):
    from data.dataset import OCRDataset, ocr_collate_fn
    from config.charset import Charset
    cs = Charset()
    ds = OCRDataset(root_dir=sample_dataset, charset=cs)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=ocr_collate_fn)
    batch = next(iter(loader))
    images, labels, label_lengths, texts = batch
    assert images.dim() == 4  # (B, C, H, W)
    assert images.shape[0] == 2
