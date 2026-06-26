import pytest
import torch
from pathlib import Path
from PIL import Image
import csv


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create tiny dataset for training tests."""
    for split in ["train", "val"]:
        split_dir = tmp_path / split
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True)
        labels = []
        for i in range(20):
            text = str(i % 10)  # Simple digits
            img = Image.new("L", (64, 32), color=200)
            img_path = img_dir / f"{i:05d}.png"
            img.save(img_path)
            labels.append({"image": str(img_path), "text": text})
        with open(split_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "text"])
            writer.writeheader()
            writer.writerows(labels)
    return tmp_path


def test_trainer_runs_one_epoch(tiny_dataset):
    from training.train import Trainer
    from config.charset import Charset
    from config.settings import Settings
    from data.dataset import OCRDataset

    cs = Charset()
    settings = Settings(device="cpu", num_epochs=1, batch_size=4)
    settings.model_dir = tiny_dataset / "checkpoints"

    train_ds = OCRDataset(root_dir=tiny_dataset / "train", charset=cs)
    val_ds = OCRDataset(root_dir=tiny_dataset / "val", charset=cs)

    trainer = Trainer(charset=cs, settings=settings)
    history = trainer.train(train_ds, val_ds)

    assert "train_loss" in history
    assert len(history["train_loss"]) == 1


def test_trainer_saves_checkpoint(tiny_dataset):
    from training.train import Trainer
    from config.charset import Charset
    from config.settings import Settings
    from data.dataset import OCRDataset

    cs = Charset()
    settings = Settings(device="cpu", num_epochs=2, batch_size=4)
    settings.model_dir = tiny_dataset / "checkpoints"

    train_ds = OCRDataset(root_dir=tiny_dataset / "train", charset=cs)
    trainer = Trainer(charset=cs, settings=settings)
    trainer.train(train_ds, val_dataset=None)

    assert (settings.model_dir / "best_model.pt").exists()
