"""End-to-end test: generate → train → evaluate → predict."""
import pytest
from pathlib import Path
from PIL import Image
import csv


def test_e2e_pipeline(tmp_path):
    """Test the complete pipeline with tiny data."""
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset
    from config.settings import Settings
    from data.dataset import OCRDataset
    from training.train import Trainer
    from benchmark.evaluate import Evaluator
    from inference.predict import OCRPredictor

    cs = Charset()

    # Step 1: Generate synthetic data
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    generate_synthetic_dataset(output_dir=train_dir, num_samples=30, charset=cs)
    generate_synthetic_dataset(output_dir=val_dir, num_samples=10, charset=cs)

    # Step 2: Train
    settings = Settings(
        device="cpu",
        num_epochs=2,
        batch_size=4,
        early_stop_patience=100,
    )
    settings.model_dir = tmp_path / "checkpoints"

    train_ds = OCRDataset(root_dir=train_dir, charset=cs)
    val_ds = OCRDataset(root_dir=val_dir, charset=cs)

    trainer = Trainer(charset=cs, settings=settings)
    history = trainer.train(train_ds, val_ds)
    assert len(history["train_loss"]) == 2

    # Step 3: Evaluate
    checkpoint_path = settings.model_dir / "best_model.pt"
    assert checkpoint_path.exists()

    evaluator = Evaluator(
        model=trainer.model,
        charset=cs,
        device="cpu",
    )
    results = evaluator.evaluate(val_ds)
    assert "cer" in results
    assert "sequence_accuracy" in results
    assert results["num_samples"] == 10

    # Step 4: Predict
    predictor = OCRPredictor(checkpoint_path=checkpoint_path, device="cpu")
    test_img = Image.new("L", (100, 32), color=200)
    result = predictor.predict(test_img)
    assert "text" in result
    assert "confidence" in result
