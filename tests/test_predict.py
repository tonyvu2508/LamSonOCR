import pytest
import torch
from pathlib import Path
from PIL import Image


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a dummy model checkpoint."""
    from model.crnn import CRNN
    from config.charset import Charset
    cs = Charset()
    model = CRNN(num_classes=cs.num_classes)
    path = tmp_path / "test_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "charset_size": cs.num_classes,
        "epoch": 0,
        "loss": 1.0,
    }, path)
    return path


def test_predictor_init(dummy_checkpoint):
    from inference.predict import OCRPredictor
    predictor = OCRPredictor(checkpoint_path=dummy_checkpoint, device="cpu")
    assert predictor is not None


def test_predictor_predict_pil(dummy_checkpoint):
    from inference.predict import OCRPredictor
    predictor = OCRPredictor(checkpoint_path=dummy_checkpoint, device="cpu")
    img = Image.new("L", (100, 32), color=200)
    result = predictor.predict(img)
    assert "text" in result
    assert "confidence" in result
    assert isinstance(result["text"], str)
    assert 0 <= result["confidence"] <= 1


def test_predictor_predict_path(dummy_checkpoint, tmp_path):
    from inference.predict import OCRPredictor
    predictor = OCRPredictor(checkpoint_path=dummy_checkpoint, device="cpu")
    img_path = tmp_path / "test.png"
    Image.new("L", (100, 32), color=200).save(img_path)
    result = predictor.predict(img_path)
    assert "text" in result


def test_predictor_batch(dummy_checkpoint):
    from inference.predict import OCRPredictor
    predictor = OCRPredictor(checkpoint_path=dummy_checkpoint, device="cpu")
    images = [Image.new("L", (100, 32), color=200) for _ in range(3)]
    results = predictor.predict_batch(images)
    assert len(results) == 3
    assert all("text" in r for r in results)
