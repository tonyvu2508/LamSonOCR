import torch
import pytest


def test_crnn_output_shape():
    from model.crnn import CRNN
    num_classes = 100
    model = CRNN(num_classes=num_classes)
    # Batch=2, Channels=1, H=32, W=128
    x = torch.randn(2, 1, 32, 128)
    out = model(x)
    # Output: (T, B, num_classes) where T = W / downscale_factor
    assert out.dim() == 3
    assert out.shape[1] == 2  # batch
    assert out.shape[2] == num_classes
    assert out.shape[0] > 0  # time steps


def test_crnn_variable_width():
    from model.crnn import CRNN
    model = CRNN(num_classes=50)
    x1 = torch.randn(1, 1, 32, 64)
    x2 = torch.randn(1, 1, 32, 256)
    out1 = model(x1)
    out2 = model(x2)
    assert out1.shape[0] < out2.shape[0]  # More time steps for wider image


def test_crnn_grayscale_and_rgb():
    from model.crnn import CRNN
    model_gray = CRNN(num_classes=50, img_channels=1)
    model_rgb = CRNN(num_classes=50, img_channels=3)
    x_gray = torch.randn(1, 1, 32, 128)
    x_rgb = torch.randn(1, 3, 32, 128)
    out_gray = model_gray(x_gray)
    out_rgb = model_rgb(x_rgb)
    assert out_gray.shape == out_rgb.shape


def test_crnn_log_probabilities():
    from model.crnn import CRNN
    model = CRNN(num_classes=50)
    x = torch.randn(1, 1, 32, 128)
    out = model(x)
    # Should be log probabilities (all values <= 0)
    assert (out <= 0).all()
    # Sum of exp should be ~1
    probs = torch.exp(out)
    sums = probs.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
