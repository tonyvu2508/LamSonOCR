# LamSonOCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-accuracy CRNN+CTC OCR model from scratch that can recognize Japanese + English + numbers in cropped text line images, with a complete benchmark framework to measure and improve accuracy.

**Architecture:** CNN backbone (ResNet-like) → BiLSTM sequence encoder → FC + CTC decoder. Training uses synthetic data generated via TextRecognitionDataGenerator with Japanese fonts. Benchmark suite measures CER, WER, and Sequence Accuracy.

**Tech Stack:** Python 3.10+, PyTorch (MPS/CPU), trdg (TextRecognitionDataGenerator), Pillow, editdistance, pandas, matplotlib

## Global Constraints

- Python 3.10+ required
- PyTorch with MPS backend support (torch >= 2.0)
- All paths relative to `/Volumes/SpaceX/WorkSpace/python/LamSonOcr/`
- Image height fixed at 32px, width variable (max 256px)
- CTC blank token always index 0
- UTF-8 encoding throughout
- No GPU-specific code (use `torch.device` abstraction for MPS/CPU/CUDA)
- Use Python venv at `venv/` — all `python` and `pip` commands run via `source venv/bin/activate` or `venv/bin/python`

---

### Task 1: Project Setup & Character Set

**Files:**
- Create: `venv/` (Python virtual environment)
- Create: `config/__init__.py`
- Create: `config/charset.py`
- Create: `config/settings.py`
- Create: `requirements.txt`
- Test: `tests/test_charset.py`

**Interfaces:**
- Produces:
  - `Charset` class with `encode(text: str) -> list[int]`, `decode(indices: list[int]) -> str`, `num_classes: int`
  - `Settings` dataclass with `IMG_HEIGHT=32`, `IMG_MAX_WIDTH=256`, `BATCH_SIZE=64`, etc.

- [ ] **Step 1: Create venv and install dependencies**

```bash
cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr
python3 -m venv venv
source venv/bin/activate
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
trdg>=1.8.0
Pillow>=9.0.0
editdistance>=0.6.0
pandas>=1.5.0
matplotlib>=3.6.0
pytest>=7.0.0
tqdm>=4.60.0
```

- [ ] **Step 3: Install dependencies**

```bash
cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr
venv/bin/pip install -r requirements.txt
```

- [ ] **Step 4: Write failing test for Charset**

```python
# tests/test_charset.py
import pytest

def test_charset_init():
    from config.charset import Charset
    cs = Charset()
    assert cs.num_classes > 0
    assert cs.blank_idx == 0

def test_charset_encode_decode_digits():
    from config.charset import Charset
    cs = Charset()
    text = "123"
    encoded = cs.encode(text)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    decoded = cs.decode(encoded)
    assert decoded == text

def test_charset_encode_decode_japanese():
    from config.charset import Charset
    cs = Charset()
    text = "R03 02月"
    encoded = cs.encode(text)
    decoded = cs.decode(encoded)
    assert decoded == text

def test_charset_encode_unknown_char():
    from config.charset import Charset
    cs = Charset()
    # Unknown char should be skipped or mapped to UNK
    encoded = cs.encode("€")  # Euro sign not in charset
    assert isinstance(encoded, list)

def test_charset_decode_with_blanks():
    from config.charset import Charset
    cs = Charset()
    # CTC output often has blanks and repeats
    text = "AB"
    encoded = cs.encode(text)
    # Simulate CTC output: A, blank, blank, B
    ctc_output = [encoded[0], 0, 0, encoded[1]]
    decoded = cs.ctc_decode(ctc_output)
    assert decoded == text

def test_charset_decode_with_repeats():
    from config.charset import Charset
    cs = Charset()
    text = "AA"
    encoded = cs.encode(text)
    # CTC: A, A should collapse to "A", but A, blank, A should be "AA"
    ctc_no_sep = [encoded[0], encoded[0]]
    assert cs.ctc_decode(ctc_no_sep) == "A"
    ctc_with_sep = [encoded[0], 0, encoded[0]]
    assert cs.ctc_decode(ctc_with_sep) == "AA"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_charset.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 6: Implement config/charset.py**

```python
# config/__init__.py
# (empty)

# config/charset.py
"""Character set for LamSonOCR - Japanese + English + Numbers."""

class Charset:
    """Maps characters to indices for CTC-based OCR."""

    BLANK = "<BLANK>"

    # Define character groups
    DIGITS = "0123456789"
    ENGLISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ENGLISH_LOWER = "abcdefghijklmnopqrstuvwxyz"
    HIRAGANA = (
        "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        "はひふへほまみむめもやゆよらりるれろわをん"
        "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
        "ぁぃぅぇぉっゃゅょ"
    )
    KATAKANA = (
        "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
        "ハヒフヘホマミムメモヤユヨラリルレロワヲン"
        "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
        "ァィゥェォッャュョヴー"
    )
    KANJI_COMMON = (
        "年月日人車台号番出品初度登録定員内装外色型式"
        "車名走行距離評価点検査有効期間排気量燃料"
        "自動手動無段変速機前後左右上下新中古"
        "令和平成昭和大正"
        "万千百十一二三四五六七八九零"
        "東西南北海道府県市区町村"
        "通常特別限定"
    )
    SYMBOLS = " /-()[]・。、:;\"'.!?@#%&*+=~^_|\\<>{}"

    def __init__(self):
        # Build charset: BLANK at index 0
        chars = []
        for group in [
            self.DIGITS,
            self.ENGLISH_UPPER,
            self.ENGLISH_LOWER,
            self.HIRAGANA,
            self.KATAKANA,
            self.KANJI_COMMON,
            self.SYMBOLS,
        ]:
            for c in group:
                if c not in chars:
                    chars.append(c)

        self._idx_to_char = {0: self.BLANK}
        self._char_to_idx = {self.BLANK: 0}
        for i, c in enumerate(chars, start=1):
            self._idx_to_char[i] = c
            self._char_to_idx[c] = i

    @property
    def blank_idx(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        return len(self._idx_to_char)

    def encode(self, text: str) -> list[int]:
        """Encode text string to list of character indices."""
        result = []
        for c in text:
            if c in self._char_to_idx:
                result.append(self._char_to_idx[c])
            # Skip unknown characters
        return result

    def decode(self, indices: list[int]) -> str:
        """Decode list of indices to text string (no CTC logic)."""
        return "".join(
            self._idx_to_char.get(i, "") for i in indices if i != self.blank_idx
        )

    def ctc_decode(self, indices: list[int]) -> str:
        """Decode CTC output: remove blanks and collapse repeats."""
        result = []
        prev = None
        for idx in indices:
            if idx == self.blank_idx:
                prev = None
                continue
            if idx != prev:
                result.append(idx)
            prev = idx
        return self.decode(result)
```

- [ ] **Step 7: Implement config/settings.py**

```python
# config/settings.py
"""Training and model configuration."""
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Settings:
    # Paths
    project_root: Path = field(
        default_factory=lambda: Path("/Volumes/SpaceX/WorkSpace/python/LamSonOcr")
    )
    data_dir: Path = field(default=None)
    model_dir: Path = field(default=None)

    # Image
    img_height: int = 32
    img_max_width: int = 256
    img_channels: int = 1  # Grayscale

    # Model
    cnn_output_channels: int = 512
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.1

    # Training
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    early_stop_patience: int = 10
    num_workers: int = 0  # 0 for MPS compatibility

    # Device
    device: str = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.model_dir is None:
            self.model_dir = self.project_root / "checkpoints"
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_charset.py -v`
Expected: All 6 tests PASS

- [ ] **Step 9: Commit**

```bash
cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr
git init
git add config/ tests/test_charset.py requirements.txt
git commit -m "feat: add charset and settings config for LamSonOCR"
```

---

### Task 2: CRNN Model Architecture

**Files:**
- Create: `model/__init__.py`
- Create: `model/crnn.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `Charset.num_classes` from Task 1
- Produces:
  - `CRNN(num_classes: int, img_height: int = 32, img_channels: int = 1, rnn_hidden: int = 256, rnn_layers: int = 2) -> nn.Module`
  - Forward: `forward(x: Tensor[B, C, H, W]) -> Tensor[T, B, num_classes]` (log probabilities)

- [ ] **Step 1: Write failing test for CRNN**

```python
# tests/test_model.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement model/crnn.py**

```python
# model/__init__.py
# (empty)

# model/crnn.py
"""CRNN (Convolutional Recurrent Neural Network) for text line recognition."""
import torch
import torch.nn as nn


class _CNNBackbone(nn.Module):
    """Lightweight CNN feature extractor.

    Input:  (B, C, 32, W)
    Output: (B, 512, 1, W') where W' = W // 4
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, C, 32, W) -> (B, 64, 16, W/2)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (B, 64, 16, W/2) -> (B, 128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (B, 128, 8, W/4) -> (B, 256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: (B, 256, 8, W/4) -> (B, 256, 4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Only reduce height

            # Block 5: (B, 256, 4, W/4) -> (B, 512, 2, W/4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: (B, 512, 2, W/4) -> (B, 512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Only reduce height
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CRNN(nn.Module):
    """CRNN model for text line recognition with CTC.

    Architecture: CNN backbone -> BiLSTM -> FC -> LogSoftmax

    Args:
        num_classes: Number of character classes (including CTC blank)
        img_height: Input image height (default: 32)
        img_channels: Number of input channels (1=grayscale, 3=RGB)
        rnn_hidden: Hidden size of LSTM layers
        rnn_layers: Number of LSTM layers
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 32,
        img_channels: int = 1,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()
        assert img_height == 32, "CRNN requires img_height=32"

        self.cnn = _CNNBackbone(in_channels=img_channels)
        # After CNN: (B, 512, 1, W')
        # BiLSTM input: (B, W', 512)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0,
        )
        # BiLSTM output: (B, W', rnn_hidden*2)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Log probabilities (T, B, num_classes) where T = W // 4
        """
        # CNN features
        features = self.cnn(x)  # (B, 512, 1, W')

        # Squeeze height, permute to (B, W', 512)
        features = features.squeeze(2)  # (B, 512, W')
        features = features.permute(0, 2, 1)  # (B, W', 512)

        # BiLSTM
        rnn_out, _ = self.rnn(features)  # (B, W', hidden*2)

        # FC + LogSoftmax
        output = self.fc(rnn_out)  # (B, W', num_classes)
        output = self.log_softmax(output)

        # Permute to (T, B, num_classes) for CTC loss
        output = output.permute(1, 0, 2)  # (T, B, num_classes)

        return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_model.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/ tests/test_model.py
git commit -m "feat: implement CRNN model architecture"
```

---

### Task 3: Dataset & Data Loading

**Files:**
- Create: `data/__init__.py`
- Create: `data/dataset.py`
- Create: `data/augmentation.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `Charset` from Task 1, `Settings` from Task 1
- Produces:
  - `OCRDataset(root_dir: Path, charset: Charset, img_height: int, img_max_width: int, transform: Callable | None) -> Dataset`
  - `get_augmentation_transform() -> torchvision.transforms.Compose`
  - Each dataset item: `(image_tensor: Tensor[1, 32, W], encoded_label: list[int], label_text: str)`

- [ ] **Step 1: Write failing test for OCRDataset**

```python
# tests/test_dataset.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data/dataset.py**

```python
# data/__init__.py
# (empty)

# data/dataset.py
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
            # Pad with zeros on the right
            padded = torch.nn.functional.pad(img, (0, pad_w), value=0)
        else:
            padded = img
        padded_images.append(padded)

    images_tensor = torch.stack(padded_images)
    labels_flat = torch.IntTensor([idx for label in labels for idx in label])
    label_lengths = torch.IntTensor([len(label) for label in labels])

    return images_tensor, labels_flat, label_lengths, list(texts)
```

- [ ] **Step 4: Implement data/augmentation.py**

```python
# data/augmentation.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add data/ tests/test_dataset.py
git commit -m "feat: add dataset loader and augmentation"
```

---

### Task 4: Synthetic Data Generator

**Files:**
- Create: `scripts/generate_synthetic.py`
- Create: `data/dictionaries/ja_auction.txt`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `Charset` from Task 1
- Produces:
  - `generate_synthetic_dataset(output_dir: Path, num_samples: int, charset: Charset) -> None`
  - Generates `output_dir/images/` + `output_dir/labels.csv`

- [ ] **Step 1: Write failing test for synthetic generator**

```python
# tests/test_synthetic.py
import pytest
from pathlib import Path

def test_generate_samples(tmp_path):
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset
    cs = Charset()
    generate_synthetic_dataset(
        output_dir=tmp_path,
        num_samples=10,
        charset=cs,
        fonts_dir=None,  # Use system defaults
    )
    assert (tmp_path / "labels.csv").exists()
    assert (tmp_path / "images").is_dir()

    import csv
    with open(tmp_path / "labels.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10
    for row in rows:
        assert "image" in row
        assert "text" in row
        assert Path(row["image"]).exists()

def test_generated_images_valid(tmp_path):
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset
    from PIL import Image
    cs = Charset()
    generate_synthetic_dataset(output_dir=tmp_path, num_samples=5, charset=cs)

    import csv
    with open(tmp_path / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        img = Image.open(row["image"])
        assert img.size[0] > 0
        assert img.size[1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_synthetic.py -v`
Expected: FAIL

- [ ] **Step 3: Create domain-specific dictionary**

```python
# data/dictionaries/ja_auction.txt
# Each line is a text that might appear in auction sheets.
# Generated programmatically. File contains:
# - Registration dates: R01-R08, H01-H31, years/months
# - Numbers: 1-99, vehicle IDs
# - Door counts: 1,2,3,4,5
# - Seating: 2,4,5,6,7,8
# - Interior grades: A,B,C,D,E,S
# - Common kanji/words from auction sheets
```

- [ ] **Step 4: Implement scripts/generate_synthetic.py**

```python
# scripts/generate_synthetic.py
"""Generate synthetic text line images for OCR training.

Uses PIL to render text with various Japanese fonts.
Does NOT require trdg — we generate directly for better control.
"""
import csv
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Domain-specific text generators
def _random_registration_date() -> str:
    """Generate a Japanese vehicle registration date."""
    era = random.choice(["H", "R", "S"])
    year = random.randint(1, 31)
    month = random.randint(1, 12)
    formats = [
        f"{era}{year:02d} {month:02d}月",
        f"{era}{year} {month}月",
        f"{era}{year:02d}/{month:02d}",
        f"{era}{year}年{month}月",
    ]
    return random.choice(formats)


def _random_exhibit_number() -> str:
    """Generate auction exhibit numbers."""
    prefix = random.choice(["", f"[{random.randint(1000,9999)}] "])
    number = f"{random.randint(1, 99999):05d}"
    return f"{prefix}{number}"


def _random_door_count() -> str:
    return str(random.choice([2, 3, 4, 5]))


def _random_seating() -> str:
    return str(random.choice([2, 4, 5, 6, 7, 8]))


def _random_interior_grade() -> str:
    return random.choice(["A", "B", "C", "D", "E", "S"])


def _random_number_string() -> str:
    length = random.randint(1, 6)
    return "".join(random.choices(string.digits, k=length))


def _random_mixed_text() -> str:
    """Generate various text patterns."""
    generators = [
        _random_registration_date,
        _random_exhibit_number,
        _random_door_count,
        _random_seating,
        _random_interior_grade,
        _random_number_string,
    ]
    return random.choice(generators)()


def _get_system_fonts() -> list[str]:
    """Find available Japanese-capable fonts on the system."""
    font_paths = []
    search_dirs = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
    ]
    japanese_font_names = [
        "Hiragino", "YuGothic", "YuMincho", "Osaka",
        "HiraginoSans", "HiraginoSerif",
    ]

    for font_dir in search_dirs:
        if font_dir.exists():
            for f in font_dir.rglob("*.ttf"):
                font_paths.append(str(f))
            for f in font_dir.rglob("*.ttc"):
                font_paths.append(str(f))
            for f in font_dir.rglob("*.otf"):
                font_paths.append(str(f))

    # Filter for Japanese-capable fonts if possible
    ja_fonts = [f for f in font_paths if any(n.lower() in f.lower() for n in japanese_font_names)]
    if ja_fonts:
        return ja_fonts
    return font_paths[:10] if font_paths else []


def render_text_image(
    text: str,
    font_path: str | None = None,
    font_size: int = 24,
    img_height: int = 32,
) -> Image.Image:
    """Render text to a grayscale image.

    Args:
        text: Text to render
        font_path: Path to .ttf/.otf font file
        font_size: Font size in pixels
        img_height: Output image height

    Returns:
        Grayscale PIL Image
    """
    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Calculate text size
    dummy_img = Image.new("L", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 10  # padding
    text_h = bbox[3] - bbox[1] + 6

    # Create image with white background
    bg_color = random.randint(220, 255)
    img = Image.new("L", (max(text_w, 10), max(text_h, img_height)), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    text_color = random.randint(0, 50)
    x = random.randint(2, 5)
    y = max(0, (img.size[1] - text_h) // 2)
    draw.text((x, y), text, fill=text_color, font=font)

    # Resize to target height maintaining aspect ratio
    w, h = img.size
    new_w = max(1, int(w * img_height / h))
    img = img.resize((new_w, img_height), Image.BILINEAR)

    return img


def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int,
    charset=None,
    fonts_dir: str | None = None,
    img_height: int = 32,
) -> None:
    """Generate a synthetic OCR dataset.

    Creates:
        output_dir/images/00001.png, ...
        output_dir/labels.csv

    Args:
        output_dir: Directory to save generated dataset
        num_samples: Number of samples to generate
        charset: Charset instance (for validation)
        fonts_dir: Directory containing font files (None = use system fonts)
        img_height: Target image height
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Find fonts
    if fonts_dir and Path(fonts_dir).exists():
        fonts = list(Path(fonts_dir).rglob("*.ttf")) + list(Path(fonts_dir).rglob("*.otf"))
        fonts = [str(f) for f in fonts]
    else:
        fonts = _get_system_fonts()

    if not fonts:
        fonts = [None]  # Use default font

    samples = []
    for i in range(num_samples):
        text = _random_mixed_text()
        font_path = random.choice(fonts)
        font_size = random.randint(18, 28)

        img = render_text_image(
            text=text,
            font_path=font_path,
            font_size=font_size,
            img_height=img_height,
        )

        img_path = img_dir / f"{i:06d}.png"
        img.save(img_path)

        samples.append({
            "image": str(img_path),
            "text": text,
        })

    # Write labels
    label_file = output_dir / "labels.csv"
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"Generated {num_samples} samples in {output_dir}")


if __name__ == "__main__":
    import argparse
    from config.charset import Charset

    parser = argparse.ArgumentParser(description="Generate synthetic OCR data")
    parser.add_argument("--output", type=str, default="data/train")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--fonts-dir", type=str, default=None)
    args = parser.parse_args()

    cs = Charset()
    generate_synthetic_dataset(
        output_dir=Path(args.output),
        num_samples=args.num_samples,
        charset=cs,
        fonts_dir=args.fonts_dir,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_synthetic.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/ data/dictionaries/ tests/test_synthetic.py
git commit -m "feat: add synthetic data generator"
```

---

### Task 5: Training Engine

**Files:**
- Create: `training/__init__.py`
- Create: `training/train.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `CRNN` from Task 2, `OCRDataset` + `ocr_collate_fn` from Task 3, `Charset` from Task 1, `Settings` from Task 1
- Produces:
  - `Trainer(model: CRNN, charset: Charset, settings: Settings) -> Trainer`
  - `Trainer.train(train_dataset: OCRDataset, val_dataset: OCRDataset | None) -> dict` (training history)
  - Saves best model checkpoint to `checkpoints/best_model.pt`

- [ ] **Step 1: Write failing test for training**

```python
# tests/test_training.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_training.py -v`
Expected: FAIL

- [ ] **Step 3: Implement training/train.py**

```python
# training/__init__.py
# (empty)

# training/train.py
"""Training engine for CRNN OCR model."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from model.crnn import CRNN
from data.dataset import ocr_collate_fn


class Trainer:
    """CRNN model trainer with CTC loss."""

    def __init__(self, charset, settings):
        self.charset = charset
        self.settings = settings
        self.device = torch.device(settings.device)

        # Create model
        self.model = CRNN(
            num_classes=charset.num_classes,
            img_height=settings.img_height,
            img_channels=settings.img_channels,
            rnn_hidden=settings.rnn_hidden_size,
            rnn_layers=settings.rnn_num_layers,
        ).to(self.device)

        # CTC Loss
        self.criterion = nn.CTCLoss(blank=charset.blank_idx, zero_infinity=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=settings.lr_scheduler_patience,
            factor=settings.lr_scheduler_factor,
        )

        # Ensure checkpoint dir exists
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels, label_lengths, _ in dataloader:
            images = images.to(self.device)

            # Forward
            log_probs = self.model(images)  # (T, B, C)
            T = log_probs.shape[0]
            B = log_probs.shape[1]
            input_lengths = torch.full((B,), T, dtype=torch.int32)

            # CTC Loss
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for images, labels, label_lengths, _ in dataloader:
            images = images.to(self.device)
            log_probs = self.model(images)
            T = log_probs.shape[0]
            B = log_probs.shape[1]
            input_lengths = torch.full((B,), T, dtype=torch.int32)
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, train_dataset, val_dataset=None) -> dict:
        """Run full training loop.

        Args:
            train_dataset: Training OCRDataset
            val_dataset: Validation OCRDataset (optional)

        Returns:
            History dict with 'train_loss', 'val_loss' lists
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.settings.batch_size,
            shuffle=True,
            collate_fn=ocr_collate_fn,
            num_workers=self.settings.num_workers,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.settings.batch_size,
                shuffle=False,
                collate_fn=ocr_collate_fn,
                num_workers=self.settings.num_workers,
            )

        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.settings.num_epochs):
            # Train
            train_loss = self._train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)
                monitor_loss = val_loss
            else:
                self.scheduler.step(train_loss)
                monitor_loss = train_loss

            # Print progress
            msg = f"Epoch {epoch + 1}/{self.settings.num_epochs} - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f" - val_loss: {val_loss:.4f}"
            lr = self.optimizer.param_groups[0]["lr"]
            msg += f" - lr: {lr:.6f}"
            print(msg)

            # Save best model
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, best_loss)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.settings.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint."""
        path = self.settings.model_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "charset_size": self.charset.num_classes,
        }, path)
        print(f"Saved checkpoint: {path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_training.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/ tests/test_training.py
git commit -m "feat: implement training engine with CTC loss"
```

---

### Task 6: Benchmark & Metrics

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/metrics.py`
- Create: `benchmark/evaluate.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `CRNN` from Task 2, `OCRDataset` from Task 3, `Charset` from Task 1
- Produces:
  - `compute_cer(prediction: str, ground_truth: str) -> float`
  - `compute_wer(prediction: str, ground_truth: str) -> float`
  - `compute_sequence_accuracy(predictions: list[str], ground_truths: list[str]) -> float`
  - `Evaluator.evaluate(dataset: OCRDataset) -> dict` with CER, WER, seq_acc, per-sample results

- [ ] **Step 1: Write failing test for metrics**

```python
# tests/test_metrics.py
import pytest

def test_cer_identical():
    from benchmark.metrics import compute_cer
    assert compute_cer("hello", "hello") == 0.0

def test_cer_completely_wrong():
    from benchmark.metrics import compute_cer
    cer = compute_cer("abc", "xyz")
    assert cer == 1.0  # 3 substitutions / 3 chars

def test_cer_partial():
    from benchmark.metrics import compute_cer
    cer = compute_cer("helo", "hello")  # 1 insertion needed
    assert 0 < cer < 1

def test_cer_empty_prediction():
    from benchmark.metrics import compute_cer
    cer = compute_cer("", "hello")
    assert cer == 1.0

def test_cer_empty_both():
    from benchmark.metrics import compute_cer
    cer = compute_cer("", "")
    assert cer == 0.0

def test_wer():
    from benchmark.metrics import compute_wer
    assert compute_wer("hello world", "hello world") == 0.0
    assert compute_wer("hello", "world") == 1.0

def test_sequence_accuracy():
    from benchmark.metrics import compute_sequence_accuracy
    preds = ["hello", "world", "foo"]
    gts   = ["hello", "worlD", "foo"]
    acc = compute_sequence_accuracy(preds, gts)
    assert abs(acc - 2/3) < 1e-6

def test_sequence_accuracy_japanese():
    from benchmark.metrics import compute_sequence_accuracy
    preds = ["R03 02月", "5", "B"]
    gts   = ["R03 02月", "5", "B"]
    assert compute_sequence_accuracy(preds, gts) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement benchmark/metrics.py**

```python
# benchmark/__init__.py
# (empty)

# benchmark/metrics.py
"""OCR evaluation metrics: CER, WER, Sequence Accuracy."""
import editdistance


def compute_cer(prediction: str, ground_truth: str) -> float:
    """Compute Character Error Rate.

    CER = edit_distance(pred, gt) / len(gt)
    Returns 0.0 if both empty, 1.0 if gt is empty but pred is not.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(prediction, ground_truth)
    return min(distance / len(ground_truth), 1.0)


def compute_wer(prediction: str, ground_truth: str) -> float:
    """Compute Word Error Rate.

    WER = edit_distance(pred_words, gt_words) / len(gt_words)
    """
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(pred_words, gt_words)
    return min(distance / len(gt_words), 1.0)


def compute_sequence_accuracy(
    predictions: list[str], ground_truths: list[str]
) -> float:
    """Compute exact-match sequence accuracy.

    Returns fraction of predictions that exactly match ground truth.
    """
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)
```

- [ ] **Step 4: Implement benchmark/evaluate.py**

```python
# benchmark/evaluate.py
"""Evaluate OCR model on a dataset and produce benchmark report."""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime

from model.crnn import CRNN
from data.dataset import OCRDataset, ocr_collate_fn
from benchmark.metrics import compute_cer, compute_wer, compute_sequence_accuracy


class Evaluator:
    """Run evaluation on a dataset and compute metrics."""

    def __init__(self, model: CRNN, charset, device: str = "cpu"):
        self.model = model.to(device)
        self.charset = charset
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self, dataset: OCRDataset, batch_size: int = 32
    ) -> dict:
        """Evaluate model on dataset.

        Returns:
            {
                "cer": float,
                "wer": float,
                "sequence_accuracy": float,
                "num_samples": int,
                "per_sample": [{"prediction": str, "ground_truth": str, "cer": float}, ...]
            }
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ocr_collate_fn,
        )

        all_predictions = []
        all_ground_truths = []
        per_sample = []

        for images, labels, label_lengths, texts in loader:
            images = images.to(self.device)
            log_probs = self.model(images)  # (T, B, C)

            # Greedy decode
            _, preds_indices = log_probs.max(2)  # (T, B)
            preds_indices = preds_indices.permute(1, 0)  # (B, T)

            for i, (pred_seq, gt_text) in enumerate(zip(preds_indices, texts)):
                pred_text = self.charset.ctc_decode(pred_seq.tolist())
                all_predictions.append(pred_text)
                all_ground_truths.append(gt_text)

                sample_cer = compute_cer(pred_text, gt_text)
                per_sample.append({
                    "prediction": pred_text,
                    "ground_truth": gt_text,
                    "cer": sample_cer,
                })

        avg_cer = sum(compute_cer(p, g) for p, g in zip(all_predictions, all_ground_truths)) / max(len(all_predictions), 1)
        avg_wer = sum(compute_wer(p, g) for p, g in zip(all_predictions, all_ground_truths)) / max(len(all_predictions), 1)
        seq_acc = compute_sequence_accuracy(all_predictions, all_ground_truths)

        return {
            "cer": avg_cer,
            "wer": avg_wer,
            "sequence_accuracy": seq_acc,
            "num_samples": len(all_predictions),
            "per_sample": per_sample,
            "timestamp": datetime.now().isoformat(),
        }

    def save_report(self, results: dict, output_path: Path):
        """Save evaluation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Report saved: {output_path}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add benchmark/ tests/test_metrics.py
git commit -m "feat: add benchmark metrics and evaluator"
```

---

### Task 7: Inference Module

**Files:**
- Create: `inference/__init__.py`
- Create: `inference/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `CRNN` from Task 2, `Charset` from Task 1, `Settings` from Task 1
- Produces:
  - `OCRPredictor(checkpoint_path: Path, device: str) -> OCRPredictor`
  - `OCRPredictor.predict(image: PIL.Image | Path) -> dict` with `{"text": str, "confidence": float}`
  - `OCRPredictor.predict_batch(images: list) -> list[dict]`

- [ ] **Step 1: Write failing test for predictor**

```python
# tests/test_predict.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_predict.py -v`
Expected: FAIL

- [ ] **Step 3: Implement inference/predict.py**

```python
# inference/__init__.py
# (empty)

# inference/predict.py
"""OCR inference: predict text from images."""
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from model.crnn import CRNN
from config.charset import Charset


class OCRPredictor:
    """Load a trained CRNN model and predict text from images."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        img_height: int = 32,
        img_max_width: int = 256,
    ):
        self.device = torch.device(device)
        self.img_height = img_height
        self.img_max_width = img_max_width
        self.charset = Charset()

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        num_classes = checkpoint.get("charset_size", self.charset.num_classes)
        self.model = CRNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image) -> torch.Tensor:
        """Preprocess a single image for the model.

        Args:
            image: PIL.Image or file path

        Returns:
            Tensor of shape (1, 1, H, W)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("L")

        # Resize to target height, maintain aspect ratio
        w, h = image.size
        new_w = max(1, int(w * self.img_height / h))
        new_w = min(new_w, self.img_max_width)
        image = image.resize((new_w, self.img_height), Image.BILINEAR)

        tensor = TF.to_tensor(image)  # (1, H, W)
        return tensor.unsqueeze(0)  # (1, 1, H, W)

    @torch.no_grad()
    def predict(self, image) -> dict:
        """Predict text from a single image.

        Args:
            image: PIL.Image or file path

        Returns:
            {"text": str, "confidence": float}
        """
        input_tensor = self._preprocess(image).to(self.device)
        log_probs = self.model(input_tensor)  # (T, 1, C)

        # Greedy decode with confidence
        probs = torch.exp(log_probs.squeeze(1))  # (T, C)
        max_probs, pred_indices = probs.max(dim=1)  # (T,)

        # CTC decode
        pred_text = self.charset.ctc_decode(pred_indices.tolist())

        # Confidence: average probability of non-blank predictions
        non_blank_mask = pred_indices != self.charset.blank_idx
        if non_blank_mask.any():
            confidence = max_probs[non_blank_mask].mean().item()
        else:
            confidence = 0.0

        return {
            "text": pred_text,
            "confidence": round(confidence, 4),
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list[dict]:
        """Predict text from multiple images.

        Args:
            images: List of PIL.Image or file paths

        Returns:
            List of {"text": str, "confidence": float}
        """
        # Process each image individually (variable width)
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_predict.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add inference/ tests/test_predict.py
git commit -m "feat: add inference predictor module"
```

---

### Task 8: Main CLI & End-to-End Pipeline

**Files:**
- Create: `main.py`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces:
  - CLI with commands: `generate`, `train`, `evaluate`, `predict`
  - End-to-end workflow: generate data → train → evaluate → predict

- [ ] **Step 1: Write failing e2e test**

```python
# tests/test_e2e.py
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
```

- [ ] **Step 2: Implement main.py CLI**

```python
# main.py
"""LamSonOCR - Main CLI entry point."""
import argparse
from pathlib import Path


def cmd_generate(args):
    """Generate synthetic training data."""
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset

    cs = Charset()
    output_dir = Path(args.output)
    generate_synthetic_dataset(
        output_dir=output_dir,
        num_samples=args.num_samples,
        charset=cs,
        fonts_dir=args.fonts_dir,
    )
    print(f"✅ Generated {args.num_samples} samples in {output_dir}")


def cmd_train(args):
    """Train the CRNN model."""
    from config.charset import Charset
    from config.settings import Settings
    from data.dataset import OCRDataset
    from data.augmentation import OCRAugmentation
    from training.train import Trainer

    cs = Charset()
    settings = Settings(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    if args.checkpoint_dir:
        settings.model_dir = Path(args.checkpoint_dir)

    train_ds = OCRDataset(
        root_dir=Path(args.train_data),
        charset=cs,
        transform=OCRAugmentation() if not args.no_augment else None,
    )

    val_ds = None
    if args.val_data:
        val_ds = OCRDataset(root_dir=Path(args.val_data), charset=cs)

    print(f"📊 Training samples: {len(train_ds)}")
    if val_ds:
        print(f"📊 Validation samples: {len(val_ds)}")
    print(f"🏋️ Device: {settings.device}")
    print(f"🔤 Character classes: {cs.num_classes}")

    trainer = Trainer(charset=cs, settings=settings)
    history = trainer.train(train_ds, val_ds)

    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    if history.get('val_loss'):
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")


def cmd_evaluate(args):
    """Evaluate model on a dataset."""
    import torch
    from config.charset import Charset
    from data.dataset import OCRDataset
    from model.crnn import CRNN
    from benchmark.evaluate import Evaluator

    cs = Charset()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    model = CRNN(num_classes=checkpoint.get("charset_size", cs.num_classes))
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = OCRDataset(root_dir=Path(args.data), charset=cs)

    evaluator = Evaluator(model=model, charset=cs, device=args.device)
    results = evaluator.evaluate(dataset)

    print(f"\n📋 Benchmark Results")
    print(f"   Samples: {results['num_samples']}")
    print(f"   CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    print(f"   WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"   Seq Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_accuracy']*100:.2f}%)")

    if args.report:
        evaluator.save_report(results, Path(args.report))


def cmd_predict(args):
    """Predict text from images."""
    from inference.predict import OCRPredictor
    from pathlib import Path

    predictor = OCRPredictor(
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
    )

    images = args.images
    for img_path in images:
        result = predictor.predict(img_path)
        print(f"📝 {img_path}: '{result['text']}' (conf: {result['confidence']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="LamSonOCR - CRNN-based text line OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate
    gen = subparsers.add_parser("generate", help="Generate synthetic training data")
    gen.add_argument("--output", type=str, default="data/train", help="Output directory")
    gen.add_argument("--num-samples", type=int, default=50000, help="Number of samples")
    gen.add_argument("--fonts-dir", type=str, default=None, help="Fonts directory")

    # Train
    trn = subparsers.add_parser("train", help="Train the model")
    trn.add_argument("--train-data", type=str, required=True, help="Training data dir")
    trn.add_argument("--val-data", type=str, default=None, help="Validation data dir")
    trn.add_argument("--batch-size", type=int, default=64)
    trn.add_argument("--epochs", type=int, default=50)
    trn.add_argument("--lr", type=float, default=0.001)
    trn.add_argument("--checkpoint-dir", type=str, default=None)
    trn.add_argument("--no-augment", action="store_true")

    # Evaluate
    evl = subparsers.add_parser("evaluate", help="Evaluate model")
    evl.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    evl.add_argument("--data", type=str, required=True, help="Test data dir")
    evl.add_argument("--device", type=str, default="cpu")
    evl.add_argument("--report", type=str, default=None, help="Save report to file")

    # Predict
    prd = subparsers.add_parser("predict", help="Predict text from images")
    prd.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    prd.add_argument("--device", type=str, default="cpu")
    prd.add_argument("images", nargs="+", help="Image files to predict")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run e2e test**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/test_e2e.py -v`
Expected: PASS

- [ ] **Step 4: Run all tests**

Run: `cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr && venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_e2e.py
git commit -m "feat: add CLI and end-to-end pipeline"
```

---

### Task 9: Generate Data & Run First Training

**Files:**
- No new files (uses existing pipeline)

**Interfaces:**
- Consumes: CLI from Task 8

- [ ] **Step 1: Generate training data**

```bash
cd /Volumes/SpaceX/WorkSpace/python/LamSonOcr
venv/bin/python main.py generate --output data/train --num-samples 10000
venv/bin/python main.py generate --output data/val --num-samples 2000
venv/bin/python main.py generate --output data/test --num-samples 1000
```

- [ ] **Step 2: Run training**

```bash
venv/bin/python main.py train \
  --train-data data/train \
  --val-data data/val \
  --batch-size 32 \
  --epochs 20 \
  --lr 0.001
```

- [ ] **Step 3: Run benchmark**

```bash
venv/bin/python main.py evaluate \
  --checkpoint checkpoints/best_model.pt \
  --data data/test \
  --report reports/benchmark_v1.json
```

- [ ] **Step 4: Test prediction**

```bash
# Pick a few test images
venv/bin/python main.py predict \
  --checkpoint checkpoints/best_model.pt \
  data/test/images/000001.png \
  data/test/images/000002.png \
  data/test/images/000003.png
```

- [ ] **Step 5: Commit results**

```bash
git add reports/
git commit -m "benchmark: first training run results"
```
