# LamSonOCR - Design Specification

## Problem Statement

Hiện tại, hệ thống OCR dùng GLM-OCR (VLM 0.9B) cho auction sheet xe Nhật đang gặp lỗi nghiêm trọng:
- **初度登録年月** (ngày đăng ký): Nhiều giá trị sai (`R54`, `130 8`, `Rb 1`, `286`) thay vì format chuẩn `H/R + năm + tháng`
- **定員** (số chỗ): Trả về `人` thay vì số
- **ドア** (cửa): Trả về `←`, `f`, `定局`, `markdown code blocks`

Cần một model OCR chuyên biệt, nhẹ, train từ đầu, chạy được trên Mac M1.

## Goals

1. Train model CRNN+CTC text line recognition từ đầu
2. Hỗ trợ tiếng Nhật (Hiragana, Katakana, Kanji) + tiếng Anh + số
3. Benchmark framework đo CER, WER, Sequence Accuracy
4. Chạy trên Mac M1 (MPS backend)
5. Tạo synthetic dataset cho training

## Non-Goals

- Text detection (tìm vùng text) — assume text đã được crop
- Full page OCR
- Real-time video OCR
- Export ONNX (có thể thêm sau)

## Architecture

### CRNN Model

```
Input Image (H=32, W=variable)
    │
    ▼
┌──────────────┐
│  CNN Backbone │  ResNet-like (nhẹ, ~2M params)
│  - Conv layers│  Output: (batch, channels, 1, W')
│  - BatchNorm  │
│  - ReLU       │
│  - MaxPool    │
└──────┬───────┘
       │ Reshape to (batch, W', channels)
       ▼
┌──────────────┐
│  BiLSTM      │  2 layers, hidden=256
│  (Sequence)  │  Output: (batch, W', num_classes)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  FC Layer    │  Linear(hidden*2, num_classes)
│  + LogSoftmax│
└──────┬───────┘
       │
       ▼
  CTC Decoding → Text Output
```

### Data Pipeline

1. **Synthetic Data Generator**: TextRecognitionDataGenerator (TRDG)
   - Font: Nhiều Japanese fonts (Gothic, Mincho, handwriting styles)
   - Content: Từ điển domain-specific (số xe, ngày tháng, mã, tên)
   - Augmentation: blur, noise, rotation nhẹ, perspective transform
   
2. **Public Datasets**:
   - ETL Character Database (Hiragana, Katakana, Kanji)
   - KMNIST (Kuzushiji-MNIST)
   - Kaggle Japanese Handwriting datasets

3. **Dataset Format**:
   ```
   data/
   ├── train/
   │   ├── images/
   │   │   ├── 00001.png
   │   │   └── ...
   │   └── labels.csv  # image_path, text
   ├── val/
   │   └── ...
   └── test/
       └── ...
   ```

### Character Set

```python
CHARSET = {
    'digits': '0123456789',
    'english_upper': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'english_lower': 'abcdefghijklmnopqrstuvwxyz',
    'hiragana': 'あいうえお...ん',  # 46 basic + variants
    'katakana': 'アイウエオ...ン',  # 46 basic + variants
    'kanji_common': '年月日人車...', # ~1000 common kanji
    'symbols': '/-()[]・。、',
    'blank': '<BLANK>',  # CTC blank token (index 0)
}
```

### Benchmark Suite

Metrics đo:
- **CER** (Character Error Rate): Edit distance / total chars
- **WER** (Word Error Rate): Cho text dài
- **Sequence Accuracy**: Đúng hoàn toàn cả chuỗi
- **Per-field Accuracy**: Đo riêng cho từng loại field (出品番号, 初度登録年月, etc.)
- **Confidence Calibration**: So sánh predicted confidence vs actual accuracy

### Project Structure

```
LamSonOcr/
├── config/
│   ├── charset.py          # Character set definitions
│   ├── training_config.py  # Hyperparameters
│   └── paths.py            # Data/model paths
├── data/
│   ├── generate_synthetic.py  # Synthetic data generator
│   ├── dataset.py            # PyTorch Dataset class
│   ├── augmentation.py       # Data augmentation transforms
│   └── utils.py              # Label encoding/decoding
├── model/
│   ├── crnn.py              # CRNN model architecture
│   ├── backbone.py          # CNN backbone options
│   └── decoder.py           # CTC decoder (greedy + beam search)
├── training/
│   ├── train.py             # Training loop
│   ├── loss.py              # CTC Loss wrapper
│   └── lr_scheduler.py      # Learning rate scheduler
├── benchmark/
│   ├── metrics.py           # CER, WER, Seq Accuracy
│   ├── evaluate.py          # Evaluation runner
│   └── report.py            # Generate benchmark reports
├── inference/
│   ├── predict.py           # Single image prediction
│   └── batch_predict.py     # Batch prediction
├── tests/
│   ├── test_charset.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_training.py
├── scripts/
│   ├── download_datasets.py  # Download public datasets
│   └── prepare_data.py       # Prepare/convert datasets
├── requirements.txt
└── README.md
```

## Training Plan

1. **Phase 1 — Synthetic Only**: Train trên 50K synthetic images
2. **Phase 2 — Add Public Data**: Thêm ETL/KMNIST character data
3. **Phase 3 — Domain Fine-tune**: Fine-tune trên auction sheet crops (khi có labeled data)

## Hardware Requirements

- Mac M1 (MPS backend) hoặc CPU
- RAM: 8GB minimum
- Storage: ~5GB cho data + models

## Success Criteria

- CER < 5% trên test set
- Sequence Accuracy > 90% cho text ngắn (1-5 chars)
- Training time < 4 hours trên M1
- Inference < 50ms per image
