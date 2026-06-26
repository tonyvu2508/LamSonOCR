# LamSonOCR — CRNN-based Japanese/English OCR

LamSonOCR là một giải pháp OCR nhận dạng dòng chữ chuyên dụng (Text Line Recognition) được thiết kế đặc biệt để nhận diện chữ viết tay và chữ in tiếng Nhật (Hiragana, Katakana, Kanji), tiếng Anh và chữ số trên các tài liệu như phiếu đấu giá xe (auction sheets).

Hệ thống được tối ưu hóa chạy mượt mà trên MacOS (bao gồm các chip Apple Silicon M1/M2/M3 bằng MPS backend).

---

## 🏗️ Kiến trúc Model

Mô hình hoạt động theo kiến trúc **CRNN + CTC Loss**:
1. **CNN (VGG/ResNet-like)**: Trích xuất các đặc trưng không gian từ ảnh dòng chữ đầu vào (kích thước cố định $32 \times W$).
2. **Map-to-Sequence**: Biến đổi tensor đặc trưng không gian thành chuỗi vector thời gian.
3. **BiLSTM (2 lớp)**: Học các đặc trưng chuỗi ngữ cảnh hai chiều.
4. **CTC Loss (Connectionist Temporal Classification)**: Cho phép huấn luyện mô hình mà không cần gán nhãn căn chỉnh (alignment) cụ thể cho từng ký tự.

---

## 🚀 Hướng dẫn Cài đặt

1. **Khởi tạo môi trường ảo (venv):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Cài đặt các gói thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   pip install bitstring
   ```

---

## 📊 Chuẩn bị Dữ liệu Huấn luyện

Dự án hỗ trợ 2 nguồn dữ liệu chính để huấn luyện: **Dữ liệu giả lập (Synthetic)** và **Dữ liệu chữ viết tay thực tế (ETL Dataset)**.

### Cách 1: Tạo dữ liệu giả lập (Synthetic Data)
Sử dụng script sinh chữ tiếng Nhật tự động kết hợp với các fonts hệ thống để sinh dữ liệu ngẫu nhiên (chữ số, biển số xe, ngày tháng, nội dung hỗn hợp):

```bash
# Sinh 50,000 mẫu cho tập huấn luyện
python main.py generate --output data/train --num-samples 50000

# Sinh 5,000 mẫu cho tập kiểm thử/đánh giá
python main.py generate --output data/val --num-samples 5000
```

### Cách 2: Trích xuất dữ liệu chữ viết tay ETL4 (Hiragana)
Đặt file nhị phân của bộ dữ liệu ETL vào thư mục dự án (ví dụ `ETL/ETL4/ETL4C`), sau đó chạy script giải nén:

```bash
python scripts/prepare_etl.py --input ETL/ETL4/ETL4C --output data/etl_train
```

**Các tham số:**
*   `--input`: Đường dẫn tới file cơ sở dữ liệu nhị phân ETL (mặc định: `ETL/ETL4/ETL4C`).
*   `--output`: Thư mục đầu ra để lưu trữ ảnh PNG đã giải nén và file `labels.csv` (mặc định: `data/etl_train`).


---

## 🏋️ Huấn luyện Model (Training)

Chạy lệnh train tích hợp sẵn để tự động phát hiện GPU/MPS và tiến hành tối ưu hóa:

```bash
# Huấn luyện trên tập dữ liệu giả lập
python main.py train --train-data data/train --val-data data/val --epochs 50 --batch-size 64

# Huấn luyện trên tập dữ liệu ETL thực tế
python main.py train --train-data data/etl_train --epochs 50 --batch-size 64
```

> [!NOTE]
> Mô hình tự động lưu checkpoint tối ưu nhất tại `checkpoints/best_model.pt` trong suốt quá trình train.

---

## 📋 Đánh giá & Dự đoán (Evaluation & Inference)

### Đánh giá mô hình trên tập kiểm thử (CER / WER)
Hệ thống sẽ tính toán Character Error Rate (CER), Word Error Rate (WER) và Sequence Accuracy:
```bash
python main.py evaluate --checkpoint checkpoints/best_model.pt --data data/val
```

### Nhận diện ký tự trên ảnh thực tế (Inference)
Dự đoán văn bản trong một hoặc nhiều ảnh dòng chữ:
```bash
python main.py predict --checkpoint checkpoints/best_model.pt path/to/image1.png path/to/image2.png
```

---

## 🧪 Chạy Kiểm thử hệ thống (Tests)
Toàn bộ mã nguồn đi kèm bộ unit/integration test bao phủ đầy đủ các module:
```bash
pytest tests/ -v
```

