---
name: lamson-ocr-ops
description: Use when starting a new session or running operations (testing, ssh, git, training) on the LamSonOCR project.
---

# LamSonOCR Operations Runbook

## Overview
This skill provides the standard operational commands, environment setups, and workflows for developing and training the LamSonOCR system.

## Quick Reference Commands

### 1. Giả lập / Kiểm thử (Testing)
Để chạy toàn bộ suite test bằng PyTest:
```bash
venv/bin/python -m pytest tests/ -v
```

### 2. Thiết lập & Huấn luyện (Setup & Training)
Để bắt đầu quá trình huấn luyện trực tiếp không qua giải nén (Zero-Extraction) với GPU RTX 3090 (đọc từ file nhị phân trong thư mục `ETL` hoặc `ETL_parts`):

Để việc huấn luyện không bị ngắt quãng khi mất kết nối mạng, hãy sử dụng **tmux** để chạy ngầm trên máy chủ:

*   **Tạo một session tmux mới tên là `ocr`**:
    ```bash
    tmux new -s ocr
    ```
*   **Chạy lệnh huấn luyện bên trong tmux**:
    ```bash
    ./setup_and_train.sh --train --batch-size 1024 --epochs 200
    ```
*   **Thoát tạm thời ra ngoài (Detach session)**:
    Ấn tổ hợp phím `Ctrl + B` rồi ấn phím `D`.
*   **Quay lại session đang chạy (Attach session)**:
    ```bash
    tmux attach -t ocr
    ```
*   **Liệt kê các session đang có**:
    ```bash
    tmux ls
    ```

*Note*: Nếu chạy lần đầu, hãy chắc chắn đã `git pull` bản mới nhất và đã kích hoạt môi trường ảo.

### 3. Git Operations
Khi kết thúc phiên làm việc hoặc cập nhật bản vá lỗi:
```bash
# Đồng bộ code mới
git pull

# Commit & Push thay đổi
git add <các_file_thay_đổi>
git commit -m "Mô tả thay đổi rõ ràng"
git push
```

### 4. Kết nối từ xa (Remote SSH)
Các câu lệnh SSH thường dùng để kết nối đến máy chủ GPU (RunPod hoặc server nội bộ):
*   **SSH tới server GPU 1**:
    ```bash
    ssh root@194.26.196.156 -p 10268 -i ~/.ssh/id_ed25519
    ```
*   **SSH tới RunPod**:
    ```bash
    ssh 1f5kuad6m8zprk-6441169a@ssh.runpod.io -i ~/.ssh/id_ed25519
    ```

---

## Kiến trúc nạp dữ liệu Direct ETL Binary Loading
*   Dữ liệu ETL không còn được trích xuất thành ảnh PNG riêng lẻ. Thay vào đó, tệp nhị phân gốc được ánh xạ qua bộ nhớ ảo sử dụng **Memory Mapping (`mmap`)** thông qua lớp `ETLBinaryDataset`.
*   **Bộ mã hóa tự động loại bỏ rác**: Tệp nhãn không hợp lệ hoặc các tệp metadata (`INFO`, `.txt`, `.zip`) tự động được lọc khỏi danh sách huấn luyện để ngăn lỗi `NaN` trong hàm CTC Loss.
*   **Chuẩn hóa dữ liệu**: Mọi ảnh thô từ ETL đều được đưa về dải `[0, 1]` đồng bộ với tập sinh tự động (Synthetic Dataset) và được đệm khít về độ rộng lớn nhất trong batch.

## Khắc phục lỗi thường gặp
*   **Lỗi sụp đổ `torch.compile`**: Đã khắc phục bằng cách chuyển đổi lớp `nn.AdaptiveAvgPool2d((1, None))` thành lớp `MeanHeight` tự định nghĩa (tính `mean(dim=2)`). Không thay đổi thuật toán của mô hình.
*   **Lỗi `ValueError: cannot reshape...`**: Do nhận diện nhầm định dạng record giữa các nhóm ETL. Cơ chế phân loại mới dựa trên từ khóa tên tệp (`ETL1`, `ETL9`,...) đã giải quyết triệt để lỗi này.
