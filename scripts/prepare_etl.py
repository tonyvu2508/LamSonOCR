"""ETL Dataset Extractor.

Extracts handwritten characters from binary C-type ETL3 and ETL4 files,
resizes them to 32px height, and creates a dataset ready for training.
"""
import os
import csv
import bitstring
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Mapping from JIS X 0201 Katakana character codes to Hiragana (for ETL4)
JIS_TO_HIRAGANA = {
    0xB1: "あ", 0xB2: "い", 0xB3: "う", 0xB4: "え", 0xB5: "お",
    0xB6: "か", 0xB7: "き", 0xB8: "く", 0xB9: "け", 0xBA: "こ",
    0xBB: "さ", 0xBC: "し", 0xBD: "す", 0xBE: "せ", 0xBF: "そ",
    0xC0: "た", 0xC1: "ち", 0xC2: "つ", 0xC3: "て", 0xC4: "と",
    0xC5: "な", 0xC6: "に", 0xC7: "ぬ", 0xC8: "ね", 0xC9: "の",
    0xCA: "は", 0xCB: "ひ", 0xCC: "ふ", 0xCD: "へ", 0xCE: "ほ",
    0xCF: "ま", 0xD0: "み", 0xD1: "む", 0xD2: "め", 0xD3: "も",
    0xD4: "や", 0xD5: "ゆ", 0xD6: "よ",
    0xD7: "ら", 0xD8: "り", 0xD9: "る", 0xDA: "れ", 0xDB: "ろ",
    0xDC: "わ", 0xDD: "を", 0xDE: "ん",
    0xA7: "ぁ", 0xA8: "ぃ", 0xA9: "ぅ", 0xAA: "ぇ", 0xAB: "ぉ",
    0xAF: "っ", 0xAC: "ゃ", 0xAD: "ゅ", 0xAE: "ょ",
}

def decode_char(char_code) -> str:
    """Decode JIS character code to string."""
    if isinstance(char_code, bytes):
        try:
            # 2-byte JIS X 0208 (Kanji/Hiragana)
            jis_bytes = b'\x1b\x24\x42' + char_code + b'\x1b\x28\x42'
            return jis_bytes.decode('iso2022_jp')
        except Exception:
            return None

    # 1. Check if it's in our Katakana -> Hiragana map (ETL4)
    if char_code in JIS_TO_HIRAGANA:
        return JIS_TO_HIRAGANA[char_code]
        
    # 2. Check if it's standard ASCII (ETL3 Alphanumeric)
    if 32 <= char_code <= 126:
        return chr(char_code)
        
    return None

def extract_etl(file_path: Path, output_dir: Path):
    """Extract all records from ETL binary file (supports M-Type, C-Type, and B-Type)."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    file_size = os.path.getsize(file_path)
    
    # Auto-detect record type
    if file_size % 2052 == 0:
        record_size = 2052
        mode = "M-Type"
    elif file_size % 2952 == 0:
        record_size = 2952
        mode = "C-Type"
    elif file_size % 512 == 0:
        record_size = 512
        mode = "B-Type"
    elif file_size % 576 == 0:
        record_size = 576
        mode = "B-Type"
    else:
        raise ValueError(f"Unknown record format for file size {file_size} of {file_path.name}")
        
    num_records = file_size // record_size
    
    print(f"Extracting {num_records} samples from {file_path.name} (Format: {mode})...")
    
    samples = []
    
    with open(file_path, "rb") as f:
        for i in tqdm(range(num_records)):
            _bytes = f.read(record_size)
            if len(_bytes) < record_size:
                break
                
            # Parse fields depending on record type
            if mode == "M-Type":
                # M-Type (ETL1, ETL6, ETL7)
                # Parse with bitstring equivalent
                f_bit = bitstring.ConstBitStream(bytes=_bytes)
                r = f_bit.readlist('uint:16,bytes:2,uint:16,hex:8,hex:8,4*uint:8,uint:32,4*uint:16,4*uint:8,pad:32,bytes:2016,pad:32')
                char_code = r[1][0]  # First byte of the 2-byte char code
                img_bytes = r[18]
                height, width = 63, 64
                is_binary = False
            elif mode == "C-Type":
                # C-Type (ETL3, ETL4, ETL5)
                f_bit = bitstring.ConstBitStream(bytes=_bytes)
                r = f_bit.readlist('2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,15*uint:36,pad:1008,bytes:2736')
                char_code = r[2]
                img_bytes = r[23]
                height, width = 76, 72
                is_binary = False
            else:
                # B-Type (ETL8B, ETL9B)
                # Unpack using struct
                if record_size == 512:
                    r = struct.unpack(">H2s4s504s", _bytes)
                else:
                    r = struct.unpack(">H2s4s504s64x", _bytes)
                char_code = r[1]  # 2-byte character code bytes
                img_bytes = r[3]
                height, width = 63, 64
                is_binary = True
            
            char = decode_char(char_code)
            if not char:
                continue
                
            if is_binary:
                # Unpack 1-bit packed pixels (B-Type)
                pixels = []
                for b in img_bytes:
                    for bit in range(7, -1, -1):
                        p = (b >> bit) & 1
                        pixels.append(p)
                img_array = np.array(pixels, dtype=np.uint8) * 255
                img_array = img_array.reshape(height, width)
                # Auto polarity detection: ensure background is white (255)
                if img_array.mean() < 127:
                    img_array = 255 - img_array
            else:
                # Decode 4-bit gray levels (0-15) to 8-bit (0-255), and invert (255 - val)
                pixels = []
                for b in img_bytes:
                    p1 = (b >> 4) & 0x0F
                    p2 = b & 0x0F
                    pixels.extend([p1, p2])
                    
                img_array = 255 - (np.array(pixels, dtype=np.uint8) * 17)
                img_array = img_array.reshape(height, width)
            
            # Convert to PIL and resize to 32px height to match our CRNN input
            img = Image.fromarray(img_array)
            w, h = img.size
            new_w = max(1, int(w * 32 / h))
            img = img.resize((new_w, 32), Image.BILINEAR)
            
            # Unique image name based on file name and index
            img_filename = f"{file_path.stem}_{i:06d}.png"
            img_path = img_dir / img_filename
            img.save(img_path)
            
            samples.append({
                "image": f"data/etl_train/images/{img_filename}",
                "text": char
            })
        
    # Append or write metadata labels CSV
    labels_csv = output_dir / "labels.csv"
    file_exists = labels_csv.exists()
    
    # Read existing samples if file exists to append
    existing_samples = []
    if file_exists:
        try:
            with open(labels_csv, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_samples = list(reader)
        except Exception:
            pass
            
    # Combine existing and new samples, avoiding duplicate image paths
    combined_samples = {s["image"]: s for s in existing_samples}
    for s in samples:
        combined_samples[s["image"]] = s
        
    with open(labels_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(combined_samples.values())
        
    print(f"✅ Successfully extracted {len(samples)} samples. Total in labels.csv: {len(combined_samples)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ETL3/ETL4 dataset")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/etl_train")
    args = parser.parse_args()
    
    extract_etl(Path(args.input), Path(args.output))
