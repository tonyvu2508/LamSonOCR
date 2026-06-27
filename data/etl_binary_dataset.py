"""Direct binary ETL dataset reader using memory mapping (mmap)."""
import os
import mmap
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

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

def decode_char(char_code_bytes) -> str:
    """Decode JIS character code to string."""
    try:
        if len(char_code_bytes) == 2:
            # 2-byte JIS X 0208
            jis_bytes = b'\x1b\x24\x42' + char_code_bytes + b'\x1b\x28\x42'
            return jis_bytes.decode('iso2022_jp')
    except Exception:
        pass

    # For 1 byte
    code = char_code_bytes[0] if len(char_code_bytes) > 0 else 0
    if code in JIS_TO_HIRAGANA:
        return JIS_TO_HIRAGANA[code]
        
    if 32 <= code <= 126:
        return chr(code)
        
    return None

class ETLBinaryDataset(Dataset):
    def __init__(self, etl_dir: str, charset, target_height: int = 32, transform=None):
        """
        Args:
            etl_dir: Directory containing raw ETL .dat files.
            charset: Charset instance for text encoding.
            target_height: Output height of the image (default 32 for CRNN).
            transform: Optional data augmentation to apply to the images.
        """
        super().__init__()
        self.etl_dir = Path(etl_dir)
        self.charset = charset
        self.target_height = target_height
        self.transform = transform
        
        # We store index of items: (file_index, byte_offset, mode, char_label)
        self.index_table = []
        
        # We hold mmaps inside a dictionary (lazy loaded per worker process)
        self.files = []
        self._mmaps = {} 
        self._fhandles = {}
        
        self._build_index()

    def _build_index(self):
        print(f"Scanning binary ETL files in {self.etl_dir}...")
        
        if not self.etl_dir.exists():
            print(f"Warning: Directory {self.etl_dir} does not exist.")
            return

        etl_files = sorted(list(self.etl_dir.rglob("*")))
        for f in etl_files:
            name_upper = f.name.upper()
            if not f.is_file() or name_upper.endswith(".ZIP") or "INFO" in name_upper or name_upper.endswith(".TXT") or name_upper.endswith(".JSON"):
                continue
                
            file_size = os.path.getsize(f)
            if file_size == 0:
                continue

            # Auto-detect record type based on dataset family name in path
            path_upper = f.path.upper() if hasattr(f, 'path') else str(f).upper()
            
            if "ETL1" in path_upper or "ETL6" in path_upper or "ETL7" in path_upper:
                record_size = 2052
                mode = "M-Type"
            elif "ETL3" in path_upper or "ETL4" in path_upper or "ETL5" in path_upper:
                record_size = 2952
                mode = "C-Type"
            elif "ETL8" in path_upper or "ETL9" in path_upper:
                # B-Type is either 512 or 576 bytes
                if file_size % 512 == 0:
                    record_size = 512
                elif file_size % 576 == 0:
                    record_size = 576
                else:
                    record_size = 576 # fallback
                mode = "B-Type"
            else:
                # Fallback to modulo if no keyword matches
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
                    continue
            
            file_idx = len(self.files)
            self.files.append(f)
            
            # Read first pass to extract labels (takes only a few seconds)
            num_records = file_size // record_size
            with open(f, "rb") as bf:
                for i in range(num_records):
                    chunk = bf.read(record_size)
                    
                    if mode == "M-Type":
                        char_bytes = chunk[2:4]
                    elif mode == "C-Type":
                        char_bytes = bytes([chunk[9]])
                    else: # B-Type
                        char_bytes = chunk[2:4]
                        
                    char = decode_char(char_bytes)
                    if char:
                        self.index_table.append((file_idx, i * record_size, mode, record_size, char))

        print(f"Index built: {len(self.index_table)} records found.")

    def __len__(self):
        return len(self.index_table)
        
    def _get_mmap(self, file_idx):
        if file_idx not in self._mmaps:
            fpath = self.files[file_idx]
            fh = open(fpath, "rb")
            self._fhandles[file_idx] = fh
            # 0 length means map the whole file
            self._mmaps[file_idx] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[file_idx]

    def _close_mmaps(self):
        for m in self._mmaps.values():
            m.close()
        for fh in self._fhandles.values():
            fh.close()
        self._mmaps.clear()
        self._fhandles.clear()
        
    def __del__(self):
        self._close_mmaps()

    def __getitem__(self, idx):
        file_idx, byte_offset, mode, record_size, label = self.index_table[idx]
        mm = self._get_mmap(file_idx)
        
        # Read exact slice from memory map
        record = mm[byte_offset : byte_offset + record_size]
        
        # Safety check: if record is truncated at EOF, pad it with zeros
        if len(record) < record_size:
            record = record.ljust(record_size, b'\x00')
            
        if mode == "M-Type":
            # 2016 bytes starting at offset 38. 63x64 pixels. 4-bit per pixel
            img_bytes = record[38:38+2016]
            # Convert 4-bit to 8-bit
            # Each byte has 2 pixels. High nibble, low nibble.
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            p1 = (arr >> 4) & 0x0F
            p2 = arr & 0x0F
            # Interleave p1 and p2
            pixels = np.empty((arr.size * 2,), dtype=np.uint8)
            pixels[0::2] = p1
            pixels[1::2] = p2
            img_array = 255 - (pixels * 17) # Invert
            img_array = img_array.reshape(63, 64)
            
        elif mode == "C-Type":
            # 2736 bytes starting at offset 216. 76x72 pixels. 4-bit per pixel
            img_bytes = record[216:216+2736]
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            p1 = (arr >> 4) & 0x0F
            p2 = arr & 0x0F
            pixels = np.empty((arr.size * 2,), dtype=np.uint8)
            pixels[0::2] = p1
            pixels[1::2] = p2
            img_array = 255 - (pixels * 17)
            img_array = img_array.reshape(76, 72)
            
        else: # B-Type
            # 504 bytes starting at offset 8. 63x64 pixels. 1-bit per pixel
            img_bytes = record[8:8+504]
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            # Unpack bits: np.unpackbits is super fast
            pixels = np.unpackbits(arr) * 255
            img_array = pixels.reshape(63, 64)
            # Polarity check
            if img_array.mean() < 127:
                img_array = 255 - img_array

        # Convert to PIL for resize
        img = Image.fromarray(img_array)
        w, h = img.size
        new_w = max(1, int(w * self.target_height / h))
        img = img.resize((new_w, self.target_height), Image.BILINEAR)
        
        if self.transform:
            img = self.transform(img)
            
        # Convert to tensor [1, H, W], normalized to [-1, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dim
        img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1]
        
        encoded_label = self.charset.encode(label)
        
        return img_tensor, encoded_label
