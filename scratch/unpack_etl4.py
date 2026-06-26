import os
import struct
import numpy as np
from PIL import Image

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"
output_dir = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/scratch/etl4_extracted"
os.makedirs(output_dir, exist_ok=True)

with open(filepath, "rb") as f:
    record = f.read(2952)
    
    # Header is 216 bytes, image is 2736 bytes (72 x 76 pixels, 4-bit/pixel)
    header = record[:216]
    img_bytes = record[216:]
    
    # Let's inspect the header
    # Shift-JIS or JIS code is usually at offset 8 (2 bytes) or similar
    # Let's unpack the first few fields
    # Format of C-type header (216 bytes):
    # Serial sheet: 2 bytes
    # Serial number: 2 bytes
    # JIS code: 2 bytes (or similar)
    # Let's see:
    unpacked_header = struct.unpack(">H H H H H H H H H H", header[:20])
    print("Header shorts:", unpacked_header)
    
    # In ETL4, the character code (JIS) is at bytes 8-9 (index 4 in shorts)
    # Let's verify
    jis_code = unpacked_header[4]
    print(f"JIS Code (hex): {hex(jis_code)}")
    
    # Try decoding
    try:
        # In ETL, JIS codes are often stored as Shift-JIS or we need to map them
        # Let's decode the bytes
        jis_bytes = bytes([jis_code >> 8, jis_code & 0xFF])
        char_cp932 = jis_bytes.decode('cp932', errors='ignore')
        print(f"Decoded CP932: {char_cp932}")
    except Exception as e:
        print("Decode error:", e)
        
    # Let's decode the 4-bit image data (76 rows, 72 columns)
    # Each byte contains two 4-bit pixels (high 4 bits and low 4 bits)
    pixels = []
    for b in img_bytes:
        p1 = (b >> 4) & 0x0F
        p2 = b & 0x0F
        pixels.extend([p1, p2])
        
    # Convert 4-bit (0-15) to 8-bit (0-255)
    # 0 is usually white (background), 15 is black (ink), or vice-versa.
    # Let's multiply by 17
    img_array = np.array(pixels, dtype=np.uint8) * 17
    
    # Reshape to 76 height, 72 width
    img_array = img_array.reshape(76, 72)
    
    # Save the image
    img = Image.fromarray(img_array)
    img.save(os.path.join(output_dir, "sample_0.png"))
    print(f"Saved sample image to {output_dir}/sample_0.png")
