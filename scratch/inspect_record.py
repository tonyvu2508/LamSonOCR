import bitstring
import numpy as np
from PIL import Image
import os

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"

f = bitstring.ConstBitStream(filename=filepath)

# Let's read the first 10 records and look at Field 14 (index 14) and decode it
for pos in range(10):
    f.bytepos = pos * 2952
    r = f.readlist('2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,15*uint:36,pad:1008,bytes:2736')
    jis_code = r[14]
    
    # Standard conversion from JIS code (2 bytes) to CP932/Shift_JIS:
    # Or try translating to bytes and decode as 'iso2022_jp' (which is the standard JIS X 0208 codec in python)
    # The JIS code from ETL is usually in hex, e.g. 0x2422 (あ). Let's see:
    try:
        # iso-2022-jp escape sequence for JIS X 0208
        jis_bytes = b'\x1b$B' + bytes([jis_code >> 8, jis_code & 0xFF]) + b'\x1b(B'
        char_iso = jis_bytes.decode('iso2022_jp')
    except Exception as e:
        char_iso = f"error: {e}"
        
    print(f"Record {pos}: Field 14 = {jis_code} (hex: {hex(jis_code)}) -> JIS Char: '{char_iso}'")

# Reshape the first one
f.bytepos = 0
r = f.readlist('2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,15*uint:36,pad:1008,bytes:2736')
img_bytes = r[23]

pixels = []
for b in img_bytes:
    p1 = (b >> 4) & 0x0F
    p2 = b & 0x0F
    pixels.extend([p1, p2])

img_array = np.array(pixels, dtype=np.uint8) * 17
img_array = img_array.reshape(76, 72)

# Save image
os.makedirs("/Volumes/SpaceX/WorkSpace/python/LamSonOcr/scratch", exist_ok=True)
img = Image.fromarray(img_array)
img.save("/Volumes/SpaceX/WorkSpace/python/LamSonOcr/scratch/bitstring_sample.png")
print("Saved image to /Volumes/SpaceX/WorkSpace/python/LamSonOcr/scratch/bitstring_sample.png")
