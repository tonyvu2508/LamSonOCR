import struct
from collections import Counter

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"

labels = []
with open(filepath, "rb") as f:
    for idx in range(6120):
        record = f.read(2952)
        if len(record) < 2952:
            break
        
        # Let's extract the first 10 shorts (20 bytes)
        shorts = struct.unpack(">10H", record[:20])
        
        # Let's also look at the first 30 bytes as different data types
        # C-type header structure (usually):
        # 0: Serial Sheet (2 bytes)
        # 2: JIS Code (2 bytes or similar)
        # 4: EBCDIC Code or JIS / Shift-JIS or similar?
        # Let's print the shorts for the first 5 records
        if idx < 5:
            print(f"Record {idx}: shorts = {shorts}")
            # Try to print cp932/shift-jis for each short
            for i, val in enumerate(shorts):
                try:
                    b = bytes([val >> 8, val & 0xFF])
                    dec = b.decode('cp932', errors='ignore')
                    # remove non-printable/control chars
                    dec = "".join(c for c in dec if c.isprintable())
                    print(f"  Short {i}: {val} (hex: {hex(val)}) -> dec: '{dec}'")
                except:
                    pass
        
        # Let's also try decoding the first 30 bytes as string or individual bytes
        # Some C-type headers store character codes as a JIS/Shift-JIS byte pair or a single byte.
        # Let's count occurrences of the short at index 2, 3, 4, etc.
        # In our previous run:
        # shorts[2] was 16384 (0x4000)
        # shorts[4] was 35249 (0x89b1) -> "桶"
        # Let's record index 4
        labels.append(shorts[4])

print("\nFrequency of shorts[4] values (top 10):")
counter = Counter(labels)
for val, count in counter.most_common(10):
    try:
        b = bytes([val >> 8, val & 0xFF])
        char = b.decode('cp932', errors='ignore')
    except:
        char = "error"
    print(f"Value: {val} (hex: {hex(val)}) -> Char: '{char}' | Count: {count}")
