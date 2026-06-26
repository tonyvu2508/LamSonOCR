import os
import struct

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"

if not os.path.exists(filepath):
    print(f"Error: File {filepath} does not exist.")
    exit(1)

filesize = os.path.getsize(filepath)
print(f"File size: {filesize} bytes")
print(f"Expected number of records (size / 2952): {filesize / 2952:.1f}")

with open(filepath, "rb") as f:
    # Read the first record
    record = f.read(2952)
    print(f"Read first record: {len(record)} bytes")
    
    # Print the first 100 bytes as hex and representation
    print("First 100 bytes hex:")
    print(record[:100].hex())
    
    # Let's see some metadata
    # The format of ETL3, ETL4, ETL5 C-type is typically:
    # 0-1: Serial sheets (short)
    # 2-3: Serial number (short)
    # 4-5: Character code (short or JIS)
    # Let's try printing different interpretations of the header
    print("Unpacked short integers (first 20 bytes):")
    shorts = struct.unpack(">10H", record[:20])
    print(shorts)
    
    # JIS code is typically at offset 2 (4-5 bytes or 1 word)
    # Let's check typical shift_jis decode
    jis = shorts[2]
    print(f"JIS Code at index 2: {jis} (Hex: {hex(jis)})")
    
    # Let's print chars around it
    try:
        jis_bytes = bytes([jis >> 8, jis & 0xFF])
        print(f"Decoded JIS bytes (Shift_JIS): {jis_bytes.decode('shift_jis', errors='ignore')}")
        print(f"Decoded JIS bytes (CP932): {jis_bytes.decode('cp932', errors='ignore')}")
        print(f"Decoded JIS bytes (EUC-JP): {jis_bytes.decode('euc_jp', errors='ignore')}")
    except Exception as e:
        print("Decode error:", e)
