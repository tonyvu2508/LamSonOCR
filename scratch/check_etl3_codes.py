import bitstring

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL3/ETL3C_1"
f = bitstring.ConstBitStream(filename=filepath)

for i in range(10):
    pos = i * 200  # ETL3 has 200 samples per class
    f.bytepos = pos * 2952
    r = f.readlist('2*uint:36,uint:8,pad:28')
    char_code = r[2]
    # Try converting to character if it's ascii
    try:
        char = chr(char_code)
    except:
        char = "unknown"
    print(f"Record {pos:4d}: char_code = {char_code} (hex: {hex(char_code)}) -> char: '{char}'")
