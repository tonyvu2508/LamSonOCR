import struct

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"

# Read records at 0, 120, 240, 360, 480, 600, etc.
with open(filepath, "rb") as f:
    for i in range(15):
        pos = i * 120
        f.seek(pos * 2952)
        record = f.read(2952)
        if len(record) < 2952:
            break
        shorts = struct.unpack(">10H", record[:20])
        print(f"Record {pos:4d}: shorts[1]={shorts[1]} ({hex(shorts[1])}), shorts[2]={shorts[2]} ({hex(shorts[2])}), shorts[4]={shorts[4]} ({hex(shorts[4])})")
