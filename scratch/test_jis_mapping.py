import bitstring

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"
f = bitstring.ConstBitStream(filename=filepath)

# Mapping from JIS X 0201 Katakana to Hiragana
# JIS X 0201 Katakana characters start at 0xA1 (｡) to 0xDF (ﾟ).
# 0xB1 is ｱ (a), 0xB2 is ｲ (i), 0xB3 is ｳ (u), 0xB4 is ｴ (e), 0xB5 is ｵ (o), etc.
# Let's map these to Hiragana:
jis_to_hiragana = {
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
    # Include small ones
    0xA7: "ぁ", 0xA8: "ぃ", 0xA9: "ぅ", 0xAA: "ぇ", 0xAB: "ぉ",
    0xC3: "て", # wait, small tsu is 0xAF (ッ) -> っ
    0xAF: "っ",
    0xAC: "ゃ", 0xAD: "ゅ", 0xAE: "ょ",
}

for i in range(51):
    pos = i * 120
    f.bytepos = pos * 2952
    r = f.readlist('2*uint:36,uint:8,pad:28')
    char_code = r[2]
    hira = jis_to_hiragana.get(char_code, "unknown")
    print(f"Record {pos:4d}: char_code = {char_code} (hex: {hex(char_code)}) -> Hiragana: {hira}")
