import bitstring

filepath = "/Volumes/SpaceX/WorkSpace/python/LamSonOcr/ETL/ETL4/ETL4C"

# Open file
f = bitstring.ConstBitStream(filename=filepath)

# Read first record as 3936 6-bit unsigned integers
r = f.readlist('3936*uint:6')

print("First 30 6-bit characters in decimal:")
print(r[:30])

# T56 character mapping from shared.py
t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'

# Decode first 30 characters using T56
decoded_t56 = []
for val in r[:30]:
    if val < len(t56s):
        decoded_t56.append(t56s[val])
    else:
        decoded_t56.append('?')

print("First 30 decoded as T56 string:")
print("".join(decoded_t56))
