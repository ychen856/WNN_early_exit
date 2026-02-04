import json
ms = json.load(open('addr_bits_per_lut.json'))  # [m0, m1, ...]
with open('addr_bits_per_lut.mem','w') as f:
    for m in ms: f.write(f"{m:x}\n")   # hex


KB = json.load(open('kept_bits.json'))  # [[b0..b_m-1], ...]
MAX_M=7
with open('kept_bits_flat.mem','w') as f:
    for row in KB:
        row = list(row)+[0]*(MAX_M-len(row))
        f.write(" ".join(f"{x:x}" for x in row)+"\n")


C=10; COUNT_BITS=16
import re, pathlib
for coe in pathlib.Path('coe').glob('lut_*.coe'):
    text = coe.read_text()
    vec  = re.search(r"memory_initialization_vector\s*=\s*(.*);", text, re.S).group(1)
    rows = [r.strip() for r in vec.split(",") if r.strip()]
    with open(f"mem/{coe.stem}.mem","w") as f:
        for r in rows:
            nums = [int(x) for x in r.split(",")]
            word = 0
            for c,v in enumerate(nums):
                word |= (v & ((1<<COUNT_BITS)-1)) << (c*COUNT_BITS)
            f.write(f"{word:x}\n")


