import numpy as np
from pathlib import Path
import math

def inspect_bin_auto_shape(path: str, dtype=np.uint16):
    """Infer image dimensions from file size and print stats."""
    path = Path(path)

    itemsize = np.dtype(dtype).itemsize
    nbytes = path.stat().st_size
    nvals = nbytes // itemsize

    print("File:", path)
    print("Bytes:", nbytes)
    print("Values:", nvals)

    # Try common satellite widths
    common_widths = [4096, 3072, 2048, 1024]

    for w in common_widths:
        if nvals % w == 0:
            h = nvals // w
            print(f"Possible shape: {h} × {w}")

    # Pick width = 4096 if possible
    if nvals % 4096 == 0:
        width = 4096
        height = nvals // 4096
    else:
        width = int(math.sqrt(nvals))
        height = nvals // width

    img = np.fromfile(path, dtype=dtype).reshape((height, width))

    img_bits = 2**16


    print("Chosen shape:", img.shape)
    print("Min :", int(img.min()), img.min()/img_bits)
    print("Max :", int(img.max()), img.max()/img_bits)
    print("Mean:", float(img.mean()), img.mean()/img_bits)

    return img


inspect_bin_auto_shape(
   "dataset/PHISAT-2_CLOUD_000004087_20251214122417_20251214122420_DC68DF9D/"
   "intermediate_output/post_proc_bands/AC/"
   "Bp_0_0_4096_4096_0_0_4096_4096_12_1.bin"
   )
