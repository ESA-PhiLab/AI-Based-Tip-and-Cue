import os
import numpy as np
import matplotlib.pyplot as plt



os.chdir("generated_output")


def parse_spd(path):
    wavelengths = []
    values = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('//'):
                continue
            s = s.replace('\t', ' ').replace(',', ' ')
            parts = [p for p in s.split() if p not in [':', ';']]
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            if len(nums) >= 2:
                wavelengths.append(nums[0])
                values.append(nums[1])

    wl = np.array(wavelengths, dtype=float)
    val = np.array(values, dtype=float)
    if wl.size > 0:
        idx = np.argsort(wl)
        wl = wl[idx]
        val = val[idx]
    return wl, val

files = {
    "WV3_Blue.spd": "blue",
    "WV3_Green.spd": "green",
    "WV3_Red.spd": "red",
}

plt.figure(figsize=(9, 5.5))

for fname, color in files.items():
    fp = os.path.join(".", fname)
    wl, val = parse_spd(fp)
    plt.plot(wl, val, color=color, label=fname.split(".")[0])

plt.title("WV3 Spectral Files")
plt.xlabel("Wavelength")
plt.ylabel("Value")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # legend on the side
plt.tight_layout()

out_path = "WV3_SPD_plot_rgb.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {out_path}")
