import os
from PIL import Image
import matplotlib.pyplot as plt

data_dir = 'data/DIV2K_valid_HR'

# Gather the stats

stats = dict()

for png_name in os.listdir(data_dir):
    png_path = os.path.join(data_dir, png_name)

    png_size = os.path.getsize(png_path)
    stats.setdefault("png_size", []).append(png_size)

    png_img = Image.open(png_path)
    (w, h) = png_img.size
    raw_size = w * h * 24   # see the comments in bench.rs
    compression_ratio = png_size / raw_size

# Generate the plot

# plt.ylabel('Compression ratio (lower is better)')
# plt.ylim(0, 100)

print(stats)

plt.gca().set_xticklabels(stats.keys())

plt.boxplot(stats.values())
plt.savefig("output/png_stats.png")
plt.show()
