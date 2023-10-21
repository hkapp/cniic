import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np

data_dir = 'data/DIV2K_valid_HR'

# Gather the stats

stats = dict()
dimensions = dict()

for png_name in os.listdir(data_dir):
    png_path = os.path.join(data_dir, png_name)

    # png_size
    png_size = os.path.getsize(png_path)
    stats.setdefault("png_size", []).append(png_size / 1e6)

    png_img = Image.open(png_path)
    # dimensions
    (w, h) = png_img.size
    dims = str(w) + 'x' + str(h)
    dims_count = dimensions.get(dims, 0) + 1
    dimensions[dims] = dims_count

    # ncolors
    # np_img = np.asarray(png_img)
    # ncolors = np.unique(np_img)  # this takes 26s (user time)
    # stats.setdefault("ncolors", []).append(ncolors)
    stats.setdefault("ncolors", []).append(len(png_img.getcolors(w*h)))  # this takes 19s (user time)
    # stats.setdefault("ncolors", []).append(1)

    # png_path (used by color distribution)
    stats.setdefault("png_path", []).append(png_path)


# Generate the plot

fig, subp = plt.subplots(2, 2, figsize=(10, 10))

# png_size
p = subp[0, 0]
p.boxplot(stats["png_size"], showmeans=True)
p.set_ylabel('Size (MB)')
p.set_title('png_size')

# dimensions
p = subp[0, 1]
# Sort the bar plot
sorted_keys = sorted(dimensions, key=dimensions.get, reverse=True)
sorted_values = [dimensions[k] for k in sorted_keys]
# Bundle the singles at the far right
first_single = sorted_values.index(1)
bar_keys = sorted_keys[:first_single]
bar_keys.append("singles")
bar_values = sorted_values[:first_single]
bar_values.append(len(sorted_values) - first_single)
# bar plot
p.bar(bar_keys, bar_values)
p.set_title('dimensions')
p.set_xticklabels(bar_keys, rotation=60)

# ncolors
p = subp[1, 0]
p.boxplot(stats["ncolors"], showmeans=True)
p.set_ylabel('Number of colors')
p.set_title('ncolors')

# color distribution
rand_path = random.choice(stats["png_path"])
rand_img = Image.open(rand_path)
np_img = np.asarray(rand_img)
colors, color_counts = np.unique(np_img, return_counts=True)
# plot
p = subp[1, 1]
p.bar(range(len(color_counts)), sorted(color_counts))
p.set_ylabel('Occurrences')
p.set_title('Color distribution for ' + rand_path)


fig.tight_layout()
plt.savefig("output/png_stats.png")
# plt.show()
