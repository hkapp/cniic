import os
from PIL import Image
import matplotlib.pyplot as plt

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
    stats.setdefault("ncolors", []).append(len(png_img.getcolors(w*h)))

# Generate the plot

fig, subp = plt.subplots(1, 3, figsize=(15, 5))

# png_size
p = subp[0]
p.boxplot(stats["png_size"], showmeans=True)
p.set_ylabel('Size (MB)')
p.set_title('png_size')

# dimensions
p = subp[1]
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
# Make enough space for the rotated labels
plt.subplots_adjust(bottom=0.20)

# ncolors
p = subp[2]
print(stats["ncolors"])
p.boxplot(stats["ncolors"], showmeans=True)
p.set_ylabel('Number of colors')
p.set_title('ncolors')

plt.savefig("output/png_stats.png")
plt.show()
