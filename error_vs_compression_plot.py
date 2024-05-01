import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

csv_folder = 'output/'

# Split the files according to codec
# This allows making separate data series for the scatter plot

data_series = dict()

for csv_path in glob.glob(csv_folder + "*.csv"):
    # Retrieve the codec name between the last '/' and the first '_'
    file_name = os.path.split(csv_path)[1]
    display_name = os.path.splitext(file_name)[0]
    codec_name = display_name.split('_')[0]

    if not data_series.get(codec_name):
        data_series[codec_name] = [csv_path]
    else:
        data_series[codec_name].append(csv_path)

# Scatter plot each codec as a different data series

for (codec_name, codec_files) in data_series.items():
    x = []
    y = []
    for csv_path in codec_files:
        csv_data = pd.read_csv(csv_path)

        compression = csv_data["compression_ratio"].mean() / 100
        x.append(compression)

        if "error" in csv_data.columns:
            err = csv_data["error"].mean()
        else:
            err = 0
        y.append(err)

    plt.scatter(x, y, label=codec_name)

# Generate the plot

plt.xlabel('Compression ratio (lower is better)')
# plt.xlim(0, 100)

plt.ylabel('Error (lower is better)')

# plt.gca().set_xticklabels(names)

ax = plt.gca()
# for i, label in enumerate(names):
    # ax.annotate(label, (x[i]+1, y[i]+1))

ax.set_xscale('log')
ax.set_yscale('symlog')
plt.ylim(-1)

plt.legend()

plt.savefig("output/error_vs_compression.png")
plt.show()
