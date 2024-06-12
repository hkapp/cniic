import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

csv_folder = 'output/'

# Retrieve the data and names from the various csv files

names = []
data = []

for csv_path in glob.glob(csv_folder + "*.csv"):
    csv_data = pd.read_csv(csv_path)
    # Only keep the lossless codecs
    if not "error" in csv_data or csv_data["error"].mean() == 0:
        data.append(csv_data["compression_ratio"])

        # Go from 'dir/filename.csv' to 'filename'
        file_name = os.path.split(csv_path)[1]
        display_name = os.path.splitext(file_name)[0]
        names.append(display_name)

# Generate the plot

ax = plt.gca()

plt.ylabel('Compression ratio (lower is better)')
plt.ylim(0, 100)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)+'%'))

ax.set_xticklabels(names)

plt.boxplot(data, showmeans=True)
plt.savefig("output/boxplot.png")
plt.show()
