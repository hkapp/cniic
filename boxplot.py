import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

csv_folder = 'output/'

# Retrieve the data and names from the various csv files

names = []
data = []

for csv_path in glob.glob(csv_folder + "*.csv"):
    csv_data = pd.read_csv(csv_path)
    data.append(csv_data["compression_ratio"])

    # Go from 'dir/filename.csv' to 'filename'
    file_name = os.path.split(csv_path)[1]
    display_name = os.path.splitext(file_name)[0]
    names.append(display_name)

# Generate the plot

plt.ylabel('Compression ratio (lower is better)')
plt.ylim(0, 100)

plt.gca().set_xticklabels(names)

plt.boxplot(data, showmeans=True)
plt.savefig("output/boxplot.png")
plt.show()
