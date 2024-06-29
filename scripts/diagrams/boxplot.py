import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cniic

def unzip(ls):
    return list(zip(*ls))

def sort_data(data, names):
    zipped = list(zip(data, names))
    zipped.sort(key = lambda pair: pair[0].mean())
    return unzip(zipped)

csv_folder = cniic.output_folder()

# Retrieve the data and names from the various csv files

names = []
data = []

for csv_path in cniic.diagram_csvs():
    csv_data = pd.read_csv(csv_path)
    # Only keep the lossless codecs
    if not "error" in csv_data or csv_data["error"].mean() == 0:
        data.append(csv_data["compression_ratio"])

        # Go from 'dir/filename.csv' to 'filename'
        file_name = os.path.split(csv_path)[1]
        display_name = os.path.splitext(file_name)[0]
        names.append(display_name)

(data, names) = sort_data(data, names)

# Generate the plot

ax = plt.gca()

plt.ylabel('Compression ratio (lower is better)')
plt.ylim(0, 100)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)+'%'))

ax.set_xticklabels(names)

plt.boxplot(data, showmeans=True)
plt.savefig(os.path.join(cniic.output_folder(), "boxplot.png"))
plt.show()
