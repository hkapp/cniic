import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + ' <csv file>')
    exit(1)

csv_path = sys.argv[1]

csv_data = pd.read_csv(csv_path)
print(csv_data)

plt.ylim(0, 100)

# Go from 'dir/filename.csv' to 'filename'
file_name = os.path.split(csv_path)[1]
display_name = os.path.splitext(file_name)[0]
# https://stackoverflow.com/questions/37039685/hide-tick-label-values-but-keep-axis-labels
plt.xticks(color='w')
plt.xlabel(display_name)

plt.boxplot(csv_data["compression_ratio"])
plt.savefig("output/boxplot.png")
plt.show()
