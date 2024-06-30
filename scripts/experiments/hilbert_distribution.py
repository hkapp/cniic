import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
csv_data = pd.read_csv(csv_path)

for color in ["red", "green", "blue"]:
    data_series = csv_data[color]
    diffs = data_series.diff()
    distribution = diffs.value_counts()
    print(distribution)
    plt.scatter(distribution.index, distribution, color=color)

plt.show()
