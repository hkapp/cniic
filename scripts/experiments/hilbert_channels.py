import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
csv_data = pd.read_csv(csv_path)

rstart = 0 #64 * int(len(csv_data["red"])/100)
rend = len(csv_data["red"]) #65 * int(len(csv_data["red"])/100)
plt.plot(csv_data["red"][rstart:rend], color="red")
plt.plot(csv_data["green"][rstart:rend], color="green")
plt.plot(csv_data["blue"][rstart:rend], color="blue")
plt.show()
