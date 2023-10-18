import os
import csv
from PIL import Image

output_path = 'output/png.csv'
data_dir = 'data/DIV2K_valid_HR'

with open(output_path, 'w') as csv_file:
    csv_w = csv.writer(csv_file)
    csv_w.writerow(['name', 'compressed_size', 'compression_ratio'])
    for png_name in os.listdir(data_dir):
        png_path = os.path.join(data_dir, png_name)

        png_size = os.path.getsize(png_path)

        png_img = Image.open(png_path)
        (w, h) = png_img.size
        raw_size = w * h * 24   # see the comments in bench.rs
        compression_ratio = png_size / raw_size

        csv_w.writerow([png_path, png_size, compression_ratio * 100])
