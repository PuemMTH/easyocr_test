import os
import random
import matplotlib.pyplot as plt
import easyocr
import csv
import pandas as pd

# Paths - ให้ output อยู่ที่เดียวกับ input
input_file = '/project/lt200384-ff_bio/puem/ocr/dataset/processed_lmdb_format/test/labels.txt'
output_file = '/project/lt200384-ff_bio/puem/ocr/dataset/processed_lmdb_format/test/labels.csv'

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

with open(input_file, 'r', encoding='utf-8') as txtfile, open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'words'])
    for line in txtfile:
        parts = line.rstrip('\n').split('\t', 1)
        if len(parts) == 2:
            writer.writerow(parts)
