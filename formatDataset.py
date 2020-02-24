# Execute this code just to separate images into folders where each folder correspond to a class.
# This is used for training purposes only.

import csv
import os

with open('./dataset/train.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        os.makedirs(f'./dataset/train/{row["label"]}', exist_ok=True)
        os.replace(f'./dataset/train_img/{row["image_id"]}.png', f'./dataset/train/{row["label"]}/{row["image_id"]}.png')
        #print(f'\t{row["image_id"]} \t {row["label"]}')
        line_count += 1
    print(f'Processed {line_count} lines.')