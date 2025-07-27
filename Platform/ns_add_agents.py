import os
import csv
import random

# Paths to the folders
folder_path = 'agents/simulating_social_media_non_partisan_democrats'
democrats_path = 'agents/simulating_social_media_democrats'

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        dem_file_path = os.path.join(democrats_path, filename)

        # Read the non-partisan file
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            rows = list(csv.reader(csvfile))

        # Read the corresponding democrat file (first 3 rows after header)
        with open(dem_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            dem_rows = list(csv.reader(csvfile))[1:4]

        # Insert the democrat rows into index 10-12 (which is row 11-13)
        rows[10:10] = dem_rows

        # Extract rows 1 to 13 (excluding header), shuffle them
        to_shuffle = rows[1:14]
        random.shuffle(to_shuffle)
        rows[1:14] = to_shuffle

        # Write back to the same file
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
