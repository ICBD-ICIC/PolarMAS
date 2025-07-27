import os
import csv
import random

# Path to the folder containing CSVs
folder_path = 'agents/simulating_social_media_non_partisan_democrats/broken_democrat_v5_2'

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Read all rows
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            rows = list(csv.reader(csvfile))

        # Interchange row 2 and 11 (index 1 and 10)
        #rows[1], rows[10] = rows[10], rows[1]

        sub_rows = rows[1:14]
        random.shuffle(sub_rows)
        rows[1:14] = sub_rows

        # Write back to the same file
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

