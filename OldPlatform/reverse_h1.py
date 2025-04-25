import os
import pandas as pd
from pathlib import Path

# Define the directory containing the CSV files
csv_dir = Path("agents/h1_reverse")

# Loop through all CSV files in the directory
for csv_file in csv_dir.glob("*.csv"):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Reverse the DataFrame rows and retain original index
    df_reversed = df[::-1].set_index(df.index)

    # Save it back to the same file (or you can save to a new directory if needed)
    df_reversed.to_csv(csv_file, index=False)