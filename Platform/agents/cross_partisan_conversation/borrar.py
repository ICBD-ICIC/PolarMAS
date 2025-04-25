import os
import pandas as pd
from sqlalchemy.sql.base import elements

# Obtener todos los archivos CSV en el directorio actual
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

# Recorrer cada archivo CSV
for file in csv_files:
    number = int(file.split('_')[-1].replace('.csv', ''))
    if number % 2 != 0: #republican starts
        df = pd.read_csv(file)
        df = df[::-1].set_index(df.index)
        df.to_csv(f'{file}', index=False)
