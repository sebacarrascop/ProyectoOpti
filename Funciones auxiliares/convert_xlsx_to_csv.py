import os
import pandas as pd

def convert_xlsx_to_csv(path):
    for filename in os.listdir(path):
        if filename.endswith('.xlsx'):
            # Lee el archivo .xlsx
            df = pd.read_excel(os.path.join(path, filename), engine='openpyxl')
            
            # Define el nombre del archivo .csv
            filename_no_extension = os.path.splitext(filename)[0]
            csv_filename = os.path.join(path, f"{filename_no_extension}.csv")
            
            # Guarda el archivo .csv
            df.to_csv(csv_filename, index=False)
            print(f"Converted {filename} to {csv_filename}")

# Especifica el camino a la carpeta con archivos .xlsx
convert_xlsx_to_csv('data/LasCondes')