import pandas as pd
import os
import glob
import openpyxl

file = "./1hour_1.xlsx"
output_file = "./merged_results/1hour_1.xlsx"

df = pd.read_excel(file)
df = df.dropna()
df.to_excel(output_file, index=False, engine = 'openpyxl')
print("done")