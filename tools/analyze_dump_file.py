import glob
import os
import pandas as pd
import sys

from pandas import DataFrame


folder1 = sys.argv[1]
folder2 = sys.argv[2]


def relate_error_over_column(df1:DataFrame,
                             df2:DataFrame,
                             columns:list) -> DataFrame:
    df1, df2 = df1[columns], df2[columns]
    diff = df1 - df2
    return (diff / df1).abs()


def compare_over_folder(folder1, folder2):
    if not os.path.exists(folder1):
        print(f'not exists {folder1}')
        return
    if not os.path.exists(folder2):
        print(f'not exists {folder2}')
        return

    print(f"folder: {folder1}\n"
          f"        {folder2}")
    
    for fname in glob.glob("*.parquet", root_dir=folder1):
        f1 = os.path.join(folder1, fname)
        f2 = os.path.join(folder2, fname)

        if not os.path.exists(f2):
            print(f'not found {f2}')
            continue

        print(f" > analyze: {fname}")
        df1 = pd.read_parquet(f1)
        df2 = pd.read_parquet(f2)

        ret = relate_error_over_column(df1, df2, df1.columns)
        ret.to_csv(f1+'.csv')


# assume 2 folder has same rank sub-folder
folder1 = {p.split('-')[-1]: p for p in glob.glob(folder1+"/*-rk*")}
folder2 = {p.split('-')[-1]: p for p in glob.glob(folder2+"/*-rk*")}
folder_pair = [(folder1[rk], folder2[rk]) for rk in folder1]

for f1, f2 in folder_pair:
    compare_over_folder(f1, f2)
