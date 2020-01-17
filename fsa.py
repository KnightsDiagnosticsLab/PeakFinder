import csv
import pandas as pd

def use_csv_module(filename):
    with open(filename, newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        l = [r for r in reader]
    headers = l.pop(0)
    df = pd.DataFrame(l, columns=headers)
    df.replace(r'^\s*$', pd.np.nan, regex=True, inplace=True)
    # print(df)
    return df
