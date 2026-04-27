import pandas as pd

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    return df
