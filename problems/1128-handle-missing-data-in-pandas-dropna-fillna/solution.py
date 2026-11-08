import pandas as pd

def solution(df):
    df = df.dropna(axis = 1, thresh = len(df)/2)
    df = df.dropna(axis = 0, thresh = len(df.columns) /2)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    return df.reset_index(drop = True) # Without drop=True, pandas keeps the old index as a new column
    # So, drop=True means: Reset the index and do not save the old index as a column.