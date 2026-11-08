import pandas as pd

def solution(df):
    df = df[df["status"] == "completed"]
    df = df.groupby("region", as_index = False)["amount"].sum()
    # Group by region, add the amounts, and keep region as a normal column.
    df = df.sort_values("amount",ascending = False).head(10)
    return df.reset_index(drop = True)