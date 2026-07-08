import pandas as pd
import plotly.express as px
import sys
import os
import matplotlib.pyplot as plt
import plotly.io as pio
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.svm import SVC, SVR
import pickle
import recentGraphs_2_23 as rg_2_23
import addExtraMetrics as ae
import interactiveGraphs as ig

def main():
    left = sys.argv[1]              
    right = sys.argv[2]            
    outputName= sys.argv[3]
    l = pd.read_csv(left)
    r = pd.read_csv(right)
    print(f"Rows in File 1: {len(l)}")
    print(f"Rows in File 2: {len(r)}")
    print(f"Expected total rows: {len(l) + len(r)}")
    if len(l.columns) != len(r.columns):
        print(len(l.columns))
        print(len(r.columns))
        print(l.columns)
        print(r.columns)
        print("missing from right:")
        for x in l.columns:
            if x not in r.columns:
                print(x)
        print("missing from left:")
        for x in r.columns:
            if x not in l.columns:
                print(x)
        common = [c for c in l.columns if c in r.columns]
        if "FakeNN JSON Name" not in common:
            raise Exception("Error: 'FakeNN JSON Name' column is required in both csv files")
        print("Warning: column sets differ, merging on the intersection of columns above")
    else:
        common = list(l.columns)

    lr = pd.concat([l[common],r[common]],axis=0, ignore_index=True)
    # left (first argument) is treated as the newer/authoritative file, so on
    # duplicate keys its row is kept over the right file's
    print(f"Rows in concatted file: {len(lr)}")
    df_cleaned = lr.drop_duplicates(subset=["FakeNN JSON Name"], keep='first')
    print(f"Rows in concatted file (after removing duplicates): {len(df_cleaned)}")
    df_cleaned.to_csv(outputName, index=False)


if __name__ == "__main__":
    main()
