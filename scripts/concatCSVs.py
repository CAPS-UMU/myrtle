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
        print("missing from left:")
        for x in l.columns:
            if x not in r.columns:
                print(x)
        print("missing from right:")
        for x in r.columns:
            if x not in l.columns:
                print(x)           
        raise Exception("Error: the to csv files contain differing number of columns")

    lr = pd.concat([l[list(l.columns)],r[list(l.columns)]],axis=0, ignore_index=True)
    print(f"Rows in concatted file: {len(lr)}")
    df_cleaned = lr.drop_duplicates(subset=["FakeNN JSON Name"])
    print(f"Rows in concatted file (after removing duplicates): {len(df_cleaned)}")
    df_cleaned.to_csv(outputName, index=False)


if __name__ == "__main__":
    main()
