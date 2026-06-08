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
import visualizer as viz

# this script take in 1 CSVs: the full search space timed and annotated.
def addFakeKernelTime(df_ut, df_t):
    avgTime = sum(df_t["Kernel Time"].values) / len(df_t["Kernel Time"].values)
    df_ut["Kernel Time"] = avgTime
    df_ut["dma"] = avgTime
    df_ut["absoluteRank"] = -1
    df_ut["Overlap Stall Time Total"] = -1
    df_ut["Raw Compute Time Total"] = -1
    df_ut["timeout"] = False
    return df_ut

def main():
    timed = sys.argv[1]             # full path to timed CSV
    titleOfWebpage = sys.argv[2]     
    htmlName = sys.argv[3]    

    # Read in the timed CSV file
    df = pd.read_csv(timed)
    ae.addExtrasQ(df)

    # Find the row timeout row if it exists, and pull out its dma cycle value
    matching_rows = df.loc[df["FakeNN JSON Name"] == "timeout", "dma"]
    if not matching_rows.empty:
        raise Exception("Detected timeout but expecting Quidditch data (no timeout!)")
    else:
        df["timeout"]=False
        print("Warning: 'timeout' row not found!.")

    # rank timed points
    df_sorted = df.sort_values(by="dma", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted 
    

    html = viz.visualizePruningQ(df, titleOfWebpage)
    
    # --- Write to file ---
    with open(f"{htmlName}.html", "w") as f:
        f.write(html)

    print(f":) Saved as {htmlName} — open it in your browser.")








if __name__ == "__main__":
    main()
