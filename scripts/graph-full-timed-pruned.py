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

# this script take in 3 CSVs: the full search space, the timed points, the pruned, annotated search space
# we assume the timed points are a subset of the pruned, annotated search space

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
    pruned_analyzed = sys.argv[2]   # full path to csv annotated, pruned CSV
    full=sys.argv[3]                # full path to full search space CSV
    titleOfWebpage = sys.argv[4]     
    htmlName = sys.argv[5]    

    # Read in the timed CSV file
    df = pd.read_csv(timed)

    # Find the row timeout row if it exists
    matching_rows = df.loc[df["FakeNN JSON Name"] == "timeout", "dma"]
    if not matching_rows.empty:
        max_dma = df["dma"].max()
        timeout_dma_value=max_dma*1.5
        # Apply the updates to fields "dma" and "Global Sim E2E_dma"
        df["timeout"] = df["dma"] == -1
        df.loc[df["dma"] == -1, "dma"] = timeout_dma_value
        df.loc[df["Global Sim E2E_dma"] == -1, "Global Sim E2E_dma"] = timeout_dma_value
    else:
        df["timeout"]=False
        print("Warning: 'timeout' row not found!.")

    # read in the pruned analysis csv file
    df_ann = pd.read_csv(pruned_analyzed)
    # compute derived features
    df_ann = ae.addExtras(df_ann)

    # merge timed with analysis
    df_merged = df.merge(df_ann,how="left",on="FakeNN JSON Name")
    df = df_merged
    # rank timed points
    df_sorted = df.sort_values(by="Global Sim E2E_dma", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted 
    # give analyzed points fake time data
    df_ann = ae.addFakeKernelTime(df_ann, df)
    df_ann["timeout"] = False
    # eventually load FULL search space
    df_full = pd.read_csv(full)

    html = viz.visualizePruning(df, df_ann, df_full, titleOfWebpage)
    
    # --- Write to file ---
    with open(f"{htmlName}.html", "w") as f:
        f.write(html)

    print(f":) Saved as {htmlName} — open it in your browser.")








if __name__ == "__main__":
    main()
