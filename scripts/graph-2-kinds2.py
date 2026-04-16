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
import interactiveGraphs3 as ig

# this script graphs X vs Y for the input CSV and exports an interactive version of the graph to an html file.
# shows search space pruned for a certain X value
# plots untimed points, but colors them gray
# TODO: plots "padded" points, timed and untimed.

def addFakeKernelTime(df_ut, df_t):
    avgTime = sum(df_t["Kernel Time"].values) / len(df_t["Kernel Time"].values)
    df_ut["Kernel Time"] = avgTime
    df_ut["dma"] = avgTime
    df_ut["absoluteRank"] = -1
    df_ut["Overlap Stall Time Total"] = -1
    df_ut["Raw Compute Time Total"] = -1
    return df_ut

def main():
    timed = sys.argv[1]              # full path to timed CSV
    analyzed = sys.argv[2]           # full path to csv with metrics from TSA
    titleOfWebpage = sys.argv[3]     # title of output webpage
    htmlName = sys.argv[4]
    mode=sys.argv[5]

    # Read in the timed CSV file
    df = pd.read_csv(timed)
    # read in the analysis csv file
    df_ann = pd.read_csv(analyzed)
    df_ann = ae.addExtras(df_ann)
   
    # print(df.shape)
    # print(df_ann.shape)
    # print("after merge")
    # merge timed with analysis
    df_merged = df.merge(df_ann,how="left",on="FakeNN JSON Name")
    df = df_merged
    # sort merged by e2e dma execution time
   # print(df.shape)
    df_sorted = df.sort_values(by="dma", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted 
    df_ann = ae.addFakeKernelTime(df_ann, df)
  #  print(df["remainderTiles"])
    if mode == "stallCyclesTimed":
         #html = ig.generateExperimentalPruningGraphs(df, df_ann, titleOfWebpage)
         html="hoodle"
    else:
        html = ig.generateExperimentalPruningGraphs(df, df_ann, titleOfWebpage)
    
    # --- Write to file ---
    with open(f"{htmlName}.html", "w") as f:
        f.write(html)

    print(f":) Saved as {htmlName} — open it in your browser.")








if __name__ == "__main__":
    main()
