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

# this script graphs X vs Y for the input CSV and exports an interactive version of the graph to an html file.
# shows search space pruned for a certain X value
# plots untimed points, but colors them gray
# TODO: plots "padded" points, timed and untimed.

def main():
    input = sys.argv[1]              # full path to timed CSV
    output = sys.argv[2]             # full path to output html file (w/o .html extension)
    inputPadded = sys.argv[3]        # full path to padded, timed CSV
    inputPaddedUntimed = sys.argv[4] # full path to padded, untimed CSV
    titleOfWebpage = sys.argv[5]     # title of output webpage
    inputUntimed = sys.argv[6]       # full path to untimed CSV
    divisorAnalyzed = sys.argv[7]    # full path to divisor csv with metrics from TSA
   
    start = len("../sensitivity-analysis/remainder-vs-divisor/div/")
    title = f"{input[start:-4]}"

    # Read in the divisor CSV file
    df = pd.read_csv(input)

    if inputUntimed != "no":    # load and further annotate untimed data
        df_untimed = pd.read_csv(inputUntimed)
        df_untimed = ae.addExtras(df_untimed)    
    if inputPadded != "no":
        print("TODO: HANDLE PADDED DATA")
        df_padded = pd.read_csv(inputPadded)
        df_padded = ae.addExtras(df_padded)
    if inputPaddedUntimed != "no":
        print("TODO: HANDLE PADDED, UNTIMED DATA")
    if divisorAnalyzed != "no":
        df_ann = pd.read_csv(divisorAnalyzed)
        print(df_ann.keys())
        df_ann = ae.addExtras(df_ann)

    df_sorted = df.sort_values(by="Kernel Time", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted   

    if inputUntimed == "no":    # Create interactive scatter plots
        df_merged = df.merge(df_ann,how="outer",on="FakeNN JSON Name")
        df_merged.to_csv("out/merged.csv",index=False)
        html = ig.generateInteractiveGraphs(df_merged, title, titleOfWebpage)
    else:
        if inputPadded != "no":
            print("TODO")
            html = rg_2_23.generateInteractiveGraphsTimedAndUntimedAndPadded(df, title, titleOfWebpage, df_untimed, df_padded)
        else:
            print("TODO")
            html = rg_2_23.generateInteractiveGraphsTimedAndUntimed(df, title, titleOfWebpage, df_untimed)
    
    # --- Write to file ---
    with open(f"{output}.html", "w") as f:
        f.write(html)

    print(f":) Saved as {output} — open it in your browser.")






def specializeMarkers(df, shapeMetric):
    markerOptions = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle",
        "pentagon",
        "hexagram",
        "star",
        "hourglass",
        "bowtie",
        "asterisk",
        "hash",
    ]
    #  markerOptions=[f"$${i}$$" for i in range(0,13)]
    # print(markerOptions)
    tileC_ccs = list(set(list(df[shapeMetric].values)))
    #print(tileC_ccs)
    tileC_ccs.sort()
    #print(tileC_ccs)
    markerOptions=[f"{i}" for i in tileC_ccs]
    marker = dict(zip(tileC_ccs, markerOptions))
    df["Marker"] = df.apply(lambda y: marker[y[shapeMetric]], axis=1)
    # df["Marker"]=df.apply(lambda x: f'${x["absoluteRank"]}$',axis=1)

    # norm = max(list(df["tileC_cc"]))
    # df["MarkerSize"] = (df["tileC_cc"]/norm)*10
    # print(df["MarkerSize"])
    # df["MarkerSize"] = 0.1
    # print(set(list(df["tileC_cc"].values)))
    # for c in tileC_ccs:
    #     print(c)
    # print(df[["JSON Name","Marker"]])
    return df

def get_lines_from_file(file_name):
    """
    Opens a file, reads its contents, and returns a list of strings
    with the trailing newline characters removed.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            # .splitlines() is better than .readlines() because it 
            # automatically strips the '\n' from each string.
            return file.read().splitlines()
    except FileNotFoundError:
        return f"Error: The file '{file_name}' was not found."
    
# python padding-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "padding-naive/Cube128x128x128-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x128" "$FOLDER/128x128x128wm-n-k_searchSpace_c_analyzed-untimed.csv"




if __name__ == "__main__":
    main()
