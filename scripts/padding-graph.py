# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Step 1: Read the CSV file ---
# df = pd.read_csv("review-phenomizer.csv")

# # --- Step 2: Choose which columns to plot ---
# x_col = "Regular Loads"   # column for the x-axis
# y_col = "Kernel Time"   # column for the y-axis

# # --- Step 3: Plot ---
# plt.figure(figsize=(8, 5))
# plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color='b')

# # --- Step 4: Label and show ---
# plt.title(f"{y_col} vs {x_col}")
# plt.xlabel(x_col)
# plt.ylabel(y_col)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

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
import scripts.recentGraphs_2_23 as rg_2_23

# this script graphs X vs Y for the input CSV and exports an interactive version of the graph to an html file.
# shows search space pruned for a certain X value
# plots untimed points, but colors them gray
# TODO: plots "padded" points, timed and untimed.

def addFeatures(df):
    df["Hardware Loops"] = df["M"] * df["N"] * df["K"] / (8 * df["k"])
    df["SSR Configs"] = df["SSR Config Count"]
    df["L1 Usage"] = df["Space Needed in L1"]
    df["m-n-k"] = df["JSON Name"]
    df["L1 Loads"] = df["Regular Loads"]
    return df


def addCost(df, c1=1.0,c2=1.0):
    # c1=5461.0/107684
    # c2=8.0
    c1 = 27552.0
    c2 = 131072.0
    df["sumSSRsRegs"] = (df["SSR Config Count"] + df["Regular Loads"])# * df["L3 Loads"]
    df["regPerStream"] = df["n"] * df["m"] / (128.0 * df["k"])
    df["mk/n"] = df["m"] * df["k"] / (1.0 * df["n"])
    df["fmaddsPerCore"] = df["m"] * df["n"] * df["k"] / 8
    df["L3 Loads Timed"] = df["L3 Loads"] - df["tileC"] - df["tileA"] - df["tileB"]
    df["L3 Stores Timed"] = df["M"] * df["N"] - df["m"] * df["n"]
    df["L3 L/S Timed"] = df["L3 Stores Timed"] + df["L3 Loads Timed"]
    df["CC L1 Footprint"] = df.apply(
        lambda y: (y["m"] * y["n"] + y["m"] * y["k"]) / 8.0 + y["k"] * y["n"], axis=1
    )
    df["L1/CC L1"] = df["L1 Usage"] + df["CC L1 Footprint"]
    df["k/n"] = df["k"] / 1.0 / df["n"]#(df["tileA_cc"] / 1.0 / df["tileC_cc"])
    df["SSRconfigsXregPerStream"] = df["SSR Config Count"]*1.0 * df["regPerStream"]
    df["L1UsageXregPerStream"] = df["L1 Usage"]*1.0 * df["regPerStream"]
    df["CCL1FootprintXregPerStream"] = df["CC L1 Footprint"]*1.0 * df["regPerStream"]
    df["k/nXregPerStream"] = df["k/n"] * df["regPerStream"]
    cmFeatures = [
        "fmaddsPerCore",
        "L3 L/S Timed",
        "CC L1 Footprint",
        "L3 Loads",
        "SSR Configs",
        "Hardware Loops",
        "m",
        "n",
        "k",
        "tileA",
        "tileB",
        "tileC",
        "tileA_cc",
        "tileC_cc",
        "A SSR Reuse Loads",
        "A Not Reused SSR Loads",
        "B SSR Loads",
    ]
    #df["cost"] = df[cmFeatures].sum(axis=1)
    #df["cost"] =1.0* df["SSR Config Count"]/c1 - df["k/n"]*(c2/c1)
    #df["cost"] = df["SSR Config Count"] - df["k/n"]*df["Hardware Loops"] - df["k/n"] - df["L1 Usage"]
    df["cost"] = df["regPerStream"]
    #df["cost"] = df["k/n"]*df["Hardware Loops"]
    return df, cmFeatures

def addFx(df):
    df["fmaddsPerCore"] = df["m"] * df["n"] * df["k"] / 8
    df["L3 Loads Timed"] = df["L3 Loads"] - df["tileC"] - df["tileA"] - df["tileB"]
    df["L3 Stores Timed"] = df["M"] * df["N"] - df["m"] * df["n"]
    df["L3 L/S Timed"] = df["L3 Stores Timed"] + df["L3 Loads Timed"]
    df["CC L1 Footprint"] = df.apply(
        lambda y: (y["m"] * y["n"] + y["m"] * y["k"]) / 8.0 + y["k"] * y["n"], axis=1
    )
    cmFeatures = [
        "fmaddsPerCore",
        "L3 L/S Timed",
        "CC L1 Footprint",
    ]
    #df["fx"] = df["L3 L/S Timed"] + -df["fmaddsPerCore"] - df["CC L1 Footprint"]
   # df["fx"] = (1/df["k"])*df["tileC_cc"] - df["fmaddsPerCore"]
    df["fx"] = df["SSR Configs"]/df["fmaddsPerCore"] *df["CC L1 Footprint"] * df["tileC"]
    #df["fx"] = df["tileC_cc"] - +df["k"]
    return df, cmFeatures




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

def main():
    input = sys.argv[1]  # "review-phenomizer.csv"
    output = sys.argv[2]  # an html file
    inputPadded = sys.argv[3]
    inputPaddedUntimed = sys.argv[4]
    titleOfWebpage = sys.argv[5]
    inputUntimed = sys.argv[6]
    #print(f'features of svr are {features}')
    # x_col = "Regular Loads"
    # y_col = "Kernel Time"
    title = f"{input[38:-4]}"
    # titleOfWebpage = f"{modelPickle} tested on {input}"
    # --- Step 1: Read the CSV file ---
    df = pd.read_csv(input)

    if inputUntimed != "":    # load and further annotate untimed data
        df_untimed = pd.read_csv(inputUntimed)
        df_untimed = addFeatures(df_untimed)
        df_untimed, cmFeaturesUnused = addCost(df_untimed)
        df_untimed, fxParams = addFx(df_untimed)
    
    if inputPadded != "":
        print("TODO: HANDLE PADDED DATA")
        df_padded = pd.read_csv(inputPadded)
        df_padded = addFeatures(df_padded)
        df_padded, cmFeaturesUnused = addCost(df_padded)
        df_padded, fxParams = addFx(df_padded)
    if inputPaddedUntimed != "":
        print("TODO: HANDLE PADDED, UNTIMED DATA")
       

    df_sorted = df.sort_values(by="Kernel Time", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted
    df = addFeatures(df)
    df, cmFeatures = addCost(df)
    df.to_csv("pumpkin.csv")
    df, fxParams = addFx(df)
    shapeMetric = "CC L1 Footprint"#"fmaddsPerCore"
    df = specializeMarkers(df, shapeMetric)
    # df = specializeMarkers(df,"fmaddsPerCore")

    df_sorted = df.sort_values(by="cost", ascending=True)
    df_sorted["costRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted
    # dealWTieBreakers(df)
    # print(cmFeatures)

    

   # modelPickle = "svr.pickle"
    # features=["L3 L/S Timed","fmaddsPerCore","CC L1 Footprint","Regular Loads"] # decFeatures
    # features=["SSR Configs","fmaddsPerCore","L1 Usage"] # janFeatures
    # features = [
    #     "M",
    #     "N",
    #     "K",
    #     "m",
    #     "n",
    #     "k",
    #     "SSR Config Count",
    #     "Regular Loads",
    #     "tileC_cc",
    #     # "tileA",
    #     # "L3 Loads Timed"
    # ]
    # if not os.path.exists(modelPickle):
    #     print("We are training a NEW SVR...")
    #     df.to_csv(f"svr_training_data_{sys.argv[3]}.csv")
    #     target = "Kernel Time"
    #     learnCostTest(f"svr_training_data_{sys.argv[3]}.csv",modelPickle,features,target)
        
    # df_w_prediction, coeffs = testTrained(df, modelPickle, features)

    if inputUntimed == "":    # Create interactive scatter plots
        html = rg_2_23.generateInteractiveGraphs(df, title, titleOfWebpage)
    else:
        if inputPadded != "":
            html = rg_2_23.generateInteractiveGraphsTimedAndUntimedAndPadded(df, title, titleOfWebpage, df_untimed, df_padded)
        else:
            html = rg_2_23.generateInteractiveGraphsTimedAndUntimed(df, title, titleOfWebpage, df_untimed)
    # --- Write to file ---
    with open(f"{output}.html", "w") as f:
        f.write(html)

    print(f":) Saved as {output} — open it in your browser.")


if __name__ == "__main__":
    main()
