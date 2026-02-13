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
import messy_graphs as mg
import recentGraphs_1_21 as rg

# from itertools import zip

# python svr-graph.py "review-phenomizer.csv" "Phemonizer768x384x384-svr"
# python svr-graph.py "review-cube.csv" "Cube256x256x256-svr"

# this script graphs X vs Y for the input CSV and exports an interactive version of the graph to an html file.
# it also compares Y with an SVR's predicted Y value trained on the data.
# it also looks at points with same C cc tile, regular load and SSR config counts and tries to distinguish them




def dealWTieBreakers(df):
    ccTileSize = [64, 128, 256, 512, 1024]
    loadCounts = list(set(df["Regular Loads"].values))
    groups = {}
    pairs = {}
    for lc in loadCounts:
        lc_group = df[(df["Regular Loads"] == lc)]
        groups[lc] = lc_group
        for cc in ccTileSize:
            cc_group = df[(df["Regular Loads"] == lc) & (df["tileC_cc"] == cc)]
            if not cc_group.empty:
                pairs[(lc, cc)] = cc_group  # groups[lc][(df['tileC_cc'] == cc)]
                print()
                print("\t", end="")
                print(f"C tile Size {cc} with regular load count {lc}:")
                print(
                    pairs[(lc, cc)].sort_values(by="absoluteRank", ascending=True)[
                        [
                            "m-n-k",
                            "SSR Configs",
                            "tileC_cc",
                            "L3 Loads Timed",
                            "L1 Usage",
                            "tileA",
                            "tileB",
                            "absoluteRank",
                        ]
                    ]
                )


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
    df["sumSSRsRegs"] = df["SSR Config Count"] + df["Regular Loads"]
    df["fmaddsPerCore"] = df["m"] * df["n"] * df["k"] / 8
    df["L3 Loads Timed"] = df["L3 Loads"] - df["tileC"] - df["tileA"] - df["tileB"]
    df["L3 Stores Timed"] = df["M"] * df["N"] - df["m"] * df["n"]
    df["L3 L/S Timed"] = df["L3 Stores Timed"] + df["L3 Loads Timed"]
    df["CC L1 Footprint"] = df.apply(
        lambda y: (y["m"] * y["n"] + y["m"] * y["k"]) / 8.0 + y["k"] * y["n"], axis=1
    )
    df["L1/CC L1"] = df["L1 Usage"] + df["CC L1 Footprint"]
    df["k/n"] = df["k"] / 1.0 / df["n"]#(df["tileA_cc"] / 1.0 / df["tileC_cc"])
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
    df["cost"] = df[cmFeatures].sum(axis=1)
    df["cost"] =1.0* df["SSR Config Count"]/c1 - df["k/n"]*(c2/c1)
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




# adding line of best fit for SSR configs ^^^^^^^


# train an SVR given data, output file name, features and target name.
def learnCostTest(dfName, outputModelName, feature_names, target_name):
    df = pd.read_csv(dfName)  # read in the CSV
    df = addFeatures(df)
    # lowIsGood = True
    # by = target_name
    # df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    # df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
    # df_sorted = df_sorted[:9] # top 10
    # print(f"shape is {df_sorted.shape}")
    # print(df_sorted[["JSON Name", target_name, "rank"]])
    # print("is it the conversion to numpy that is holding us up?")
    # df = df_sorted
    X = np.array(df[feature_names].astype(int))
    y = np.array(df[target_name].astype(int))

    #print(f"size of x is {X.size} and shape is {X.shape} and type is {type(X)}")
    #print(f"size of y is {y.size} and shape is {y.shape} and type is {type(y)}")

    #print(f"x[0] is {X[0]} and y[0] is {y[0]}")
    # Build the model
    svm = SVR(kernel="linear", gamma=0.5, C=1.0)  # maybe try poly or rbf?

    #print("before training")
    # Train the model
    svm.fit(X, y)
    print(svm._decision_function)
    #print("done")

    file = open(outputModelName, "wb")
    # # dump information to that file
    pickle.dump(svm, file)
    # # close the file
    file.close()
    return df


def testTrained(df, svm_name, features):
    file = open(svm_name, "rb")
    svr = pickle.load(file)
    # print(f'The weights are {svr.coef_}')
    df["Predicted Kernel Time"] = df.apply(
        lambda y: svr.predict([y[features]])[0], axis=1
    )
    df["kernelTimeDiff"] = (df["Kernel Time"] - df["Predicted Kernel Time"]) / df["Kernel Time"]
    df_sorted = df.sort_values("Predicted Kernel Time", ascending=True)
    df_sorted["predictedRank"] = range(1, int(df_sorted.shape[0] + 1))
    df_sorted["rankDiff"] = abs(df_sorted["predictedRank"] - df_sorted["absoluteRank"])
    df = df_sorted
    # print(df[["JSON Name", "Kernel Time", "Predicted Kernel Time", "diff"]])
    root = svm_name[: (len(svm_name) - len(".pickle"))]
    outputFileName = f"{root}-accuracy.csv"
    df.to_csv(outputFileName)
    return df, svr.coef_


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
    
# python svr-graph.py "review-cube.csv" "decFeatures/Cube256x256x256-svr" "256-cube-svr"
def main():
    input = sys.argv[1]  # "review-phenomizer.csv"
    output = sys.argv[2]  
    modelPickle = f'{sys.argv[3]}.pickle'
    features = get_lines_from_file(sys.argv[4])
    titleOfWebpage = sys.argv[5]
    #print(f'features of svr are {features}')
    x_col = "Regular Loads"
    y_col = "Kernel Time"
    title = f"{input[38:-4]}"
    # titleOfWebpage = f"{modelPickle} tested on {input}"
    # --- Step 1: Read the CSV file ---
    df = pd.read_csv(input)

    df_sorted = df.sort_values(by="Kernel Time", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted
    df = addFeatures(df)
    df, cmFeatures = addCost(df)
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
    if not os.path.exists(modelPickle):
        print("We are training a NEW SVR...")
        df.to_csv(f"svr_training_data_{sys.argv[3]}.csv")
        target = "Kernel Time"
        learnCostTest(f"svr_training_data_{sys.argv[3]}.csv",modelPickle,features,target)
        
    df_w_prediction, coeffs = testTrained(df, modelPickle, features)

    # Create interactive scatter plots
    html = rg.generateInteractiveGraphs(df, title, titleOfWebpage, shapeMetric,cmFeatures, coeffs, features, df_w_prediction)

    # --- Write to file ---
    with open(f"{output}.html", "w") as f:
        f.write(html)

    print(f":) Saved as {output} — open it in your browser.")


if __name__ == "__main__":
    main()
