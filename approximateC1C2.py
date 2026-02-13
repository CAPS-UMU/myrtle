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
from itertools import combinations

def addFeatures(df):
    df["Hardware Loops"] = df["M"] * df["N"] * df["K"] / (8 * df["k"])
    df["SSR Configs"] = df["SSR Config Count"]
    df["L1 Usage"] = df["Space Needed in L1"]
    df["m-n-k"] = df["JSON Name"]
    df["L1 Loads"] = df["Regular Loads"]
    return df

def extractVars(p):
    return int(p["Kernel Time"].iloc[0]), int(p["M"].iloc[0]),int(p["N"].iloc[0]),int(p["K"].iloc[0]),int(p["m"].iloc[0]),int(p["n"].iloc[0]),int(p["k"].iloc[0])

def calc_c1_c2(p1,p2):
    t1, M, N, K, m1, n1, k1 = extractVars(p1)
    t2, M, N, K, m2, n2, k2 = extractVars(p2)
    c2 = float(8*M*N*K*(t2*m2*n2-t1*m1*n1)) / float(t2*m1*m2*n2*k1 - t1*m1*n1*m2*k2)
    c1 = float(8*M*N*K-m1*k1)/float(m1*n1*t1) * c2
    return c1, c2, t1, t2

# python approximateC1C2.py "$FOLDER/review-cube.csv" "c1-c2.txt"
def main():
    input = sys.argv[1]  # "$FOLDER/review-cube.csv"
    output = sys.argv[2]  # "c1-c2.txt"

    # --- Step 1: Read the CSV file ---
    df = pd.read_csv(input)
    # add helpful columns
    df_sorted = df.sort_values(by="Kernel Time", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted
    df = addFeatures(df)
   

    # print(df["m-n-k"])
    # print(df["m-n-k"][0])
    # print(df[df["m-n-k"]=="32-32-32"])
    cols = ["pair","c1","c2","Time Diff","fst","snd","P1 SSR Configs","P2 SSR Configs","Time P1","Time P2", "P1 Rank", "P2 Rank"]
    rows = []
    for c in combinations(df["m-n-k"].to_list(),r=2):
         p1 = df[df["m-n-k"]==c[0]]
         #print(int(p1["Kernel Time"]))
         p2 = df[df["m-n-k"]==c[1]]
         c1, c2, t1, t2 = calc_c1_c2(p1,p2)
         p1_ssr_configs = int(p1["SSR Config Count"].iloc[0]) 
         p2_ssr_configs = int(p2["SSR Config Count"].iloc[0])
         p1_rank = int(p1["absoluteRank"].iloc[0])
         p2_rank = int(p2["absoluteRank"].iloc[0])
         timeDiff = abs(t1-t2)
        # print(f'{c},{c1}, {c2}')
         rows.append((f'{c}',c1,c2,timeDiff,f'{c[0]}',f'{c[1]}',p1_ssr_configs,p2_ssr_configs,t1,t2,p1_rank,p2_rank))

    approx = pd.DataFrame(rows, columns=cols)
    #print(approx)
    # print(len(approx))
    c1_avg = sum(approx["c1"].to_list())/len(approx)
    c2_avg = sum(approx["c2"].to_list())/len(approx)
    avg_time_diff = sum(approx["Time Diff"].to_list())/len(approx)
    avg_p1_configs = sum(approx["P1 SSR Configs"].to_list())/len(approx)
    avg_p2_configs = sum(approx["P2 SSR Configs"].to_list())/len(approx)
    avg_p1_time = avg_p2_configs = sum(approx["Time P1"].to_list())/len(approx)
    avg_p2_time = avg_p2_configs = sum(approx["Time P2"].to_list())/len(approx)
    approx.loc[len(approx)] = ["average", c1_avg, c2_avg, avg_time_diff,"avg P1","avg P2",avg_p1_configs,avg_p2_configs,avg_p1_time,avg_p2_time,10,10]  # adding a row
    #print(approx)
    approx.to_csv(output)

    titleOfWebpage = "c1 and c2 values from the 256x256x256 Matmul"

    html = rg.generateInteractiveC1C2Graph(approx, titleOfWebpage)

    # --- Write to file ---
    with open("c1-c2-approx.html", "w") as f:
        f.write(html)

    print(":) Saved as c1-c2-approx — open it in your browser.")
if __name__ == "__main__":
    main()
