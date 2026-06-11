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
# python findNewPointsBGESmall.py ../sensitivity-analysis/remainder-vs-divisor/both/timed/192x384x384-bgeSmall-results.csv ../more-data-june-5/192x384x384wm-n-k_ss_c_rem_div_ana_pruned-results-unskipped.csv
def main():
    oldData = sys.argv[1]              
    newData = sys.argv[2]      
    oD = pd.read_csv(oldData)
    nD = pd.read_csv(newData)
    nD["timedOut"]= nD["dma"] == -1
    print(f"old data has length {len(oD)}, while new data has length {len(nD)}")
    newPoints =  nD[~nD["FakeNN JSON Name"].isin(oD["FakeNN JSON Name"])]
    print(f"newPoints has length {len(newPoints)}")
    oldPlusNewPoints = pd.concat([oD,newPoints])
    print(f"old and new points together has length {len(oldPlusNewPoints)} which is {len(newPoints)+len(oD)}")
    oldPlusNewPoints.to_csv("out/BGESmallOldPlusNewPoints.csv",index=False)
    # print(newPoints[["FakeNN JSON Name","dma"]])
    # didAllTimeout = newPoints[["timedOut"]].all()
    # print(f"Did all new points timeout? {didAllTimeout}")
    # print(newPoints[newPoints["timedOut"]==False][["FakeNN JSON Name","dma"]])  

   # print(oD.columns)

if __name__ == "__main__":
    main()
