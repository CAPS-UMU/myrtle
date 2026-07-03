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

# concatCSVs.py /home/hoppip/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/128x128x128-reg-SPM-results.csv out/128x128x128wm-n-k_ss_c_rem_div_ana-results-unskipped.csv out/128-concatted.csv

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
        print(f"column count does not match! left df col count:{len(l.columns)} right df col count:{len(r.columns)}")
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
        #r = r.reindex(columns=l.columns).dropna(how='all', axis=1)  # get rid of cols in B that do not match A        
        raise Exception("Error: the to csv files contain differing number of columns")

    lr = pd.concat([l[list(l.columns)],r[list(l.columns)]],axis=0, ignore_index=True)
    print(f"Rows in concatted file: {len(lr)}")
    df_cleaned = lr.drop_duplicates(subset=["FakeNN JSON Name"])
    print(f"Rows in concatted file (after removing duplicates): {len(df_cleaned)}")
    df_cleaned.to_csv(outputName, index=False)


if __name__ == "__main__":
    main()
