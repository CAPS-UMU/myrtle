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

# this script compares the myrtle output before and after tile size analysis is adjusted

def sameCol(left,right,colName,colNickName):
    if (left[[colName]] == right[[colName]])[[colName]].all(axis='columns').all():
        print(f'old and new contain the same  {colNickName}')
    else:
        print(f'these csvs do NOT contain the same {colNickName}')


def main():
    left = sys.argv[1]              
    right = sys.argv[2]            
    outputName= sys.argv[3]
    l = pd.read_csv(left)
    r = pd.read_csv(right)
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
    
    lr = pd.concat([l[list(l.columns)],r[list(l.columns)]])
    # print(len(lr.columns))
    # print(lr.columns)
    lr.to_csv(outputName, index=False)
    # oR = pd.read_csv(oldRemainders)
    # nR = pd.read_csv(newRemainders)
    # # categories I do NOT expect to change:
    # noChanges = ['SSR Config Count','Space Needed in L1','L3 Stores','SSR Loads', 'FMADDs', 'MULs',
    #    'FMADDsMULs','Total SSR Loads','FMADDsPerCore']
    # print("old vs new DIVISORS")
    # sameCol(oD,nD,"FakeNN JSON Name","tiling schemes") # reality check
    # for cat in noChanges:
    #     sameCol(oD,nD,cat,cat)
    # # categories I do NOT expect to change: 
    # noChanges = ['Space Needed in L1','L3 Stores']
    # print("\nold vs new REMAINDERS")
    # sameCol(oR,nR,"FakeNN JSON Name","tiling schemes") # reality check
    # for cat in noChanges:
    #     sameCol(oR,nR,cat,cat)
    # print("\nwe do expect SOME changes in the remainder tile search space...")
    # changes=['SSR Config Count','SSR Loads', 'FMADDs', 'MULs','FMADDsMULs','Total SSR Loads','FMADDsPerCore']
    # for cat in changes:
    #     sameCol(oR,nR,cat,cat)
    

    # print(oD.columns)

if __name__ == "__main__":
    main()
