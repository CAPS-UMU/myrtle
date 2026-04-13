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
    oldDivisors = sys.argv[1]              
    oldRemainders = sys.argv[2]            
    newDivisors = sys.argv[3]       
    newRemainders = sys.argv[4] 
    oD = pd.read_csv(oldDivisors)
    nD = pd.read_csv(newDivisors)
    oR = pd.read_csv(oldRemainders)
    nR = pd.read_csv(newRemainders)
    # categories I do NOT expect to change:
    noChanges = ['SSR Config Count','Space Needed in L1','L3 Stores','SSR Loads', 'FMADDs', 'MULs',
       'FMADDsMULs','Total SSR Loads','FMADDsPerCore']
    print("old vs new DIVISORS")
    sameCol(oD,nD,"FakeNN JSON Name","tiling schemes") # reality check
    for cat in noChanges:
        sameCol(oD,nD,cat,cat)
    # categories I do NOT expect to change: 
    noChanges = ['Space Needed in L1','L3 Stores']
    print("\nold vs new REMAINDERS")
    sameCol(oR,nR,"FakeNN JSON Name","tiling schemes") # reality check
    for cat in noChanges:
        sameCol(oR,nR,cat,cat)
    print("\nwe do expect SOME changes in the remainder tile search space...")
    changes=['SSR Config Count','SSR Loads', 'FMADDs', 'MULs','FMADDsMULs','Total SSR Loads','FMADDsPerCore']
    for cat in changes:
        sameCol(oR,nR,cat,cat)
    

    print(oD.columns)

if __name__ == "__main__":
    main()
