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

def addExtras(df):
    df["Hardware Loops"] = df["M"] * df["N"] * df["K"] / (8 * df["k"])
    df["SSR Configs"] = df["SSR Config Count"]
    df["L1 Usage"] = df["Space Needed in L1"]
    df["m-n-k"] = df["JSON Name"]
    df["L1 Loads"] = df["Regular Loads"]
    # more
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
    df["CC L1 / L1"] = df["CC L1 Footprint"] / df["L1 Usage"] 
    df["L1/CC L1"] = df["L1 Usage"] / df["CC L1 Footprint"]
    df["k/n"] = df["k"] / 1.0 / df["n"]#(df["tileA_cc"] / 1.0 / df["tileC_cc"])
    df["SSRconfigsXregPerStream"] = df["SSR Config Count"]*1.0 * df["regPerStream"]
    df["L1UsageXregPerStream"] = df["L1 Usage"]*1.0 * df["regPerStream"]
    df["CCL1FootprintXregPerStream"] = df["CC L1 Footprint"]*1.0 * df["regPerStream"]
    df["k/nXregPerStream"] = df["k/n"] * df["regPerStream"]
    # even more
    df["fmaddsPerCore"] = df["m"] * df["n"] * df["k"] / 8
    df["L3 Loads Timed"] = df["L3 Loads"] - df["tileC"] - df["tileA"] - df["tileB"]
    df["L3 Stores Timed"] = df["M"] * df["N"] - df["m"] * df["n"]
    df["L3 L/S Timed"] = df["L3 Stores Timed"] + df["L3 Loads Timed"]
    df["CC L1 Footprint"] = df.apply(
        lambda y: (y["m"] * y["n"] + y["m"] * y["k"]) / 8.0 + y["k"] * y["n"], axis=1
    )
    return df
    
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