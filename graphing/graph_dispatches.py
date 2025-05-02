from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as ticker
from dataclasses import dataclass, field


@dataclass
class Keys2D:
    """Class for keeping track of 2D graph axes + label"""

    x: str = "xKey"
    x_unit: str = "units for x axis"
    y: str = "yKey"
    y_unit: str = "units for y axis"
    x_label: str = "x label"
    y_label: str = "y label"


@dataclass
class Graph2D:
    """Class for keeping track of 2D graph info _before_ calling scatter"""

    title: str = "title of the graph"
    keys: Keys2D = Keys2D()
    scatterSets: tuple = ()


def sideGraph(ax, title, keys: Keys2D, data):
    ax.set_title(title)
    scatter = None
    for clr, edgeClr, data, label in data:
        ax.scatter(
            data[keys.x],
            data[keys.y],
            c=clr,
            edgecolors=edgeClr,
            marker="o",
            label=label,
        )
    ax.set_ylabel(f"{keys.x_label} ({keys.x_unit})")
    ax.set_xlabel(f"{keys.y_label} ({keys.y_unit})")
    ax.legend(loc="upper right")

def thirdGraph(ax,  g: Graph2D):
    ax.set_title(g.title)
    scatter = None
    for clr, edgeClr, data, label in g.scatterSets:
        ax.scatter(
            data[g.keys.x],
            data[g.keys.y],
            c=clr,
            edgecolors=edgeClr,
            marker="o",
            label=label,
        )
    ax.set_ylabel(f"{g.keys.x_label} ({g.keys.x_unit})")
    ax.set_xlabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    ax.legend(loc="upper right")

def minimalSideBySide(left, right):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    sideGraph(ax, left.title, left.keys, left.scatterSets)
    ax = fig.add_subplot(1, 2, 2)
    sideGraph(ax, right.title, right.keys, right.scatterSets)
    plt.show()

def minimalSideBySideBySide(left, otherLeft, right):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    thirdGraph(ax, left)
    ax = fig.add_subplot(1, 3, 2)
    thirdGraph(ax, otherLeft)
    ax = fig.add_subplot(1, 3, 3)
    thirdGraph(ax, right)
    # plt.tight_layout()
    plt.show()

def shortcutToData():
    path = (
        lambda x, y: f"/home/emily/myrtle/graphing/toGraph/dispatch_{x}_case{y}_everything.csv"
    )
    dispatches = [1, 7, 8]
    cases = [1, 2]
    dfs = {}
    for d in dispatches:
        for c in cases:
            dfs[(d, c)] = pd.read_csv(path(d, c))
    return dfs

def graphL1vsTime(dfs):
    left = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title="matvec: <1x400>, <1200x400> -> <1x1200>",
        scatterSets=[
            ("YellowGreen", "Black", dfs[(1, 1)], "no padding"),
            ("Thistle", "RebeccaPurple", dfs[(1, 2)], "col dim padding"),
        ],
    )
    right = Graph2D(
        keys=left.keys,
        title="matvec: <1x400>, <600x400> -> <1x600>",
        scatterSets=[
            ("YellowGreen", "Black", dfs[(7, 1)], "no padding"),
            ("Thistle", "RebeccaPurple", dfs[(7, 2)], "col dim padding"),
        ],
    )
    otherLeft = Graph2D(
        keys=left.keys,
        title="matvec: <1x600>, <600x600> -> <1x600>",
        scatterSets=[
            ("YellowGreen", "Black", dfs[(8, 1)], "no padding"),
            ("Thistle", "RebeccaPurple", dfs[(8, 2)], "col dim padding"),
        ],
    )    
    minimalSideBySideBySide(left, otherLeft, right)

def getBest5(dfs,id,by,lowIsGood):
    print(f"dispatch {id[0]} case {id[1]} best 5 ({by})")
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[[0, 1, 2, 3, 4]]
    print(df_best_5[by])
    return df_best_5

def getBestX(dfs,id,by,x,lowIsGood, cols = [] ):
    if cols == []:
        cols = ["JSON Name",by]
    print(f"dispatch {id[0]} case {id[1]} best {x} ({by})")
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0,x)]
    print(df_best_5[cols])
    return df_best_5

def getBestXFrom(df,by,x,lowIsGood, cols = []):
    if cols == []:
        cols = ["JSON Name",by]
    print(f" best {x} ({by})")
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0,x)]
    print(df_best_5[cols])
    return df_best_5


def graphL1vsTimeTop5(dfs):
    # rankings
    x = getBestX(dfs,(1,1),"Kernel Time", 5, True)
    y = getBestX(dfs,(1,1),"Space Needed in L1",5, False)
    z = getBestX(dfs,(1,1),"Total Loads",5, True)
    q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
    print()
    x = getBestX(dfs,(7,1),"Kernel Time", 5, True)
    y = getBestX(dfs,(7,1),"Space Needed in L1",5, False)
    z = getBestX(dfs,(7,1),"Total Loads",5, True)
    q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
    print()
    x = getBestX(dfs,(8,1),"Kernel Time", 5, True)
    y = getBestX(dfs,(8,1),"Space Needed in L1",5, False)
    z = getBestX(dfs,(8,1),"Total Loads",5, True)
    q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
    return None

def fiddling(dfs,d):
    print(f"LET'S ONLY THINK ABOUT DISPATCH {d}...")
    x = getBestX(dfs,(d,1),"Kernel Time", 5, True)
    y = getBestX(dfs,(d,1),"Space Needed in L1",5, False)
    z = getBestX(dfs,(d,1),"Total Loads",5, True)
    z = getBestX(dfs,(d,1),"Regular Loads",5, True)
    print("\nfilter from out of best in space needed in L1")
    q = getBestXFrom(y,"Total Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
    print("filter from out of best in space needed in L1")
    q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
    
    print()



def main():
    dfs = shortcutToData()

    #graphL1vsTime(dfs)
    #graphL1vsTimeTop5(dfs)
    fiddling(dfs,7)
    fiddling(dfs,8)


if __name__ == "__main__":
    main()

