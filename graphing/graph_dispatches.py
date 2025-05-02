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
from typing import Callable
from typing import TypeVar, Generic

T = TypeVar('T')

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
class Graph2D(Generic[T]):
    """Class for keeping track of 2D graph info _before_ calling scatter"""

    title: str = "title of the graph"
    keys: Keys2D = field(default_factory=Keys2D)
    scatterSets: tuple = field(default_factory=tuple)
    legend : bool = True
    custom_marker : bool = False
    get_marker : Callable[[T], mpl.markers]= field(default_factory=lambda x="o": "o")
    


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
    ax.set_xlabel(f"{keys.x_label} ({keys.x_unit})")
    ax.set_ylabel(f"{keys.y_label} ({keys.y_unit})")
    ax.legend(loc="upper right")

#  for index, row in best_1.iterrows():
#         print(row['JSON Name'], row['Kernel Time'])
#     # marker=f'${txt}$'
def generalGraph(ax, g:Graph2D):
    if not g.custom_marker:
        thirdGraph(ax,g)
    else:
        ax.set_title(g.title)
        scatter = None
        for clr, edgeClr, data, label in g.scatterSets:
            for index, row in data.iterrows():
                ax.scatter(row[g.keys.x], row[g.keys.y],c=clr,edgecolors=edgeClr,label=label,marker=g.get_marker(row))
        ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
        ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")

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
    ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
    ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    if g.legend:
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
    generalGraph(ax, left)
    ax = fig.add_subplot(1, 3, 2)
    generalGraph(ax, otherLeft)
    ax = fig.add_subplot(1, 3, 3)
    generalGraph(ax, right)
    # plt.tight_layout()
    plt.show()

def shortcutToData():
    path = (
        lambda x, y: f"./toGraph/dispatch_{x}_case{y}_everything.csv"
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
        title="Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        scatterSets=[
            ("YellowGreen", "Black", dfs[(1, 1)], "no padding"),
            ("Thistle", "RebeccaPurple", dfs[(1, 2)], "col dim padding"),
        ],
    )
    right = Graph2D(
        keys=left.keys,
        title="Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
        scatterSets=[
            ("YellowGreen", "Black", dfs[(7, 1)], "no padding"),
            ("Thistle", "RebeccaPurple", dfs[(7, 2)], "col dim padding"),
        ],
    )
    otherLeft = Graph2D(
        keys=left.keys,
        title="Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
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

def rankBy(dfs,id,by,lowIsGood, cols = [] ):
    if cols == []:
        cols = ["JSON Name",by,"rank"]
    print(f"dispatch {id[0]} case {id[1]} ranked by {by}")
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_sorted["rank"] = range(1,df_sorted.shape[0]+1)
    print(df_sorted[cols])
    return df_sorted

def getBestXFrom(df,by,x,lowIsGood, cols = []):
    if cols == []:
        cols = ["JSON Name",by]
    print(f" best {x} ({by})")
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0,x)]
    print(df_best_5[cols])
    return df_best_5

def graphL1vsTimeTopX(dfs,x, numMarkers = False):
    # rankings
    best_1 = getBestX(dfs,(1,1),"Kernel Time", x, True)
    best_7 = getBestX(dfs,(7,1),"Kernel Time", x, True)
    best_8 = getBestX(dfs,(8,1),"Kernel Time", x, True)
    if numMarkers:
        best_1["rank"] = range(1,x+1)
        best_7["rank"] = range(1,x+1)
        best_8["rank"] = range(1,x+1)
        pick_marker = lambda x : f'${x["rank"]}$'
    else:
        pick_marker = lambda x : "o"

    left = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title="Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        scatterSets=[
            ("YellowGreen", "Black", best_1, "no padding")
        ],
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )

    middle = Graph2D(
        keys=left.keys,
        title="Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        scatterSets=[
            ("YellowGreen", "Black", best_8, "no padding")
        ],
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )

    right = Graph2D(
        keys=left.keys,
        title="Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
        scatterSets=[
            ("YellowGreen", "Black", best_7, "no padding")
        ],
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )
    
    minimalSideBySideBySide(left, middle, right)


# def graphL1vsTimeTop5(dfs):
#     # rankings
#     x = getBestX(dfs,(1,1),"Kernel Time", 5, True)
#     y = getBestX(dfs,(1,1),"Space Needed in L1",5, False)
#     z = getBestX(dfs,(1,1),"Total Loads",5, True)
#     q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
#     print()
#     x = getBestX(dfs,(7,1),"Kernel Time", 5, True)
#     y = getBestX(dfs,(7,1),"Space Needed in L1",5, False)
#     z = getBestX(dfs,(7,1),"Total Loads",5, True)
#     q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
#     print()
#     x = getBestX(dfs,(8,1),"Kernel Time", 5, True)
#     y = getBestX(dfs,(8,1),"Space Needed in L1",5, False)
#     z = getBestX(dfs,(8,1),"Total Loads",5, True)
#     q = getBestXFrom(y,"Regular Loads",4, True,["JSON Name","Total Loads","Regular Loads","Streaming Loads", "Outer Loop Iters"])
#     return None

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

def investigateLoadsAgainstTopX(dfs,x, numMarkers, dispatchNo, dispatchTitle):
    # hardcoding dispatch 7 for now!
    # rankings
    best_x = getBestX(dfs,(dispatchNo,1),"Kernel Time", x, True)
    if numMarkers:
        best_x["rank"] = range(1,x+1)
        pick_marker = lambda x : f'${x["rank"]}$'
    else:
        pick_marker = lambda x : "o"
    #df["Tile Shape"] = df.apply(lambda x: get_tile_shape(x["Row Dim"], x["Reduction Dim"]), axis=1)
    best_x["Regular / Total Loads"]=best_x.apply(lambda x: (x["Regular Loads"] / x["Total Loads"]) * 100.0, axis=1)
    print(best_x[["JSON Name","Regular Loads","Total Loads","Regular / Total Loads"]])
    left = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Total Loads",
            y_label="Total Loads",
            y_unit="loads",
        ),
        title=dispatchTitle,
        scatterSets=[
            ("YellowGreen", "Black", best_x, "no padding")
        ],
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )

    middle = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Regular Loads",
            y_label="Regular Loads",
            y_unit="loads",
        ),
        title=left.title,
        scatterSets=left.scatterSets,
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )

    right = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Regular / Total Loads",
            y_label="Regular / Total Loads",
            y_unit="percentage",
        ),
        title=left.title,
        scatterSets=left.scatterSets,
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )
    
    minimalSideBySideBySide(left, middle, right)

def investigateLoadsAgainstTopXofBestYRankedByZ(dfs,x, y,lowIsGoodY, z, lowIsGoodZ, numMarkers, dispatchNo, dispatchTitle):
    # hardcoding dispatch 7 for now!
    # take the ranking FIRST
    ranked = rankBy(dfs,(dispatchNo,1),z,lowIsGoodZ)
    #def rankBy(dfs,id,by,lowIsGood, cols = [] ):
    #"Kernel Time"
    # then take subset of x elements
    #getBestXFrom(df,by,x,lowIsGood, cols = [])
    best_x = getBestXFrom(ranked,y, x, lowIsGoodY)

    if numMarkers:
        pick_marker = lambda x : f'${x["rank"]}$'
    else:
        pick_marker = lambda x : "o"
    #df["Tile Shape"] = df.apply(lambda x: get_tile_shape(x["Row Dim"], x["Reduction Dim"]), axis=1)
    best_x["Regular / Total Loads"]=best_x.apply(lambda x: (x["Regular Loads"] / x["Total Loads"]) * 100.0, axis=1)
    print(best_x[["JSON Name","Regular Loads","Total Loads","Regular / Total Loads"]])
    left = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Total Loads",
            y_label="Total Loads",
            y_unit="loads",
        ),
        title=dispatchTitle,
        scatterSets=[
            ("YellowGreen", "Black", best_x, "no padding")
        ],
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )

    middle = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Regular Loads",
            y_label="Regular Loads",
            y_unit="loads",
        ),
        title=left.title,
        scatterSets=left.scatterSets,
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )

    right = Graph2D(
        keys=Keys2D(
            x="Space Needed in L1",
            x_label="Space Needed in L1",
            x_unit="bytes",
            y="Regular / Total Loads",
            y_label="Regular / Total Loads",
            y_unit="percentage",
        ),
        title=left.title,
        scatterSets=left.scatterSets,
        legend=False,
        custom_marker = numMarkers,
        get_marker = pick_marker
    )
    
    minimalSideBySideBySide(left, middle, right)
    # return(left, middle, right)

def threeByThree(dfs):
    fig = plt.figure()
    r11, r12, r13 =investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,1,"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>")
    r21, r22, r23 =investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,8,"Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>")
    r31, r32, r33 =investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,7,"Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>")
    
    ax = fig.add_subplot(3, 3, 1)
    generalGraph(ax, r11)
    ax = fig.add_subplot(3, 3, 2)
    generalGraph(ax, r12)
    ax = fig.add_subplot(3, 3, 3)
    generalGraph(ax, r13)
    ax = fig.add_subplot(3, 3, 4)
    generalGraph(ax, r21)
    ax = fig.add_subplot(3, 3, 5)
    generalGraph(ax, r22)
    ax = fig.add_subplot(3, 3, 6)
    generalGraph(ax, r23)
    ax = fig.add_subplot(3, 3, 7)
    generalGraph(ax, r31)
    ax = fig.add_subplot(3, 3, 8)
    generalGraph(ax, r32)
    ax = fig.add_subplot(3, 3, 9)
    generalGraph(ax, r33)
    # ax = fig.add_subplot(1, 3, 2)
    # generalGraph(ax, ro)
    # ax = fig.add_subplot(1, 3, 3)
    # generalGraph(ax, right)
    plt.tight_layout()
    plt.show()
    return None


def main():
    dfs = shortcutToData()

    # graphL1vsTime(dfs)
    # graphL1vsTimeTop5(dfs)
    # fiddling(dfs,7)
    # fiddling(dfs,8)
    # graphL1vsTimeTopX(dfs,5,False)
    # graphL1vsTimeTopX(dfs,5,True)
    # graphL1vsTimeTopX(dfs,9,True)
    # investigateLoadsAgainstTopX(dfs,9,True,7,"Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>")
    # investigateLoadsAgainstTopX(dfs,9,True,8,"Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>")
    # investigateLoadsAgainstTopX(dfs,9,True,1,"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>")
    #rankBy(dfs,(1,1),"Kernel Time", True)
    #investigateLoadsAgainstTopXofBestYRankedByZ(dfs,x, y,lowIsGoodY, z, lowIsGoodZ, numMarkers, dispatchNo, dispatchTitle):
    #investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,1,"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>")
    # minimalSideBySideBySide(l, m, r)
    #threeByThree(dfs)
    investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,1,"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>")
    investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,8,"Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>")
    investigateLoadsAgainstTopXofBestYRankedByZ(dfs,10,"Space Needed in L1",False,"Kernel Time",True,True,7,"Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>")
    


if __name__ == "__main__":
    main()

