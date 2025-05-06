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
    legend_pos : str = "upper right"
    legend_bb : tuple[int,int] = (0,0) #bbox_to_anchor
    custom_marker : bool = False
    get_marker : Callable[[T], mpl.markers.MarkerStyle]= field(default_factory=lambda x="o": "o")
    get_marker_label : Callable[[T], str]= field(default_factory=lambda y="no label": "no label")
    get_marker_size : Callable[[T], str]= field(default_factory=lambda y=0: mpl.rcParams['lines.markersize'] ** 2)
    
#pd.core.series.Series
#get_marker_label : Callable[[pd.core.series.Series], str]= field(default_factory=lambda x: x["JSON Name"])

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

def moreGeneralGraph(ax, g:Graph2D):
    ax.set_title(g.title)
    scatter = None
    if not g.custom_marker:
        for clr, edgeClr, data, label in g.scatterSets:
            ax.scatter(
                data[g.keys.x],
                data[g.keys.y],
                c=clr,
                edgecolors=edgeClr,
                marker="o",
                label=label,
            )
    else:
        for clr, edgeClr, data, label in g.scatterSets:
            for index, row in data.iterrows():
                # print(f"type of g is {type(g)}")
                # ms=g.get_marker_size(row)
                # print(f"type of ms is {type(ms)}")
                ax.scatter(row[g.keys.x], row[g.keys.y],c=clr,edgecolors=edgeClr,label=g.get_marker_label(row),marker=g.get_marker(row))
        ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
        ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    if g.legend:
        ax.legend(loc=g.legend_pos)

#marker_size=(mpl.rcParams['lines.markersize'] ** 2 *9)

def graphEmAll(shape: tuple, graphs):
    if shape[0]*shape[1] != len(graphs):
        raise Exception("area of shape and graph count must be equal!")
    fig = plt.figure()
    for i in range(0,len(graphs)):
        ax = fig.add_subplot(shape[0], shape[1], i+1)
        moreGeneralGraph(ax, graphs[i])
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

def graphXvsYrankedbyZ(dfs,x,x_unit, y,y_unit,z,lowIsGoodZ,dispatchNo,dispatchTitle):
    # rankings
    ranked = rankBy(dfs,(dispatchNo,1),z,lowIsGoodZ)
    
    g = Graph2D(
        keys=Keys2D(
            x=x,
            x_label=x,
            x_unit=x_unit,
            y=y,
            y_label=y,
            y_unit=y_unit,
        ),
        title=dispatchTitle,
        scatterSets=[
            ("YellowGreen", "Black", ranked, "no padding")
        ],
        legend=False,
        custom_marker = True,
        get_marker = lambda x : f'${x["rank"]}$'
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    generalGraph(ax,g)
    plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd

# d = {'job_id': [1, 2, 3, 4, 5, 6, 7, 8, 9], 
#      'hub': ['ZH1', 'ZH1', 'ZH1', 'ZH2', 'ZH2', 'ZH3', 'ZH3', 'ZH3', 'ZH3'], 
#      'alerts': [18, 35, 45, 8, 22, 34, 29, 20, 30],
#     'color': ['orange', 'orange', 'orange', 'green', 'green', 'lightblue', 'lightblue', 'lightblue', 'lightblue']}

# df=pd.DataFrame(data=d)

# ax=plt.subplot(111)
# for index, row in df.iterrows():
#     ax.text(index, row['alerts'],str(row['job_id']),
#          bbox={"boxstyle" : "circle", "color":row['color']})

# ax.set(xlim=(-1,len(df)), ylim=(df["alerts"].min()-5, df["alerts"].max()+5))
# plt.show()

 #\usepackage{igo}\whitestone{1}
       # get_marker= lambda x : "$\\usepackage{igo}\\whitestone{1}$",
        #get_marker= lambda x : "$\\textcircled{"+f'{x["rank"]}'+"}$",



def getGraphXvsYrankedbyZtopQ(dfs,x,x_unit, y,y_unit,z,lowIsGoodZ,dispatchNo,dispatchTitle,q):
    # rankings
    ranked = rankBy(dfs,(dispatchNo,1),z,lowIsGoodZ)

    # if Q is invalid, graph ALL the rows instead.
    if (q <= 0) or (q > ranked.shape[0]):
        q = ranked.shape[0]
    top = getBestXFrom(ranked,"rank",q,True)
    
    return Graph2D(
        keys=Keys2D(
            x=x,
            x_label=x,
            x_unit=x_unit,
            y=y,
            y_label=y,
            y_unit=y_unit,
        ),
        title=dispatchTitle,
        scatterSets=[
            ("YellowGreen", "Black", top, "no padding")
        ],
        legend=True,
        legend_pos="upper right",
        custom_marker = True,
        get_marker = lambda x : f'${x["rank"]}$',
        get_marker_label = lambda y : y["JSON Name"]
    )



def main():
    dfs = shortcutToData()
    titles = ["Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>","Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>","Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>"]
    graphs = []
    # for dispNo, dispTitle in zip([1,8,7], titles):
    #     graphs.append(getGraphXvsYrankedbyZtopQ(dfs,"Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle,5))
    # #graphEmAll((1,3),graphs)

    dispNo =1
    dispTitle = titles[0]
    justOne = getGraphXvsYrankedbyZtopQ(dfs,"Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle,5)
    graphEmAll((1,1),[justOne])

    # d = {'job_id': [1, 2, 3, 4, 5, 6, 7, 8, 9], 
    #  'hub': ['ZH1', 'ZH1', 'ZH1', 'ZH2', 'ZH2', 'ZH3', 'ZH3', 'ZH3', 'ZH3'], 
    #  'alerts': [18, 35, 45, 8, 22, 34, 29, 20, 30],
    # 'color': ['orange', 'orange', 'orange', 'green', 'green', 'lightblue', 'lightblue', 'lightblue', 'lightblue']}

    # df=pd.DataFrame(data=d)

    # ax=plt.subplot(111)
    # for index, row in df.iterrows():
    #     ax.text(index, row['alerts'],str(row['job_id']),
    #         bbox={"boxstyle" : "circle", "color":row['color']})

    #ax.set(xlim=(-1,len(df)), ylim=(df["alerts"].min()-5, df["alerts"].max()+5))
    plt.show()

if __name__ == "__main__":
    main()

