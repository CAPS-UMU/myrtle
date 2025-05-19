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

@dataclass #p(x), label='Linear Fit', color='red')
class Curve:
    data:np.ndarray =[],
    func : np.polynomial.polynomial.Polynomial = field(default_factory=np.polynomial.polynomial.Polynomial)#np.poly1d([1,2])
    color : str = 'red',
    label : str = 'Linear Fit'

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
class CustomMarker:
    """Class for storing functions from row data to marker style"""

    marker : Callable[[T], mpl.markers.MarkerStyle]= lambda x="o": "o"
    label : Callable[[T], str]= lambda y="no label": "_no label"
    size : Callable[[T], int]= (lambda y=0: mpl.rcParams['lines.markersize'] ** 2)
    fill : Callable[[T], str]= lambda y="YellowGreen": "YellowGreen"
    stroke : Callable[[T], str]= lambda y="Black": "Black"
    

@dataclass
class Graph2D(Generic[T]):
    """Class for keeping track of 2D graph info _before_ calling scatter"""

    title: str = "title of the graph"
    keys: Keys2D = field(default_factory=Keys2D)
    scatterSets: tuple = field(default_factory=tuple)
    legend : bool = True
    legend_pos : str = "upper right"
    legend_bb : tuple[int,int] = (1,1) #bbox_to_anchor
    custom_marker : bool = False
    curves = []
    get_marker : Callable[[T], mpl.markers.MarkerStyle]= field(default_factory=lambda x="o": "o")
    get_marker_label : Callable[[T], str]= lambda y="no label": "no label"#field(default_factory=lambda y="no label": "no label")
    get_marker_size : Callable[[T], int]= (lambda y=0: mpl.rcParams['lines.markersize'] ** 2)
    
    
def generalGraph(ax, g:Graph2D):
    ax.set_title(g.title)
    for data, cm in g.scatterSets:
        for index, row in data.iterrows():
            ax.scatter(row[g.keys.x], row[g.keys.y],c=cm.fill(row),edgecolors=cm.stroke(row),s = cm.size(row),label=cm.label(row),marker=cm.marker(row))
    ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
    ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    if len(g.curves):
        labels = []
        lines = []
        for curve in g.curves:
            line = ax.plot(curve.data, curve.func(curve.data), label=curve.label, color=curve.color, linestyle='-', linewidth=2.0)
        #     labels.append(curve.label)
        #     lines.append(line)
        # ax.legend(lines,labels)
    if g.legend:
        ax.legend(loc=g.legend_pos, bbox_to_anchor=g.legend_bb)

def graphEmAll(shape: tuple, graphs):
    if shape[0]*shape[1] != len(graphs):
        raise Exception("area of shape and graph count must be equal!")
    fig = plt.figure()
    for i in range(0,len(graphs)):
        ax = fig.add_subplot(shape[0], shape[1], i+1)
        #moreGeneralGraph(ax, graphs[i])
        generalGraph(ax, graphs[i])
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

def getBestX(dfs,id,by,x,lowIsGood):
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0,x)]
    return df_best_5

def rankBy(dfs,id,by,lowIsGood):
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_sorted["rank"] = range(1,df_sorted.shape[0]+1)
    return df_sorted

def getBestXFrom(df,by,x,lowIsGood):
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0,x)]
    return df_best_5

def getGraphXvsYrankedbyZ(dfs,x,x_unit, y,y_unit,z,lowIsGoodZ,dispatchNo,dispatchTitle):
    # rankings
    ranked = rankBy(dfs,(dispatchNo,1),z,lowIsGoodZ)
    cm = CustomMarker(
        marker = lambda x : f'${x["rank"]}$'
    )
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
            (ranked, cm)
        ],
        legend=False
    )   

def getGraphXvsYrankedbyZtopQ(dfs,x,x_unit, y,y_unit,z,lowIsGoodZ,dispatchNo,dispatchTitle,q):
    # rankings
    ranked = rankBy(dfs,(dispatchNo,1),z,lowIsGoodZ)

    # if Q is invalid, graph ALL the rows instead.
    if (q <= 0) or (q > ranked.shape[0]):
        q = ranked.shape[0]
    top = getBestXFrom(ranked,"rank",q,True)
    cm = CustomMarker(
        marker = lambda x : f'${x["rank"]}$',
        label = lambda y : y["JSON Name"]
    )
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
            (top, cm)
        ],
        legend=True, 
        legend_pos="upper right"
    )




def main():
    dfs = shortcutToData()
    titles = ["Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>","Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>","Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>"]
    graphs = []
    for dispNo, dispTitle in zip([1,8,7], titles):
        graphs.append(getGraphXvsYrankedbyZtopQ(dfs,"Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle,5))
    graphEmAll((1,3),graphs)

    dispNo =1
    dispTitle = titles[0]
    justOne = getGraphXvsYrankedbyZ(dfs,"Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle)
    
    graphEmAll((1,1),[justOne])

    

if __name__ == "__main__":
    main()

