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

T = TypeVar("T")

@dataclass
class Feature:
    "Class for keeping track of kinds of data to graph"
    cat: str = "csvColName"
    units: str = ""
    lowIsGood : bool = True
    label: str = "label"

@dataclass
class Keys2D:
    """Class for keeping track of 2D graph axes + label"""

    x: str = "xKey"
    x_unit: str = "units for x axis"
    y: str = "yKey"
    y_unit: str = "units for y axis"
    x_label: str = "x label"
    y_label: str = "y label"


# clr, edgeClr, data, label
# ("Thistle", "RebeccaPurple", dfs[(1, 2)], "col dim padding"),
@dataclass
class ScatterSet(Generic[T]):
    """Class for keeping track of a set of points to plot in a scatter plot"""

    custom_marker: bool = False
    fillClr: str = "Thistle"
    strokeClr: str = "RebeccaPurple"
    data: pd.core.series.Series = None
    marker: Callable[[T], mpl.markers.MarkerStyle] = lambda x=None: "o"
    marker_size: Callable[[T], float] = (
        lambda y=None: mpl.rcParams["lines.markersize"] ** 2
    )
    marker_label: Callable[[T], str] = lambda y=None: None


@dataclass
class Graph2D(Generic[T]):
    """Class for keeping track of 2D graph info _before_ calling scatter"""

    title: str = "title of the graph"
    keys: Keys2D = field(default_factory=Keys2D)
    scatters: list = field(default_factory=tuple)
    legend: bool = True
    legend_pos: str = "upper right"
    legend_bb: tuple[int, int] = (0, 0)  # bbox_to_anchor

maxL1 = Feature("Space Needed in L1", units="Bytes", lowIsGood=False,label="Space Needed in L1")
runtime= Feature('Kernel Time', "Cycles", lowIsGood=True,label='Kernel Time')
totalLoads= Feature('Total Loads', "Loads", lowIsGood=True,label='Total Loads')
regularLoads= Feature('Regular Loads', "Loads", lowIsGood=True,label='Regular Loads')
streamingLoads= Feature('Streaming Loads', "Loads", lowIsGood=False,label='Streaming Loads')
tileSize=Feature('Tile Size', "Elements", lowIsGood=False,label='Tile Size')
wideness=Feature("Wideness","Row Dim/Col Dim", lowIsGood=False,label="Wideness")
# pd.core.series.Series
# get_marker_label : Callable[[pd.core.series.Series], str]= field(default_factory=lambda x: x["JSON Name"])


#  for index, row in best_1.iterrows():
#         print(row['JSON Name'], row['Kernel Time'])
#     # marker=f'${txt}$'
# def generalGraph(ax, g: Graph2D):
#     if not g.custom_marker:
#         thirdGraph(ax, g)
#     else:
#         ax.set_title(g.title)
#         scatter = None
#         for clr, edgeClr, data, label in g.scatterSets:
#             for index, row in data.iterrows():
#                 ax.scatter(
#                     row[g.keys.x],
#                     row[g.keys.y],
#                     c=clr,
#                     edgecolors=edgeClr,
#                     label=label,
#                     marker=g.get_marker(row),
#                 )
#         ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
#         ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")


def generalGraph(ax, g: Graph2D):
    ax.set_title(g.title)
    scatter = None
    for s in g.scatters:
        if not s.custom_marker:
            ax.scatter(
                s.data[g.keys.x],
                s.data[g.keys.y],
                c=s.fillClr,
                edgecolors=s.strokeClr,
                label=s.marker_label(),
                marker=s.marker(),
                s=s.marker_size(),
            )
        else:
            for index, row in s.data.iterrows():
                ax.scatter(
                    row[g.keys.x],
                    row[g.keys.y],
                    c=s.fillClr,
                    edgecolors=s.strokeClr,
                    label=s.marker_label(row),
                    marker=s.marker(row),
                    s=s.marker_size(row),
                )
    ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
    ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    if g.legend:
        ax.legend(loc=g.legend_pos)


# marker_size=(mpl.rcParams['lines.markersize'] ** 2 *9)


def graphEmAll(shape: tuple, graphs):
    if shape[0] * shape[1] != len(graphs):
        raise Exception("area of shape and graph count must be equal!")
    fig = plt.figure()
    for i in range(0, len(graphs)):
        ax = fig.add_subplot(shape[0], shape[1], i + 1)
        generalGraph(ax, graphs[i])
    plt.show()


def shortcutToData():
    path = lambda x, y: f"./toGraph/dispatch_{x}_case{y}_everything.csv"
    dispatches = [1, 7, 8]
    cases = [1, 2]
    dfs = {}
    for d in dispatches:
        for c in cases:
            g=pd.read_csv(path(d, c))
            g["Wideness"] = g["Row Dim"] / g["Reduction Dim"]
            dfs[(d, c)] = g
    return dfs


def getBestX(dfs, id, by, x, lowIsGood, cols=[]):
    if cols == []:
        cols = ["JSON Name", by]
    print(f"dispatch {id[0]} case {id[1]} best {x} ({by})")
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0, x)]
    print(df_best_5[cols])
    return df_best_5


def rankBy(df, by, lowIsGood):
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    df_sorted["rank_" + by] = range(1, df_sorted.shape[0] + 1)
    return df_sorted


def getBestXFrom(df, by, x, lowIsGood, cols=[]):
    if cols == []:
        cols = ["JSON Name", by]
    print(f" best {x} ({by})")
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0, x)]
    print(df_best_5[cols])
    return df_best_5


def graphXvsYrankedbyZ(
    dfs, x, x_unit, y, y_unit, z, lowIsGoodZ, dispatchNo, dispatchTitle
):
    # rankings
    ranked = rankBy(dfs[(dispatchNo, 1)], z, lowIsGoodZ)

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
        scatterSets=[("YellowGreen", "Black", ranked, "no padding")],
        legend=False,
        custom_marker=True,
        get_marker=lambda x: f'${x["rank_"+z]}$',
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    generalGraph(ax, g)
    plt.show()

def graphAvsB(
    df, dispatchTitle, a : Feature, b :Feature
):
    return Graph2D(
        keys=Keys2D(
            x=a.cat,
            x_label=a.label,
            x_unit=a.units,
            y=b.cat,
            y_label=b.label,
            y_unit=b.units,
        ),
        title=dispatchTitle,
        scatters=[ScatterSet(
                custom_marker=False,
                fillClr="YellowGreen",
                strokeClr="Black",
                data=df,
                marker_label=lambda y="no padding": y,
                marker=lambda x="o": x,
            ),],
        legend=True,
    )

    


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

# \usepackage{igo}\whitestone{1}
# get_marker= lambda x : "$\\usepackage{igo}\\whitestone{1}$",
# get_marker= lambda x : "$\\textcircled{"+f'{x["rank"]}'+"}$",
def graphABCDtopXD(a,b,c,d,x):
    return 5



def getGraphXvsYrankedbyZtopQ(
    dfs, x, x_unit, y, y_unit, z, lowIsGoodZ, dispatchNo, dispatchTitle, q
):
    # rankings
    ranked = rankBy(dfs[(dispatchNo, 1)], z, lowIsGoodZ)
    ranked = rankBy(ranked, "Space Needed in L1", False)
    # print(ranked[["JSON Name", "rank_" + z, "rank_Total Loads", "rank_Space Needed in L1"]])
    # print(ranked[["JSON Name", "rank_" + z, "rank_Space Needed in L1"]])

    # if Q is invalid, graph ALL the rows instead.
    if (q <= 0) or (q > ranked.shape[0]):
        q = ranked.shape[0]
    #top = getBestXFrom(ranked, "rank_" + z, q, True)
    # ranked = rankBy(top, "Space Needed in L1", False)
    #print(top[["JSON Name", "rank_" + z, "rank_Space Needed in L1","Space Needed in L1"]])
    top = getBestXFrom(ranked, "Space Needed in L1", q, False)
    ranked = rankBy(top, "Space Needed in L1", False)
    top= ranked.sort_values(by=z, ascending=lowIsGoodZ)

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
        scatters=[
            # ScatterSet(
            #     custom_marker=True,
            #     fillClr="YellowGreen",
            #     strokeClr="Black",
            #     data=top,
            #     marker_label=lambda y: None,
            #     marker=lambda x: "o",
            #     marker_size = lambda x: (mpl.rcParams["lines.markersize"] ** 2)*4*(5/x["rank_Space Needed in L1"])
            #     #lambda x: (mpl.rcParams["lines.markersize"] ** 2)*8*(5/x["rank_Space Needed in L1"])
            # ),
            ScatterSet(
                custom_marker=True,
                fillClr="YellowGreen",
                strokeClr="Black",
                data=top,
                marker_label=lambda y: None,
                marker=lambda x: "o",
                marker_size = lambda x: (mpl.rcParams["lines.markersize"] ** 2)*4*(5/x["rank_Space Needed in L1"])
                #lambda x: (mpl.rcParams["lines.markersize"] ** 2)*8*(5/x["rank_Space Needed in L1"])
            ),
            # ScatterSet(
            #     custom_marker=True,
            #     fillClr="YellowGreen",
            #     strokeClr="YellowGreen",
            #     data=top,
            #     marker_label=lambda y: None,
            #     marker=lambda x: "o",
            #     marker_size = lambda x: (mpl.rcParams["lines.markersize"] ** 2)*4*x["rank_Total Loads"]
            # ),
            ScatterSet(
                custom_marker=True,
                fillClr="YellowGreen",
                strokeClr="Black",
                data=top,
                marker_label=lambda y: y["JSON Name"],
                marker=lambda x: f'${x["rank_"+z]}$',
            ),
        ],
        legend=True,
        legend_pos="upper center",
        legend_bb=(0.5,-0.05)
    )

    


def main():
    dfs = shortcutToData()
    dispatchNos = [1, 8, 7]
    titles = [
        "Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        "Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        "Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
    ]
    graphs = []
    for dispNo, dispTitle in zip(dispatchNos, titles):
        graphs.append(
            getGraphXvsYrankedbyZtopQ(
                dfs,
                "Streaming Loads",
                "loads",
                "Regular Loads",
                "loads",
                "Kernel Time",
                True,
                dispNo,
                dispTitle,
                5,
            )
        )
    #wideness=Feature("Wideness","Row Dim/Col Dim", lowIsGood=False,label="Wideness")
    # add ration to data
    # for g in dfs:
    #     g["Wideness"] = df["Row Dim"] + df["Col Dim"]
    #     firstCSV["tuples"]=firstCSV.apply(lambda x: tupleIze(x["Row Dim"], x["Reduction Dim"]), axis=1)
    # tups = firstCSV["tuples"].tolist()

    # for dispNo, dispTitle in zip(dispatchNos, titles):
    #     graphs.append(
    #         getGraphXvsYrankedbyZtopQ(
    #             dfs,
    #             "Total Loads",
    #             "loads",
    #             "Wideness",
    #             "Row Dim / Col Dim",
    #             "Kernel Time",
    #             True,
    #             dispNo,
    #             dispTitle,
    #             5,
    #         )
    #     )
    # for dispNo, dispTitle in zip(dispatchNos, titles):
    #     graphs.append(
    #         graphAvsB(
    #             dfs[(dispNo,1)],
    #             dispTitle,
    #             #maxL1,
    #             totalLoads,
    #             runtime,
    #         )
    #     )
    graphEmAll((1, 3), graphs)
    # lambda y=None: mpl.rcParams['lines.markersize'] ** 2


if __name__ == "__main__":
    main()
