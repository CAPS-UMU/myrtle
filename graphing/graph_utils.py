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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

T = TypeVar("T")


@dataclass  # p(x), label='Linear Fit', color='red')
class Curve:
    data: np.ndarray = ([],)
    func: np.polynomial.polynomial.Polynomial = field(
        default_factory=np.polynomial.polynomial.Polynomial
    )  # np.poly1d([1,2])
    color: str = ("red",)
    label: str = "Linear Fit"
    id: int = -1


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
    y : str = '',
    marker: Callable[[T], mpl.markers.MarkerStyle] = lambda x="o": "o"
    label: Callable[[T], str] = lambda y="no label": "_no label"
    size: Callable[[T], int] = lambda y=0: mpl.rcParams["lines.markersize"] ** 2
    fill: Callable[[T], str] = lambda y="YellowGreen": "YellowGreen"
    stroke: Callable[[T], str] = lambda y="Black": "Black"


@dataclass
class Graph2D(Generic[T]):
    """Class for keeping track of 2D graph info _before_ calling scatter"""
    imagePath : str =""
    title: str = "title of the graph"
    keys: Keys2D = field(default_factory=Keys2D)
    scatterSets: tuple = field(default_factory=tuple)
    legend: bool = True
    legend_pos: str = "upper right"
    legend_bb: tuple[int, int] = (1, 1)  # bbox_to_anchor
    legend_title : str = ""
    custom_marker: bool = False
    table: bool = False
    table_pos : str = "upper right"
    table_bb : tuple[int, int] = (1, 1)
    table_col_widths : list[float] = field(default_factory=list)
    table_col_labels : list[str] = field(default_factory=list)
    table_row_labels : list[str] = field(default_factory=list)
    table_data : list[list[str]] = field(default_factory=list)
    curves = []
    get_marker: Callable[[T], mpl.markers.MarkerStyle] = field(
        default_factory=lambda x="o": "o"
    )
    get_marker_label: Callable[[T], str] = (
        lambda y="no label": "no label"
    )  # field(default_factory=lambda y="no label": "no label")
    get_marker_size: Callable[[T], int] = (
        lambda y=0: mpl.rcParams["lines.markersize"] ** 2
    )


def generalGraph(ax, g: Graph2D):
    ax.set_title(g.title)
    for data, cm in g.scatterSets:
        for index, row in data.iterrows():
            ax.scatter(
                row[g.keys.x],
                row[cm.y],#row[g.keys.y],#row[cm.y],
                c=cm.fill(row),
                edgecolors=cm.stroke(row),
                s=cm.size(row),
                label=cm.label(row),
                marker=cm.marker(row),
            )
    ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
    ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    if len(g.curves):
        labels = []
        lines = []
        for curve in g.curves:
            line = ax.plot(
                curve.data,
                curve.func(curve.data),
                label=curve.label,
                color=curve.color,
                linestyle="-",
                linewidth=2.0,
            )
        #     labels.append(curve.label)
        #     lines.append(line)
        # ax.legend(lines,labels)
    if g.legend:
        leg = ax.legend(loc=g.legend_pos, bbox_to_anchor=g.legend_bb,title=g.legend_title)
        #leg.set_title("")  
        leg._legend_box.align = "left"
        blue_line = mlines.Line2D([], [], color='blue', marker=f'${"0"}$',
                          markersize=15, label=g.legend_title)
        #red_patch = mpatches.Patch(color='red', label='The red data')
        #ax.legend(handles=[red_patch])
      #  h, l = ax.get_legend_handles_labels()
      #  ax.legend(handles=[blue_line] + h)
        
    if g.table:#bbox, loc,rowLabels,colLabels, cellText#colWidths
        #xmin, ymin, width, height bbox=(1,0,1,1),
        #columnWidths = [1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
        t=plt.table(loc='right',bbox=g.table_bb,colLabels=g.table_col_labels,colWidths=g.table_col_widths,cellText=g.table_data.values.tolist())
       # t=plt.table(loc='right',bbox=g.table_bb,colLabels=g.table_col_labels,cellText=g.table_data.values.tolist())
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        #the_table.scale(2, 2)
        ax.add_table(t)
        #.values.tolist()
        #ax.table(bbox=g.table_bb,loc=g.table_pos,rowLabels=g.table_row_labels,colLabels=g.table_col_labels,cellText=g.table_data)


def graphEmAll(shape: tuple, graphs):
    if shape[0] * shape[1] != len(graphs):
        raise Exception("area of shape and graph count must be equal!")
    fig = plt.figure()
    #fig.set_size_inches(4, 2)
    #fig.set_size_inches(1098/72.0,476/72.0) # 15.25 x 6.61
    fig.set_size_inches(8,6.5)
    for i in range(0, len(graphs)):
        ax = fig.add_subplot(shape[0], shape[1], i + 1)
        # moreGeneralGraph(ax, graphs[i])
        generalGraph(ax, graphs[i])
        plt.savefig(f"{graphs[i].imagePath}", bbox_inches='tight')
    #fig = plt.gcf()  # Get the current figure
    #Figure.dpi
    #1187 × 265
    # 1068x476 became 2477 x 615
    #fig.set_dpi(72)
    # fig.set_size_inches(1187/72.0,265/72.0)
    # print(type(plt.Figure.dpi))
    #fig.set_size_inches(4, 2)
    # plt.savefig(f"{g.image}.png", bbox_inches='tight')
    # plt.show()


    # label: Callable[[T], str] = lambda y="no label": "_no label"


def shortcutToData(dir="./toGraph"):
    path = lambda x, y: f"{dir}/dispatch_{x}_case{y}_everything.csv"
    dispatches = [1, 7, 8]
    cases = [1, 2]
    dfs = {}
    for d in dispatches:
        for c in cases:
            dfs[(d, c)] = pd.read_csv(path(d, c))
    return dfs


def getBestX(dfs, id, by, x, lowIsGood):
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0, x)]
    return df_best_5


def rankBy(dfs, id, by, lowIsGood):
    df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
    df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
    return df_sorted


def getBestXFrom(df, by, x, lowIsGood):
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    df_best_5 = df_sorted.iloc[range(0, x)]
    return df_best_5


def getGraphXvsYrankedbyZ(
    dfs, x, x_unit, y, y_unit, z, lowIsGoodZ, dispatchNo, dispatchTitle
):
    # rankings
    ranked = rankBy(dfs, (dispatchNo, 1), z, lowIsGoodZ)
    cm = CustomMarker(marker=lambda x: f'${x["rank"]}$')
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
        scatterSets=[(ranked, cm)],
        legend=False,
    )


def getGraphXvsYrankedbyZtopQ(
    dfs, x, x_unit, y, y_unit, z, lowIsGoodZ, dispatchNo, dispatchTitle, q
):
    # rankings
    ranked = rankBy(dfs, (dispatchNo, 1), z, lowIsGoodZ)

    # if Q is invalid, graph ALL the rows instead.
    if (q <= 0) or (q > ranked.shape[0]):
        q = ranked.shape[0]
    top = getBestXFrom(ranked, "rank", q, True)
    cm = CustomMarker(marker=lambda x: f'${x["rank"]}$', label=lambda y: y["JSON Name"])
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
        scatterSets=[(top, cm)],
        legend=True,
        legend_pos="upper right",
    )


def getComparisonOfActualTimeVsEstimated(
    dfs, x, x_unit, y, y_unit, z, lowIsGoodZ, dispatchNo, dispatchTitle, q, c
):
    # rankings
    ranked = rankBy(dfs, (dispatchNo, 1), z, lowIsGoodZ)
    print(ranked)

    # if Q is invalid, graph ALL the rows instead.
    if (q <= 0) or (q > ranked.shape[0]):
        q = ranked.shape[0]

    top = getBestXFrom(ranked, "rank", q, True)
    cm = CustomMarker(
        marker=lambda x: f'${x["rank"]}$',
        label=lambda y: y["JSON Name"],
        stroke=lambda y: c,
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
            (top, cm)  # , (top2, cm2)
        ],
        legend=False,
        legend_pos="upper right",
    )


def frog(dfs, dispatchNo, dispatchTitle):
    # dfs,(dispatchNo,1)
    ranked = rankBy(dfs, (dispatchNo, 1), "Kernel Time", True)
    est_ranked = rankBy(dfs, (dispatchNo, 1), "Kernel Time Estimate", True)
    cm2 = CustomMarker(
        marker=lambda x: "o",
        label=lambda y: y["JSON Name"],
        fill=lambda y: "YellowGreen",
        stroke=lambda y: "Black",
    )
    cm1 = CustomMarker(fill=lambda y: "Purple")
    return Graph2D(
        keys=Keys2D(
            x="Row Dim",
            x_label="Row Dim",
            x_unit="8 byte elts",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="8 byte elts",
        ),
        title=dispatchTitle,
        scatterSets=[(ranked, cm1)],
        legend=True,
        legend_pos="upper right",
    )


def toad(dfs, dispatchNo, dispatchTitle):
    # dfs,(dispatchNo,1)
    ranked = rankBy(dfs, (dispatchNo, 1), "Kernel Time", True)
    est_ranked = rankBy(dfs, (dispatchNo, 1), "Kernel Time Estimate", True)
    cm2 = CustomMarker(
        marker=lambda x: "o",
        label=lambda y: y["JSON Name"],
        fill=lambda y: "YellowGreen",
        stroke=lambda y: "Black",
    )
    cm1 = CustomMarker(fill=lambda y: "Purple")
    return Graph2D(
        keys=Keys2D(
            x="Row Dim",
            x_label="Row Dim",
            x_unit="8 byte elts",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="8 byte elts",
        ),
        title=dispatchTitle,
        scatterSets=[(est_ranked, cm2)],
        legend=True,
        legend_pos="upper right",
    )


def dimsVsQuidditchTime(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    best = getBestXFrom(ranked, "Kernel Time", 10, True)
    a = Graph2D(
        keys=Keys2D(
            x="Reduction Dim",
            x_label="Reduction Dim",
            x_unit="elements",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                best,
                CustomMarker(
                    label=lambda y: f'({y["Row Dim"]},{y["Reduction Dim"]})',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2) * 2,
                    marker=lambda x: f'${x["rank"]}$',
                ),
            )
        ],
        legend=True,
        legend_pos="upper left",
        legend_bb=(1,1)
    )
    return a


# def rankBy(dfs, id, by, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_sorted["rank"] = range(1, df_sorted.shape[0] + 1)
#     return df_sorted
label: Callable[[T], str] = lambda y="no label": "_no label"


# def rankBy(dfs, id, by, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_sorted["rank"] = range(1, df_sorted.shape[0] + 1)
#     return df_sorted


def dimsVsEstimatedTime(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time Estimate", True)
    best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    b = Graph2D(
        keys=Keys2D(
            x="Reduction Dim",
            x_label="Reduction Dim",
            x_unit="elements",
            y="Kernel Time Estimate",
            y_label="Kernel Time Estimate",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                best,
                CustomMarker(
                    label= lambda y: f'({y["Row Dim"]},{y["Reduction Dim"]})',
                    marker=lambda x: f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2) * 2,
                    stroke=lambda x: "Purple",
                ),
            )
        ],
        legend=True,
        legend_pos="upper left",
        legend_bb=(1,1)
    )
    return b


def main():
    titles = [
        "Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        "Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        "Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
    ]
    # graphs = []
    # dfs = shortcutToData("../estimated_cycles_out")
    # top10 = {}
    # for dispNo, dispTitle in zip([1, 8, 7], titles):
    #     quid = dimsVsQuidditchTime(dfs, dispNo, dispTitle)
    #     est = dimsVsEstimatedTime(dfs, dispNo, dispTitle)
    #     graphs.append(quid)
    #     graphs.append(est)
    #     top10[dispNo] = (quid,est)
    # #graphEmAll((3, 2), graphs)

    # g = top10[7]
    # graphEmAll((1,2), (g[0],g[1]))
    # data = g[0].scatterSets[0][0]
    # print(data)
    # top5=getBestXFrom(data, "Kernel Time", 5, True)
    # print(top5)
    # print("hooodle")
    # for i in range(0,5):
    #     better = top5.iloc[i]["Kernel Time"]
    #     print(top5.iloc[i]["JSON Name"],end=" compared to\n")
    #     for j in range(i+1,5):  
    #         worse = top5.iloc[j]["Kernel Time"]
    #         percentage = ((worse + 0.0) - (better + 0.0) )/ (worse + 0.0)   * 100.0   
    #         print("\t",end=f'{top5.iloc[j]["JSON Name"]} is {percentage:.2f} % better\n')


if __name__ == "__main__":
    main()
