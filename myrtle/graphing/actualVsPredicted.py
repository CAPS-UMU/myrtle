from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graphing.graph_utils import Graph2D, Keys2D, CustomMarker
from graphing.new_graph_utils import graphEmAll, rankBy, graphWPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product, islice
from PIL import Image
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, SVR
import re

def main():
    args = sys.argv[1:]
    if len(args) >= 6:
        print(
            "USAGE: python3 sponge.py  <csvFilePath> <outputImgFilePath> <predictionMode>"
        )
        print("\twhere ")
        print("\t<csvFilePath> is the path to the csv file with observed times")
        print("\t<csvFilePath2> is the path to the csv file with myrtle-predicted times")
        print("\t<outputImgFilePath> is the path to write the output graph image")
        print('\t<predictionMode> is either "svrcyc", "scyc", or "sflt"')
        exit(1)
    mode = args[3]
    act = pd.read_csv(args[0])     # read in the first CSV
    pred = pd.read_csv(args[1])    # read in the second CSV
    # combine the CSVs, only bringing in observed fields we care about
    pred = pd.merge(pred, act[["JSON Name","Kernel Time"]],on="JSON Name",how="inner")
    pred = pd.merge(pred, act[["JSON Name","Kernel Name"]],on="JSON Name",how="inner")
    pred = pd.merge(pred, act[["JSON Name","Total Time"]],on="JSON Name",how="inner")
    df = pred 
    # extract dispatch number and dimensions
    dispatchRegex = re.compile(
        r"main\$async_dispatch_(\d+)_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64"
    )
    d, M, N, K = dispatchRegex.search(df["Kernel Name_y"][0]).groups()
    title = f"Dispatch {d}\nmatvec: <{M}x{K}>, <{N}x{K}> -> <{M}x{N}>"
    # rank by individual dispatch time
    ranked = rankBy(df, "Kernel Time_y", lowIsGood=True)
    if len(args) == 5:
        topX=int(args[4])
        if topX != -1:
            ranked = ranked[ranked["rank"] < topX]
    if mode == "sflt": # graph filtering steps
        suffix = "rank-actual-filtered"
        outputPath = f"{args[2]}/d{d}-{M}-{N}-{K}-{suffix}"
        def pickColor(x):
            colors = ['#c0c8d1','#98c1f2','#468fe7','Purple']
            return colors[x["stage"]]
        x = ("rank", "Rank", "fastest to slowest Dispatch Time")
        y1 = ("Kernel Time_y", "Time", "cycles", pickColor)
        g = graphXvsYs(ranked, x, [y1], title, outputPath, 0.94)
        graphWPatch(g, 8, 10, patch_func_sflt)
    else:  # graph Actual vs Predicted Kernel Time
        suffix = f"rank-actual-predicted-{mode}"
        outputPath = f"{args[2]}/d{d}-{M}-{N}-{K}-{suffix}"
        predName = "Predicted Kernel Time" if mode == "svrcyc" else "Kernel Time Estimate"
        x = ("rank", "Rank", "fastest to slowest Dispatch Time")
        y1 = ("Kernel Time_y", "Time", "cycles", lambda x: "Black")
        y2 = (predName, "Predicted Time", "cycles", lambda x: "Purple")
        g = graphXvsYs(ranked, x, [y1, y2], title, outputPath, 0.95)
        def patch_func(ax):
            gWidth = ax.get_xlim()[1]
            gHeight = ax.get_ylim()[1]
            pnt1GW = 0.1 * gWidth
            # first set of points
            width = (gWidth - 2 * pnt1GW) / 2 / 4 / 2
            height = gHeight * 0.04
            left = gWidth + pnt1GW
            bottom = gHeight - height
            rect = plt.Rectangle(
                (left, bottom),
                width,
                height,
                edgecolor="black",
                facecolor="black",
                alpha=1.0,
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(left + 1.25 * width, bottom + 0.3 * height, "Dispatch Time (Actual)")
            # second set of points
            left = left + width + gWidth/2
            rect = plt.Rectangle(
                (left, bottom),
                width,
                height,
                edgecolor="purple",
                facecolor="purple",
                alpha=1.0,
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(left + 1.25 * width, bottom + 0.3 * height, "Dispatch Time (Predicted)")
        graphWPatch(g, 8, 10, patch_func)


def graphXvsYs(df, x, ys, title, outputPath, table_bb_height):
    tableData = df[
        [
            "rankAsStr",
            "m",
            "Row Dim",
            "Reduction Dim",
            "Little N Prime",
        ]
    ]
    colLabels = ["rank", "m","n", "k", "n'"]
    defW = 1 / (len(colLabels) * 3)  # default width
    tableColWidths = [
        defW,
        defW,
        defW * 1.5,
        defW * 1.5,
        defW * 1.25,
        defW * 0.5,
        defW * 0.5,
    ]
    keys = Keys2D(
        x=x[0],
        x_label=x[1],
        x_unit=x[2],
        y=ys[0][0],
        y_label=ys[0][1],
        y_unit=ys[0][2],
    )
    scatterSetz = []
    for y in ys:
        ss = (
            df,
            CustomMarker(
                y=y[0],
                label=lambda y: f'    {y["JSON Name"]}',
                marker=lambda x: f'${x["rank"]}$',
                size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2) * 2,
                stroke=y[3],
                fill=lambda x: "Black",
            ),
        )
        scatterSetz.append(ss)
    return Graph2D(
        imagePath=outputPath,
        keys=keys,
        title=title,
        scatterSets=scatterSetz,
        legend=False,
        table=True,
        table_pos="right",  # Bbox or [xmin, ymin, width, height]
        table_bb=(1.01, 0, 1, table_bb_height),
        table_col_widths=tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData,
    )


def patch_func_sflt(ax):
    gWidth = ax.get_xlim()[1]
    gHeight = ax.get_ylim()[1]
    pnt1GW = 0.05 * gWidth
    colors = ['#c0c8d1','#98c1f2','#468fe7','Purple']
    # 1st filter
    width = (gWidth - 4 * pnt1GW) / 5 / 2
    height = gHeight * 0.01
    left = gWidth + pnt1GW
    bottom = gHeight - height
    print(f'x limit: {ax.get_xlim()}')
    print(f'f limit: {ax.get_ylim()}')
    print(f'gWidth is {gWidth}, gHeight is {gHeight}, patch height is {height}')
   
    rect = plt.Rectangle(
        (left, bottom),
        width,
        height,
        edgecolor=colors[1],
        facecolor=colors[1],
        alpha=1.0,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(left + 1.25 * width, bottom + 0.4 * height, "Min. Microkernel Runs")
    # second filter
    left = left + ((gWidth - 4 * pnt1GW) / 3) + 2.5*pnt1GW
    rect = plt.Rectangle(
        (left, bottom),
        width,
        height,
        edgecolor=colors[2],
        facecolor=colors[2],
        alpha=1.0,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(left + 1.25 * width, bottom + 0.3 * height, "Max. L1 Usage")
    # final filter
    left = left + ((gWidth - 4 * pnt1GW) / 3) + 0.5*pnt1GW
    rect = plt.Rectangle(
        (left, bottom),
        width,
        height,
        edgecolor=colors[3],
        facecolor=colors[3],
        alpha=1.0,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(left + 1.25 * width, bottom + 0.3 * height, "Min. Regular Loads")


if __name__ == "__main__":
    main()
