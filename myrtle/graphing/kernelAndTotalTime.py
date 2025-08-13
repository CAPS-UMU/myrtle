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
    if len(args) != 3:
        print(
            "USAGE: python3 reef.py  <csvFilePath> <outputImgFilePath> <predictionMode>"
        )
        print("\twhere ")
        print("\t<csvFilePath> is the path to the input csv file")
        print("\t<outputImgFilePath> is the path to write the output graph image")
        print('\t<predictionMode> is either "svrcyc", "scyc", or "sflt"')
        exit(1)

    df = pd.read_csv(args[0])  # read in the CSV
    # extract dispatch number and dimensions
    dispatchRegex = re.compile(
        r"main\$async_dispatch_(\d+)_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64"
    )
    d, M, N, K = dispatchRegex.search(df["Kernel Name"][0]).groups()
    # rank by individual dispatch time
    ranked = rankBy(df, "Kernel Time", lowIsGood=True)

    # graph rank vs Kernel Time AND Total Time
    suffix = "rank-kernel-total"
    x = ("rank", "Rank", "fastest to slowest Dispatch Time")
    y1 = ("Kernel Time", "Time", "cycles", lambda x: "Orange")
    y2 = ("Total Time", "NN Time", "cycles", lambda x: "Green")
    title = f"Dispatch {d}\nmatvec: <{M}x{K}>, <{N}x{K}> -> <{M}x{N}>"
    outputPath = f"{args[1]}/d{d}-{M}-{N}-{K}-{suffix}"
    g = graphXvsYs(ranked, x, [y1, y2], title, outputPath)

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
            edgecolor="orange",
            facecolor="orange",
            alpha=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(left + 1.25 * width, bottom + 0.3 * height, "Dispatch Time")
        # second set of points
        left = gWidth + (gWidth - 2 * pnt1GW) / 2
        rect = plt.Rectangle(
            (left, bottom),
            width,
            height,
            edgecolor="green",
            facecolor="green",
            alpha=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(left + 1.25 * width, bottom + 0.3 * height, "Total NN Time")

    graphWPatch(g, 8, 10, patch_func)


def graphXvsYs(df, x, ys, title, outputPath):
    tableData = df[
        [
            "rankAsStr",
            "Row Dim",
            "Reduction Dim",
            "Microkernel Row Dim",
        ]
    ]
    colLabels = ["rank", "n", "k", "n'"]
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
        table_bb=(1.01, 0, 1, 0.95),
        table_col_widths=tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData,
    )


if __name__ == "__main__":
    main()
