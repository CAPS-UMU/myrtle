# plot tile dimensions vs cycles
# SHAPE_3D = r"(?P<M>\d+)x(?P<K>\d+)x(?P<N>\d+)xf(?P<precision>\d+)"
# pattern = re.compile(KERNEL_SHAPE[wildcards.kernel])
#         match = pattern.fullmatch(wildcards.shape)
#         assert match
# pd.read_csv

# path = (
#         lambda x, y: f"./toGraph/dispatch_{x}_case{y}_everything.csv"
# #     )

# match = pattern.fullmatch(wildcards.shape)
# #         assert match
#best_x["Regular / Total Loads"]=best_x.apply(lambda x: (x["Regular Loads"] / x["Total Loads"]) * 100.0, axis=1)

import pandas as pd
import re

from graphing.graph_dispatches import graphEmAll, Graph2D,Keys2D

def extractDims(x):
    pattern = re.compile(r"\w* (?P<M>\d+)x(?P<K>\d+)x(?P<N>\d+)xf(?P<precision>\d+)")
    match = pattern.fullmatch(x)
    assert match
    n = int(match.groupdict()["N"])
    k =  int(match.groupdict()["K"])
    return {"n":n,"k":k}

def inputSizeVsTime(df):
    return Graph2D(
        keys=Keys2D(
            x="CC Row * Reduction Dim",
            x_label="Input Matrix Size",
            x_unit="8 byte elements",
            y="linalg_xdsl",
            y_label="Execution Time",
            y_unit="cycles",
        ),
        title="Microkernel Input Size vs Time",
        scatterSets=[
            ("YellowGreen", "Black", df, "no padding")
        ],
        legend=False
    )

def dimsVsTime(df):
    return Graph2D(
        keys=Keys2D(
            x="CC Reduction Dim",
            x_label="CC Reduction Dim",
            x_unit="elements",
            y="linalg_xdsl",
            y_label="Execution Time",
            y_unit="cycles",
        ),
        title="Reduction Dim vs Time vs Row Dim",
        scatterSets=[
            ("YellowGreen", "Black", df, "Row Dim")
        ],
        legend=False, 
        legend_pos="upper right",
        custom_marker = True, 
        get_marker = lambda x : f'${x["CC Row Dim"]}$',
        get_marker_label = lambda y : y["CC Row Dim"]
    )

def main():
    computeCoreDataToRead = "./graphing/toGraph/pivoted.cost_model.csv"
    timeCol = "linalg_xdsl"
    nameCol = "kernels"
    df = pd.read_csv(computeCoreDataToRead)
    # extract input dimensions from the name of each kernel run
    df["CC Row Dim"] = df.apply(lambda row:  (extractDims(row[nameCol])["n"]),axis=1)
    df["CC Reduction Dim"] = df.apply(lambda row:  (extractDims(row[nameCol])["k"]),axis=1)
    # add "area" of each tile
    df["CC Row * Reduction Dim"] = df.apply(lambda row:  row["CC Row Dim"]*row["CC Reduction Dim"],axis=1)
    print(df)
    sizeVstime = inputSizeVsTime(df)   
    graphEmAll((1,2),[sizeVstime,dimsVsTime(df)])
    #graphEmAll((1,1),[sizeVstime])
    #graphEmAll((1,1),[dimsVsTime(df)])


if __name__ == "__main__":
    main()