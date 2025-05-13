# plot tile dimensions vs cycles
import pandas as pd
import re
from graphing.graph_utils import graphEmAll, Graph2D, Keys2D, CustomMarker

def extractDims(x):
    pattern = re.compile(r"\w* (?P<M>\d+)x(?P<K>\d+)x(?P<N>\d+)xf(?P<precision>\d+)")
    match = pattern.fullmatch(x)
    assert match
    n = int(match.groupdict()["N"])
    k = int(match.groupdict()["K"])
    return {"n": n, "k": k}

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
        scatterSets=[(df, CustomMarker())],
        legend=False,
    )

def dimsVsTime(df):
    cm = CustomMarker(
        marker=lambda x: f'${x["CC Row Dim"]}$',
    )
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
        scatterSets=[(df, cm)],
        legend=False,
    )


def main():
    computeCoreDataToRead = "./graphing/toGraph/pivoted.cost_model.csv"
    #computeCoreDataToRead = "./graphing/toGraph/pivoted.cost_model_30_thru_201.csv"
    nameCol = "kernels"
    df = pd.read_csv(computeCoreDataToRead)
    # extract input dimensions from the name of each kernel run
    df["CC Row Dim"] = df.apply(lambda row: (extractDims(row[nameCol])["n"]), axis=1)
    df["CC Reduction Dim"] = df.apply(
        lambda row: (extractDims(row[nameCol])["k"]), axis=1
    )
    # add "area" of each tile
    df["CC Row * Reduction Dim"] = df.apply(
        lambda row: row["CC Row Dim"] * row["CC Reduction Dim"], axis=1
    )
    print(df)
    graphEmAll((1, 2), [inputSizeVsTime(df), dimsVsTime(df)])
   


if __name__ == "__main__":
    main()
