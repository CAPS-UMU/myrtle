# plot tile dimensions vs cycles
import pandas as pd
import re
from graphing.graph_utils import graphEmAll, Graph2D, Keys2D, CustomMarker
import matplotlib.pyplot as plt
import numpy as np

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

def simpleDimsVsTime(df,col_name,label):
    return Graph2D(
        keys=Keys2D(
            x=col_name,
            x_label=label,
            x_unit="8 byte elements",
            y="linalg_xdsl",
            y_label="Execution Time",
            y_unit="cycles",
        ),
        title=f"Microkernel {label} vs Time",
        scatterSets=[(df, CustomMarker())],
        legend=False,
    )

def simpleDimsVsTimeVsEstimated(df,col_name,label):
    cm = CustomMarker()
    cm.fill=lambda y="Red": "Red"
    est = estimate(df)
    print(est)
    return Graph2D(
        keys=Keys2D(
            x=col_name,
            x_label=label,
            x_unit="8 byte elements",
            y="linalg_xdsl",
            y_label="Execution Time",
            y_unit="cycles",
        ),
        title=f"Microkernel {label} vs Time",
        scatterSets=[(df, CustomMarker()),(est,cm)],
        legend=False
    )

# def estReductionDim(row):
#     dim = row["CC Reduction Dim"]
#     cycles =0.5*dim+100.0
#     return cycles
# the unroll and jam factor is the 
# largest divisor of the reduction dimension, 
# picked out of the list 1,2,3,4,5,6, or 7.
def unrollAndJamFactor(reductionDim):
    options = [7,6,5,4,3,2]
    factor = 1
    for option in options:
        if reductionDim % option == 0:
            factor = option
            break
    return factor

def outerLoops(reductionDim):
    return reductionDim / unrollAndJamFactor(reductionDim)

def estimateCycles(rowDim,reductionDim):
    # cycles =5.0*reductionDim+100.0
   # rowDimCycles = outerLoops(reductionDim) *rowDim**2
    # rowDimCycles = rowDim+2**outerLoops(reductionDim) / 500
    # reductionDimCycles = unrollAndJamFactor(reductionDim)
    # cycles = rowDimCycles + reductionDimCycles
    #cycles = rowDim * outerLoops(reductionDim)*unrollAndJamFactor(reductionDim)+outerLoops(reductionDim)*20
    cycles = reductionDim*3+rowDim*outerLoops(reductionDim)
    #cycles =5.0*rowDim*reductionDim*outerLoops(reductionDim) / 200.0#+100.0
    #cycles = outerLoops(reductionDim)*rowDim + unrollAndJamFactor(reductionDim)
    #cycles = outerLoops(reductionDim)*rowDim + unrollAndJamFactor(reductionDim)*outerLoops(reductionDim)
    return cycles


def estimate(df):
    est = df.copy()
    est['linalg_xdsl'] = est.apply(lambda y: estimateCycles(y["CC Row Dim"],y["CC Reduction Dim"]), axis=1)
    #est['linalg_xdsl']=est["CC Reduction Dim"]*2.0 + 100.0
    #est = est[["CC Row Dim","CC Reduction Dim",'linalg_xdsl']]
    return est

def main():
    computeCoreDataToRead = "./graphing/toGraph/pivoted.cost_model.csv"
    computeCoreDataToRead = "./graphing/toGraph/pivoted.cost_model_30_thru_201.csv"
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
    for d in [14,161,12,6,1]:
        print(f"unroll and jam factor for {d} is {unrollAndJamFactor(d)} with outerLoops {outerLoops(d)}")
    
    # graphEmAll((1, 2), [inputSizeVsTime(df), dimsVsTime(df)])
    graphEmAll((2, 2), [simpleDimsVsTime(df,"CC Row Dim","Row Dim"), 
                        simpleDimsVsTime(df,"CC Reduction Dim","Col Dim"),
                        simpleDimsVsTimeVsEstimated(df,"CC Row Dim","Row Dim"),
                        simpleDimsVsTimeVsEstimated(df,"CC Reduction Dim","Col Dim"),
                        ])

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for index, row in df.iterrows():
    #         ax.scatter(row["CC Row Dim"], row["CC Reduction Dim"], row["linalg_xdsl"],marker="o",c="YellowGreen",edgecolors="Black")
    # ax.set_xlabel("Row Dim")
    # ax.set_ylabel("Reduction Dim")
    # ax.set_zlabel("Cycles")
    #plt.show()
   


if __name__ == "__main__":
    main()
