# plot tile dimensions vs cycles
import pandas as pd
import re
from graphing.graph_utils import graphEmAll, Graph2D, Keys2D, CustomMarker, Curve
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
    # cycles = 96.0/7.0 * reductionDim + 190.0/7
    b = 0.0
    if rowDim == 1:
        b = 37.0
    if rowDim > 1 :
        b = 300
    cycles = 4.0 * reductionDim + b
    # cycles =5.0*reductionDim+100.0
    # rowDimCycles = outerLoops(reductionDim) *rowDim**2
    # rowDimCycles = rowDim+2**outerLoops(reductionDim) / 500
    # reductionDimCycles = unrollAndJamFactor(reductionDim)
    # cycles = rowDimCycles + reductionDimCycles
    #cycles = rowDim * outerLoops(reductionDim)*unrollAndJamFactor(reductionDim)+outerLoops(reductionDim)*20
    # cycles = reductionDim*3+rowDim*outerLoops(reductionDim)
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

def unrollAndJamGraphs(df):
    a = Graph2D(
        keys=Keys2D(
            x="CC Reduction Dim",
            x_label="Reduction Dim",
            x_unit="8 byte elements",
            y='UnrollAndJam Factor',
            y_label="unroll and jam factor",
            y_unit="instructions",
        ),
        title="Reduction Dim vs. Unroll and Jam Factor",
        scatterSets=[(df, CustomMarker())],
        legend=False,
    )
    b = Graph2D(
        keys=Keys2D(
            x="CC Reduction Dim",
            x_label="Reduction Dim",
            x_unit="8 byte elements",
            y='UnrollAndJam Outer Loops',
            y_label="outer loops",
            y_unit="loop count",
        ),
        title="Reduction Dim vs. Outer Loops",
        scatterSets=[(df, CustomMarker())],
        legend=False,
    )
    return [a,b]

# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])

# # Perform linear fit
# coefficients = np.polyfit(x, y, 1)
# print("Linear Fit Coefficients:", coefficients)

# # Create polynomial function
# p = np.poly1d(coefficients)

# plt.scatter(x, y, label='Data Points')
# plt.plot(x, p(x), label='Linear Fit', color='red')
# plt.legend()
# plt.show()

def approximateLinesGivenFixedRowDim(df):
    rowDims = range(1,10)
    colors = ["lightcoral","sienna","orangered","darkslateblue","orange","red","peachpuff","mediumvioletred","lightpink","blue"]

    lines = []
    for rowDim in rowDims:
        c = colors[rowDim]
        fixedRowD = df.loc[df["CC Row Dim"]==rowDim]
        # Perform linear fit
        x = pd.DataFrame.to_numpy(fixedRowD[["CC Reduction Dim"]]).flatten().tolist()
        y = pd.DataFrame.to_numpy(fixedRowD[["linalg_xdsl"]]).flatten().tolist()
        print(x) 
        print(y) 
        #coefficients = np.polyfit(x[0], y[0], 1)
        #p = np.poly1d(coefficients)
        p = np.polynomial.polynomial.Polynomial.fit(x, y, 1)
        print(f"for rowDim {rowDim}, y intercept is {np.polynomial.polynomial.polyval(0,p.coef)} and roots are {p.coef}")
        print(type(p))
        print(p)
        line = Curve(fixedRowD[["CC Reduction Dim"]],p,c,label=f'Linear Fit when Row Dim is {rowDim}')
        lines.append(line)
        # print(np.array(df.loc[df["CC Row Dim"]==rowDim][["CC Row Dim","CC Reduction Dim","linalg_xdsl"]]))
        # print(rowDim)
    return lines

def approximateLine(df):
    rowDims = [200]
    fixedRowD = df.loc[df["CC Reduction Dim"]==100]
    print(df.loc[df["CC Reduction Dim"]==100])
    lines = []
    for rowDim in rowDims:
        c = "mediumvioletred"
      
        # Perform linear fit
        x = pd.DataFrame.to_numpy(fixedRowD[["CC Row Dim"]]).flatten().tolist()
        y = pd.DataFrame.to_numpy(fixedRowD[["linalg_xdsl"]]).flatten().tolist()
        print(x) 
        print(y) 
        #coefficients = np.polyfit(x[0], y[0], 1)
        #p = np.poly1d(coefficients)
        p = np.polynomial.polynomial.Polynomial.fit(x, y, 2)
        print(f"for rowDim {rowDim}, y intercept is {np.polynomial.polynomial.polyval(0,p.coef)} and roots are {p.coef}")
        print(type(p))
        print(p)
        line = Curve(fixedRowD[["CC Row Dim"]],p,c,label=f'Fit when Reduction Dim is {rowDim}')
        lines.append(line)
        # print(np.array(df.loc[df["CC Row Dim"]==rowDim][["CC Row Dim","CC Reduction Dim","linalg_xdsl"]]))
        # print(rowDim)
    return lines

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
    # add unrollAndJam info for each tile
    df['UnrollAndJam Factor'] = df.apply(lambda y: unrollAndJamFactor(y["CC Reduction Dim"]), axis=1)
    df['UnrollAndJam Outer Loops'] = df.apply(lambda y: outerLoops(y["CC Reduction Dim"]), axis=1)
    # print(df)
    for d in [14,161,12,6,1]:
        print(f"unroll and jam factor for {d} is {unrollAndJamFactor(d)} with outerLoops {outerLoops(d)}")
    
    print(f"cycle estimate for (1x30) is {estimateCycles(1,30)}")
    rowVsTime = simpleDimsVsTime(df,"CC Row Dim","Row Dim")
    rowVsTime.legend=True
    # rowVsTime.curves=approximateLine(df)
    # rowVsTime.curves[0].data= df[["CC Row Dim"]]
    rowVsTime.legend_pos="upper left"
    rowVsTime.legend_bb=(0,1)
    reductionVsTime = simpleDimsVsTime(df,"CC Reduction Dim","Col Dim")
    # rowVsTime.legend=True
    reductionVsTime.legend=True
    reductionVsTime.legend_pos="upper left"
    reductionVsTime.legend_bb=(0,1)
    lines = approximateLinesGivenFixedRowDim(df.loc[df["CC Reduction Dim"]<40])
    for line in lines:
        line.data = df[["CC Reduction Dim"]]
    reductionVsTime.curves = lines#approximateLinesGivenFixedRowDim(df.loc[df["CC Reduction Dim"]<40])
    graphEmAll((1, 2), [rowVsTime,   
                        reductionVsTime])
    #graphEmAll((1, 1), [reductionVsTime])

    #graphEmAll((1, 2), [simpleDimsVsTime(df,"CC Row Dim","Row Dim"),   
                        #simpleDimsVsTime(df,"CC Reduction Dim","Col Dim")])
    #df.loc[df['column_name'] == some_value]
    #graphEmAll((1, 2), [inputSizeVsTime(df), dimsVsTime(df)])
    # graphEmAll((2, 2), [simpleDimsVsTime(df,"CC Row Dim","Row Dim"), 
    #                     simpleDimsVsTime(df,"CC Reduction Dim","Col Dim"),
    #                     simpleDimsVsTime(df.loc[df["CC Row Dim"]==5],'UnrollAndJam Factor',"Unroll and Jam Factor"),
    #                     simpleDimsVsTime(df.loc[df["CC Row Dim"]==5],'UnrollAndJam Outer Loops',"Outer Loops"),
    #                     ])

    # graphEmAll((1, 2), examinations(df.loc[df["CC Row Dim"]==3]))
    #graphEmAll((1, 2), unrollAndJamGraphs(df))
    # graphEmAll((1, 2), [simpleDimsVsTime(df.loc[df["CC Reduction Dim"]==30],"CC Row Dim","Row Dim"),   
    #                     simpleDimsVsTime(df.loc[df["CC Row Dim"]==3],"CC Reduction Dim","Col Dim")])
    

    # graphEmAll((2, 2), [simpleDimsVsTime(df,"CC Row Dim","Row Dim"), 
    #                     simpleDimsVsTime(df,"CC Reduction Dim","Col Dim"),
    #                     simpleDimsVsTimeVsEstimated(df,"CC Row Dim","Row Dim"),
    #                     simpleDimsVsTimeVsEstimated(df,"CC Reduction Dim","Col Dim"),
    #                     ])
    
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
