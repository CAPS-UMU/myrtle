import sys
import json
import myrtle.tile_size_generator as tsg
import re
import pickle
import pandas as pd
import sklearn.svm
from graphing.graph_utils import Curve
import os


# def rankBy(dfs, id, by, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
#     return df_sorted

# def getBestX(dfs, id, by, x, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_best_5 = df_sorted.iloc[range(0, x)]
#     return df_best_5

def get_simple_cycle_estimate(timeEstimateFuncs, row_dim, col_dim, outerLoopIters, microCount): #, n, k):
    if outerLoopIters == 1:
       return timeEstimateFuncs[row_dim](col_dim) * microCount
    else: # for ex, microkernel tile of 10 x 50 will have
          # outer loop iters = unroll and jam outer loops = 2
          # unroll and jam factor of 5
          # so select function that estimates execution of microkernel
          # with row dimension 10 / 2 = 5, and multiply that by outer loops
          # TODO: ALSO, ADD A CONSTANT TO ACCOUNT FOR OVERHEAD OF SETTING UP STREAMING REGISTERS
       return timeEstimateFuncs[row_dim/outerLoopIters](col_dim)*microCount #+ outerLoopIters*100

def tileSelection(csvFile, mode):
    df = pd.read_csv(csvFile)
    myLoc=os.path.abspath(__file__)[:-(len("myrtle.py"))]  
    if mode == "svrcyc":
        file = open(f'{myLoc}/myrtle/dispatch-8-svr.pickle', 'rb')
        svr=pickle.load(file)
        df["Predicted Kernel Time"] = df.apply(lambda y: svr.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
        ranked = df.sort_values("Predicted Kernel Time", ascending=True)
        df = ranked
    if mode == "scyc":
        linearApproxFilePath = f'{myLoc}/myrtle/linesOfBestFit.pickle'
        file = open(linearApproxFilePath, 'rb')
        curves = pickle.load(file)
        lines = {}
        for c in curves:
            lines[c.id]=c.func
        df["Kernel Time Estimate"] = df.apply(lambda x: get_simple_cycle_estimate(lines,x["Microkernel Row Dim"], x["Microkernel Reduction Dim"],x["Outer Loop Iters"],x["Microkernel Count"]), axis=1)
        ranked = df.sort_values("Kernel Time Estimate", ascending=True)
        df = ranked
    else:
        # minimize microkernel runs
        df_sorted = df.sort_values("Microkernel Count", ascending=True)
        df_sorted = df_sorted.iloc[range(0, len(df_sorted)//3)]
        # maximize space used in L1
        df_sorted = df_sorted.sort_values("Space Needed in L1", ascending=False)
        df_sorted = df_sorted.iloc[range(0, len(df_sorted)//2)]
        # minimize regular loads
        final_ranking = df_sorted.sort_values("Regular Loads", ascending=False)
        df = final_ranking
    m = 1 #TODO: expand tiling to matmul!!
    n = int(df.iloc[0]["Row Dim"])
    k = int(df.iloc[0]["Reduction Dim"])
    dualBuffer = True
    return (m,n,k,dualBuffer)

# arg 1 is dispatchName as a string
# arg 2 is tile selection mode
# arg 3 is file to write tile scheme to
def main():
    print("yodelayheehoooooo")
    dispatchName = sys.argv[1]
    print (dispatchName)
    dispatchRegex=re.compile(r'main\$async_dispatch_\d+_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64')
    M,N,K = dispatchRegex.search(dispatchName).groups()
    # query myrtle!
    # generate options
    jen = tsg.TileSizeGenerator(int(N),int(K),dispatchName)
    options = jen.validOptions()
    jen.exportOptionsToCSV(f'{M}x{N}x{K}wm-n-k', 1, options)
    m,n,k,dualBuffer = tileSelection(f'{M}x{N}x{K}wm-n-k_case1_searchSpace.csv',sys.argv[2])   
    if sys.argv[2] == "sflt":
        print("We used simple filtering to select tiles.")
    if sys.argv[2] == "scyc":
        print("We used a simple cycle estimation to select tiles.") 
    if sys.argv[2] == "svrcyc":
        print("We used an SVR to select tiles.")   
    # default values
    with open(sys.argv[3], 'r') as file:
        data = json.load(file)
    node = {}    
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    
    # set node values and export to JSON
    node["tile-sizes"] = [[m], [n], [k]]
    node["dual-buffer"] = dualBuffer
    data[dispatchName]=node    
    f = open(sys.argv[3], "w") 
    f.write(f"{json.dumps(data)}")
    f.close()

   

if __name__ == "__main__":
    main()

