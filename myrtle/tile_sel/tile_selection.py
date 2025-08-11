import re
import pickle
import pandas as pd
import sklearn.svm
from graphing.graph_utils import Curve


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
    csvFileRanked = csvFile[:-(len(".csv"))]
   # myLoc=os.path.abspath(__file__)[:-(len("myrtle.py"))]  
    if mode == "svrcyc":
        file = open('dispatch-8-svr.pickle', 'rb')
        svr=pickle.load(file)
        df["Predicted Kernel Time"] = df.apply(lambda y: svr.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
        ranked = df.sort_values("Predicted Kernel Time", ascending=True)
        df = ranked
        df.to_csv(f"{csvFileRanked}-myrtle-{mode}-ranking.csv",index=False)
    else: 
        if mode == "scyc":
            linearApproxFilePath = 'linesOfBestFit.pickle'
            file = open(linearApproxFilePath, 'rb')
            curves = pickle.load(file)
            lines = {}
            for c in curves:
                lines[c.id]=c.func
            df["Kernel Time Estimate"] = df.apply(lambda x: get_simple_cycle_estimate(lines,x["Microkernel Row Dim"], x["Microkernel Reduction Dim"],x["Outer Loop Iters"],x["Microkernel Count"]), axis=1)
            ranked = df.sort_values("Kernel Time Estimate", ascending=True)
            df = ranked
            df.to_csv(f"{csvFileRanked}-myrtle-{mode}-ranking.csv",index=False)
        else:
            # minimize microkernel runs
            df_sorted = df.sort_values("Microkernel Count", ascending=True)
            stages=df_sorted
            stages["stage"]=0
            #sprint(stages[["JSON Name","stage"]])
            df_sorted = df_sorted.iloc[range(0, len(df_sorted)//3)]
            # mark which tiling schemes survived filter
            mask = stages["JSON Name"].isin(df_sorted["JSON Name"])
            stages.loc[mask, 'stage']=1
            #print(stages[["JSON Name","stage"]])
            
            # maximise L1 usage
            df_sorted = df_sorted.sort_values("Space Needed in L1", ascending=False)
            df_sorted = df_sorted.iloc[range(0, len(df_sorted)//2)]
            # mark which tiling schemes survived filter
            mask = stages["JSON Name"].isin(df_sorted["JSON Name"])
            stages.loc[mask, 'stage']=2
            #print(stages[["JSON Name","stage"]])
            
            # minimize regular loads
            final_ranking = df_sorted.sort_values("Regular Loads", ascending=True)
            df = final_ranking
            # mark which tiling schemes survived final filter
            mask = stages["JSON Name"].isin(final_ranking.iloc[0])
            stages.loc[mask, "stage"]=3
            #print(stages[["JSON Name","stage"]])
            print(f'myrtle: TSS: wrote ranking to file {csvFileRanked}')
            stages.to_csv(f"{csvFileRanked}-myrtle-{mode}-ranking.csv",index=False)
    m = 1 #TODO: expand tiling to matmul!!
    n = int(df.iloc[0]["Row Dim"])
    k = int(df.iloc[0]["Reduction Dim"])
    dualBuffer = True
    return (m,n,k,dualBuffer)