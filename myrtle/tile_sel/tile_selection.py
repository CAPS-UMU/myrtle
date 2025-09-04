import re
import pickle
import pandas as pd
import sklearn.svm
from graphing.graph_utils import Curve
import pathlib

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
    print("\t",end='')
    print(f'TSS: about to read in file {csvFile}')
    df = pd.read_csv(csvFile)
    basename = csvFile[:-(len(".csv"))]
   # myLoc=os.path.abspath(__file__)[:-(len("myrtle.py"))]  
    if mode == "svrcyc":
        file = open(f'{pathlib.Path(__file__).parent.resolve()}/dispatch-8-svr.pickle', 'rb')
        svr=pickle.load(file)
        df["Predicted Kernel Time"] = df.apply(lambda y: svr.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
        ranked = df.sort_values("Predicted Kernel Time", ascending=True)
        df = ranked
        df.to_csv(f"{basename}-myrtle-{mode}-ranking.csv",index=False)
    else: 
        if mode == "scyc":
            linearApproxFilePath = f'{pathlib.Path(__file__).parent.resolve()}/linesOfBestFit.pickle'
            file = open(linearApproxFilePath, 'rb')
            curves = pickle.load(file)
            lines = {}
            for c in curves:
                lines[c.id]=c.func
            df["Kernel Time Estimate"] = df.apply(lambda x: get_simple_cycle_estimate(lines,x["Little N Prime"], x["Little K"],x["UnrollAndJam Loop Iters"],x["SSR Config Count"]), axis=1)
            ranked = df.sort_values("Kernel Time Estimate", ascending=True)
            df = ranked
            df.to_csv(f"{basename}-myrtle-{mode}-ranking.csv",index=False)
        else:
            
            if(int(df["M"][0]) > 1):
                matmul_sflt(df,basename,mode)
            else:
                print(f'the dfs m value is {df["M"][0]}')
            # minimize microkernel runs
            #df_sorted = df.sort_values("Microkernel Count", ascending=True)
                df_sorted = df.sort_values("SSR Config Count", ascending=True)
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
            # print(f'final_ranking is {final_ranking}')
                # mark which tiling schemes survived final filter
                mask = stages["JSON Name"].isin(final_ranking.iloc[0])
                stages.loc[mask, "stage"]=3
                #print(stages[["JSON Name","stage"]])
                print("\t",end='')
                csvFileRanked = f"{basename}-myrtle-{mode}-ranking.csv"
                print(f'TSS: wrote ranking to file {csvFileRanked}')
                stages.to_csv(f"{basename}-myrtle-{mode}-ranking.csv",index=False)
    m = int(df.iloc[0]["m Dim"])
    n = int(df.iloc[0]["Row Dim"])
    k = int(df.iloc[0]["Reduction Dim"])
    dualBuffer = True
    return (m,n,k,dualBuffer)

def matmul_sflt(df,basename,mode):
    df_sorted = df.sort_values("SSR Config Count", ascending=True)
    stages=df_sorted
    stages["stage"]=0
    # stages=stages.sort_values("Kernel Time", ascending=True)
    # print(stages[["JSON Name","SSR Config Count","Space Needed in L1","Kernel Time","Weight Matrix Tile Size"]])
    # print(stages[["JSON Name","Total SSR Loads","A SSR Reuse Loads","Weight Matrix Tile Size"]])
    
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
    #final_ranking = df
    final_ranking = df_sorted.sort_values("UnrollAndJam Loop Iters", ascending=True)
    df = final_ranking
# print(f'final_ranking is {final_ranking}')
    # mark which tiling schemes survived final filter
    mask = stages["JSON Name"].isin(final_ranking.iloc[0])
    stages.loc[mask, "stage"]=3
    #print(stages[["JSON Name","stage"]])
    print("\t",end='')
    csvFileRanked = f"{basename}-myrtle-{mode}-ranking.csv"
    print(f'TSS: wrote ranking to file {csvFileRanked}')
    stages.to_csv(f"{basename}-myrtle-{mode}-ranking.csv",index=False)