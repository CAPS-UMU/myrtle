import re
import pickle
import pandas as pd
import sklearn.svm
from graphing.graph_utils import Curve
import pathlib

def labelThenTakeNSmallestX(df, x, n, df_record, label_name, label_val):
    # sort from least to greatest X
    df_sorted = df.sort_values(x, ascending=True)
    # take best N
    df_best_n = df_sorted.iloc[range(0, len(df_sorted)//n)]
    if label_val == -1:
       # print(f"YODEL taking {n} best but len(df_sorted)//n is {len(df_sorted)//n}")
        df_best_n = df_sorted.iloc[range(0, len(df_sorted))]
        # print(df_best_n[["JSON Name","Regular Loads","myrtle"]])
        # print(df_sorted[["JSON Name","Regular Loads"]])
        df_best_n[label_name] = range(1, int(df_sorted.shape[0] + 1))
        #print(df_best_n[["JSON Name","Regular Loads","myrtle"]])
        # mask = df_record["JSON Name"].isin(df_best_n["JSON Name"])
        # print("HELP")
        # print(df_record.loc[mask, label_name, "JSON Name"])
        # df_record.loc[mask, label_name]=df_best_n[label_name] 
        # df_record.update(df_best_n.set_index("JSON Name"), overwrite=True)
        # print(df_record[["JSON Name","Regular Loads","myrtle"]])
        # B = B.merge(A, on='id', how='left')
        df_record.set_index("JSON Name", inplace=True)
        df_best_n.set_index("JSON Name", inplace=True)
        # df_record = df_best_n.merge(df_record,on="JSON Name",how="left")
        df_record.update(df_best_n)
        df_record.reset_index(inplace=True)
        df_best_n.reset_index(inplace=True)
        # print("HELP")
        # print(df_record)
        df_best_n = df_sorted.iloc[range(0, len(df_sorted)//n)]
       # print(df_best_n[["JSON Name","Regular Loads","myrtle"]])
        return df_best_n
   
    # mark in record df which tiling schemes from df survived filter
    mask = df_record["JSON Name"].isin(df_best_n["JSON Name"])
    df_record.loc[mask, label_name]=label_val
    return df_best_n

def labelThenTakeNBiggestX(df, x, n, df_record, label_name, label_val):
    # print("df is ")
    # print(df[["JSON Name",x]])
    # sort from greatest to least X
    df_sorted = df.sort_values(x, ascending=False)
    # print("df_sorted is ")
    # print(df_sorted[["JSON Name",x]])
    # take best N
    df_best_n = df_sorted.iloc[range(0, len(df_sorted)//n)]
    # print("df_best_n is")
    # print(df_best_n[["JSON Name",x]])
    # mark in record df which tiling schemes from df survived filter
    mask = df_record["JSON Name"].isin(df_best_n["JSON Name"])
    # print("mask is")
    # print(mask)
    # print(f"df_record.loc[mask, {label_name}] is ")
    # print(df_record.loc[mask, label_name])
    df_record.loc[mask, label_name]=label_val
    return df_best_n


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
    basename = csvFile[:-(len("_ana.csv"))] 
    csvFileRanked = f"{basename}_sel_{mode}.csv"
    if mode == "svrcyc":
        file = open(f'{pathlib.Path(__file__).parent.resolve()}/dispatch-8-svr.pickle', 'rb')
        svr=pickle.load(file)
        df["Predicted Kernel Time"] = df.apply(lambda y: svr.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
        ranked = df.sort_values("Predicted Kernel Time", ascending=True)
        df = ranked
        df.to_csv(csvFileRanked,index=False)
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
            df.to_csv(csvFileRanked,index=False)
        else:
            stages=df
            stages["stage"]=0
            # minimize SSR configs performed
            if len(df)//3 <= 1: # only filter more if we have at least 2 more options
                print("\tTSS: ",end='')
                print("fewer than 4 options, so just apply the first filter.")
                filtered = labelThenTakeNSmallestX(df,"SSR Config Count", 1, stages, "stage", 1)
            else:
                filtered = labelThenTakeNSmallestX(df,"SSR Config Count", 3, stages, "stage", 1)           
            filtered = labelThenTakeNBiggestX(filtered,"Space Needed in L1", 2, stages, "stage", 2) 
            filtered = labelThenTakeNSmallestX(filtered,"Regular Loads", len(filtered), stages, "stage", 3)
            print("\t",end='')
            print(f'TSS: wrote ranking to file {csvFileRanked}')
            stages.to_csv(csvFileRanked,index=False)

            # save supplementary search space copies, sorted by a particular metric
            topSSRConfigs = df.sort_values("SSR Config Count", ascending=True)            
            topSSRConfigs.to_csv(f"{basename}_ord_ssrConfigs.csv",index=False)            
            # greedy baseline
            sortedByL1=df.sort_values("Space Needed in L1", ascending=False)
            sortedByL1.to_csv(f"{basename}_ord_L1.csv",index=False)
        
            
           
           

    # TODO: return the ONLY row with stage 3, NOT the first row
    m = int(df.iloc[0]["m"])
    n = int(df.iloc[0]["Row Dim"])
    k = int(df.iloc[0]["Reduction Dim"])
    dualBuffer = True
    return (m,n,k,dualBuffer)


