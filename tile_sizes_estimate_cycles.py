from dataclasses import dataclass, field
import pandas as pd
import sys
import quidditch_load_counting as qlc
from utils import InputMatrix, TileSizes, roundUpToNearestMultipleOf

def get_cycle_estimate(row_dim, col_dim): #, n, k):
    outerLoops = "???"
   # return "hoodle"
   # res = f"{row_dim}x{col_dim} X {outerLoops} outer loops; n={n} and k={k}"
    return row_dim
    

def addMoreColsForConvenience(df):
    df["Total Loads"] = df["Regular Loads"] + df["Total Streaming Loads"]
    #df["Tile Shape"] = df.apply(lambda x: get_tile_shape(x["Row Dim"], x["Reduction Dim"]), axis=1)
    return df

def main():
    args = sys.argv[1:]
    if len(args) != 5:
      print("USAGE: python3 tile_sizes_estimate_cycles.py <N> <K> <tilesRequested.csv> <dispatchName> <CaseNo>")
      print(
            "\twhere N, K, refer to dimensions in a\n\tmatrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`"
      )
      exit(1)
    print(f"args[0] is {args[0]}")
    print(f"args[1] is {args[1]}")
    df=pd.read_csv(args[2])
    dispatchName = args[3]
    caseNo = int(args[4])
    df["Kernel Time Estimate"] = df.apply(lambda x: get_cycle_estimate(x["Row Dim"], x["Reduction Dim"]), axis=1)
    print(df[["JSON Name","Row Dim","Reduction Dim","Kernel Time","Kernel Time Estimate"]])

    df.to_csv(f"estimated_cycles_out/{dispatchName}_case{caseNo}_everything.csv",index=False)
    
       
    qlc.yodel()


if __name__ == "__main__":
    main()