from dataclasses import dataclass, field
import pandas as pd
import sys
import quidditch_load_counting as qlc
from utils import unrollAndJamFactor,unrollAndJamOuterLoops
import pickle

# open a file, where you stored the pickled data
# file = open('important', 'rb')

# # dump information to that file
# data = pickle.load(file)

# # close the file
# file.close()

# print('Showing the pickled data:')

# cnt = 0
# for item in data:
#     print('The data ', cnt, ' is : ', item)
#     cnt += 1

class Myrtle:
  def __init__(self, timeEstimateFuncs):
    self.timeEstimateFuncs = timeEstimateFuncs

  def get_cycle_estimate(self,row_dim, col_dim, outerLoopIters, microCount): #, n, k):
    if outerLoopIters == 1:
       return self.timeEstimateFuncs[row_dim](col_dim) * microCount
    else: # for ex, microkernel tile of 10 x 50 will have
          # outer loop iters = unroll and jam outer loops = 2
          # unroll and jam factor of 5
          # so select function that estimates execution of microkernel
          # with row dimension 10 / 2 = 5, and multiply that by outer loops
          # TODO: ALSO, ADD A CONSTANT TO ACCOUNT FOR OVERHEAD OF SETTING UP STREAMING REGISTERS
       return self.timeEstimateFuncs[row_dim/outerLoopIters](col_dim)*microCount + outerLoopIters*100




    

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
    print(f"args[0] x args[1]: {args[0]} x {args[1]}")
   
    df=pd.read_csv(args[2])
    dispatchName = args[3]
    caseNo = int(args[4])

    # add unrollAndJam info for each tile
    df['UnrollAndJam Factor'] = df.apply(lambda y: unrollAndJamFactor(y["Microkernel Row Dim"]), axis=1)
    df['UnrollAndJam Outer Loops'] = df.apply(lambda y: unrollAndJamOuterLoops(y["Microkernel Row Dim"]), axis=1)

    linearApproxFilePath = "linesOfBestFit.pickle"
    file = open(linearApproxFilePath, 'rb')
    curves = pickle.load(file)
    lines = {}
    for c in curves:
        lines[c.id]=c.func
    print(lines)
    m = Myrtle(lines)

    df["Kernel Time Estimate"] = df.apply(lambda x: m.get_cycle_estimate(x["Microkernel Row Dim"], x["Microkernel Reduction Dim"],x["Outer Loop Iters"],x["Microkernel Count"]), axis=1)
    # print(df[["JSON Name","Row Dim","Reduction Dim","Kernel Time","Microkernel Row Dim","Microkernel Reduction Dim","Microkernel Count","Kernel Time Estimate"]])
    #print(df[["JSON Name","Microkernel Reduction Dim",'Outer Loop Iters','UnrollAndJam Outer Loops']])
    print(df[["JSON Name","Kernel Time",'Kernel Time Estimate']])
    # df.to_csv(f"estimated_cycles_out/{dispatchName}_case{caseNo}_everything.csv",index=False)
    #df.to_csv(f"estimated_cycles_out_2/{dispatchName}_case{caseNo}_everything.csv",index=False)
    # instead of above line, need to write to the folder estimated-cycles-overhead
    df.to_csv(f"estimated_cycles_overhead/{dispatchName}_case{caseNo}_everything.csv",index=False)   
    #df.to_csv(f"estimated_cycles_no_overhead/{dispatchName}_case{caseNo}_everything.csv",index=False)   
    qlc.yodel()


if __name__ == "__main__":
    main()