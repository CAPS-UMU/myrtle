import sys
import json
from tile_size_generation.TSG_Quidditch import TSG_Quidditch
from tile_size_generation.TSG_C import TSG_C
from tile_size_generation.TSG_C_Padding import TSG_C_Padding
from tile_static_analysis.TSA_Quidditch import TSA_Quidditch
# from tile_static_analysis.TSA_Manual_C_Code_deprecated import TSA_C
from tile_static_analysis.TSA_Manual_C_Code_Check import TSA_C_Check
from tile_static_analysis.TSA_C_Padding import TSA_C_Padding
import tile_sel.tile_selection as tss
import re
import pickle
import pandas as pd
import sklearn.svm
from graphing.graph_utils import Curve
import os

# arg 1 is dispatchName as a string (Quidditch Backend) or matmul dimensions (Manual C code Backend)
# arg 2 is tile selection mode
# arg 3 is file to write tile scheme to
# arg 4 is file to import tiling scheme candidates (skip tile gen)
# for Example,
# python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_1x400x200_f64" sflt "test_output-disp-0.json"
def main():
    dispatchName = sys.argv[1]
    quidditch = True
    if sys.argv[1][:6] == "matmul":
        print("\tTSG: we will prune for Manual C Backend")
        dispatchRegex=re.compile(r'matmul_(\d+)x(\d+)x(\d+)_f64')
        M,N,K = dispatchRegex.search(dispatchName).groups()
        dispatchNickName = f'{M}x{N}x{K}wm-n-k'
        quidditch=False
    else:
        print("\tTSG: we will prune for Quidditch Backend")
        dispatchRegex=re.compile(r'main\$async_dispatch_\d+_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64')
        M,N,K = dispatchRegex.search(dispatchName).groups()
        dispatchNickName = f'{M}x{N}x{K}wm-n-k'
    
    # take search space from command line if provided    
    if len(sys.argv) == 5: 
        # skip search space generation
        searchSpaceCSVName=sys.argv[4]
        print("myrtle: ",end='')
        print("Using search space passed in from command line.")
        options_as_df = pd.read_csv(searchSpaceCSVName)
    else:
        # generate options
        if quidditch:
            jen = TSG_Quidditch(int(M),int(N),int(K),dispatchName,l1MemoryBytes = 100000)
        else:
            jen = TSG_C(int(M),int(N),int(K),dispatchName=dispatchName,l1MemoryBytes = 112 * 1024, bank_size=1024, dualBuff=True)
        options = jen.validOptions(debug=False)
        options_as_df = jen.convertOptionsToDF(dispatchNickName, options)
        searchSpaceCSVName = jen.exportOptionsToCSV(dispatchNickName, options_as_df)
    
    # analyze tiling options
    ann=TSA_Quidditch() if quidditch else TSA_C_Check()
    analyzed = ann.analyze_options(options_as_df)
    analyzedSearchSpaceCSVName = ann.exportAnalysisToCSV(dispatchNickName, analyzed)

    # if using manual C backend, consider padding and generate analyzed search space for it.
    if not quidditch:
        gen = TSG_C_Padding(jen)
        paddedOptions = gen.validOptions(debug=False)
        paddedOptions_as_df = gen.convertOptionsToDF(dispatchNickName, paddedOptions)
        paddedSearchSpaceCSVName = gen.exportOptionsToCSV(dispatchNickName, paddedOptions_as_df)
        ann = TSA_C_Padding(ann)
        analyzed = ann.analyze_options(paddedOptions_as_df)
        analyzedPaddedSearchSpaceCSVName = ann.exportAnalysisToCSV(dispatchNickName, analyzed)

    # select best tiling scheme using mode        
    m,n,k,dualBuffer = tss.tileSelection(analyzedSearchSpaceCSVName,sys.argv[2])   
    if sys.argv[2] == "sflt":
        print("myrtle: ",end='')
        print("We used simple filtering to select tiles.")
    if sys.argv[2] == "scyc":
        print("myrtle: ",end='')
        print("We used a simple cycle estimation to select tiles.") 
    if sys.argv[2] == "svrcyc":
        print("myrtle: ",end='')
        print("We used an SVR to select tiles.")   
    # default values
    data = {}
    node = {}    
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    # set node values and export result to JSON
    node["tile-sizes"] = [[m], [n], [k]]
    node["dual-buffer"] = dualBuffer
    data[dispatchName]=node    
    f = open(sys.argv[3], "w") 
    f.write(f"{json.dumps(data)}")
    f.close()

   

if __name__ == "__main__":
    main()

