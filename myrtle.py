import sys
import json
import myrtle.tile_size_generator as tsg
import re
import pickle
import pandas as pd
from sklearn.svm import SVC, SVR
import os


# def rankBy(dfs, id, by, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
#     return df_sorted

# def getBestX(dfs, id, by, x, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_best_5 = df_sorted.iloc[range(0, x)]
#     return df_best_5

def tileSelection(csvFile):
    df = pd.read_csv(csvFile)
    myLoc=os.path.abspath(__file__)[:-(len("myrtle.py"))]   
    file = open(f'{myLoc}/myrtle/dispatch-8-svr.pickle', 'rb')
    svr=pickle.load(file)
    df["Predicted Kernel Time"] = df.apply(lambda y: svr.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
    ranked = df.sort_values("Predicted Kernel Time", ascending=True)
    m = 1 #TODO: expand tiling to matmul!!
    n = int(ranked.iloc[0]["Row Dim"])
    k = int(ranked.iloc[0]["Reduction Dim"])
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
    m,n,k,dualBuffer = tileSelection(f'{M}x{N}x{K}wm-n-k_case1_searchSpace.csv')   
    if sys.argv[2] == "sflt":
        print("someday, use simple filtering to select tiles...")
    else:
        if sys.argv[2] == "scyc":
            print("someday, use simply cycle count predictor to select tiles...")
        else:
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

