from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, loadDFsDispatchCaseNo, deriveMoreData,rankBy, Graph2D, Keys2D, CustomMarker, generalGraph
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product, islice
from PIL import Image

# example run: python graphing-refactored.py "/home/hoppip/myrtle/estimated_cycles_no_overhead"

def main():
    args = sys.argv[1:]
    if len(args) != 1:
      print("USAGE: python3 graphing-refactored.py <tileData.csv>")
      print("\twhere <tileData.csv>  contains static characteristics, \n\testimated time based on microkernels, and actual time")
      exit(1)
    print("HOLA")
    caseNos = [1] # We only graph case 1; no padding anywhere.
    dispatchOrder = [8,7,1] # All dispatches will be graphed in the order 8, 7, 1.
    titles = {
        8:"Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        7:"Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
        1:"Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
    }
    print(f"Reading data from {args[0]}...")
    # load dispatch data
    dfs = loadDFsDispatchCaseNo(args[0], dispatchOrder, caseNos)
    # derive more data about each dispatch
    #print(dfs[(1,1)]["Kernel Name"])
    deriveMoreData(dfs,dispatchOrder,caseNos)
    # Quidditch Time (flops and not flops) vs Attributes

    # Predicted Quidditch time vs Actual Quidditch Time
    
    
    
    
    print("HASTA LUEGO")
  


if __name__ == "__main__":
    main()