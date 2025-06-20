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
# example run: python graphing-refactored.py /home/emily/myrtle/estimated_cycles_overhead
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
    #print(dfs[(1,1)]["rank"])
    deriveMoreData(dfs,dispatchOrder,caseNos)
    print(dfs[(8,1)]["rank"])
    print(dfs[(7,1)]["rank"])
    print(dfs[(1,1)]["rank"])
    # Quidditch Time (flops and not flops) vs Attributes
    graphs = []
    for dispNo in dispatchOrder:
        for caseNo in caseNos:
            title = titles[dispNo]    
            lftGraph = lookForTrends(dfs, dispNo, caseNo, title)
            graphs.append(lftGraph)
    
    for g in graphs:
        graphEmAll((1, 1), [g])
    # Predicted Quidditch time vs Actual Quidditch Time   
    
    print("HASTA LUEGO")

def lookForTrends(dfs, dispNo, caseNo, dispTitle):
    tableData = dfs[(dispNo,caseNo)][["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J","CC OLs","Micro Runs","RLs","Reused SLs","L1 Usage","n","k"]
    defW = (1/(len(colLabels)*3)) # default width
    tableColWidths = [defW*0.6,defW*0.5,defW*0.4,defW,defW*1.25,defW,defW*1.5,defW*1.6,defW*0.5,defW*0.5]
    return Graph2D(
        imagePath=f'LFT-dispatch-{dispNo}-case-{caseNo}',
        keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            # x="Space Needed in L1",
            # x_label="L1 Usage",
            # x_unit="bytes",
            # x="Microkernel Count",
            # x_label="Microkernel Runs",
            # x_unit="",
            y="FLOP Per Cycle",
            y_label="Throughput",
            y_unit="FLOPs per cycle",
        ),
        title=dispTitle,
        scatterSets=[
            (
                dfs[(dispNo,caseNo)],#tableData,#dfs[(dispNo, 1)],
                CustomMarker(
                    y="FLOP Per Cycle",
                    label= lambda y: f'    {y["JSON Name"]}', 
                    # label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',                   
                    marker=lambda x: f'${x["rank"]}$',#f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: 'Black',#"Purple",
                    fill=lambda x: 'Black'#"Purple",
                ),
            )
        ],
        legend = False,
        table = True,
        table_pos="right",
        table_bb=(1.01,0,1,1), #self.scale(rw / w, rh / h)
        table_col_widths = tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData
    )

if __name__ == "__main__":
    main()