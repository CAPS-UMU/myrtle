from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, loadDFsDispatchCaseNo, deriveMoreData,addSVMPrediction,trimToTopX, Graph2D, Keys2D, CustomMarker, MySVM
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product, islice
from PIL import Image
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, SVR

# example run: python graphing-refactored.py "/home/hoppip/myrtle/estimated_cycles_no_overhead"
# example run: python graphing-refactored.py /home/emily/myrtle/estimated_cycles_no_overhead
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
    deriveMoreData(dfs,dispatchOrder,caseNos)
    # train some SVMs to predict runtime
    tunedOverhead = tryToFindOverHeadConstant(dfs,8,1,"tryToTuneOverheadCoef")
    generalSVM = tryGeneralSVM(dfs,8,1,"tryGeneralSVM")
    addSVMPrediction(dfs,dispatchOrder,caseNos,tunedOverhead)
    addSVMPrediction(dfs,dispatchOrder,caseNos,generalSVM)
    
    # graph ALL data points
    # Quidditch Time (FLOP/cycle) vs Attributes
    keysFLOPs = Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="FLOP Per Cycle", 
            y_label="Throughput",
            y_unit="FLOPs per cycle",
        )
    lookForTrendsDispatchCase(dfs,dispatchOrder,caseNos,titles,keysFLOPs,"hoodle","LFT-dispatches-rank-x-axis.png")
    keysFLOPs.x="Kernel Time"
    keysFLOPs.x_label="Kernel Time"
    keysFLOPs.x_unit="cycles"
    lookForTrendsDispatchCase(dfs,dispatchOrder,caseNos,titles,keysFLOPs,"hoodle","LFT-dispatches-kernel-time-x-axis.png")    
    # Actual Quidditch Time (cycles) VS Predicted Quidditch time (cycles)
    keysActualVsReal = Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualVsPredictedTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","Kernel Time Estimate",keysActualVsReal,"Estimate with Microkernel Time * Microkernel Runs",'ActualVsEstimate-3-dispatches-rank-x-axis.png')
    actualVsPredictedTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time",tunedOverhead.predName,keysActualVsReal,"Estimate with SVR-tuned overhead constant",'SVR-overhead-estimate-3-dispatches.png')
    actualVsPredictedTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time",generalSVM.predName,keysActualVsReal,"Estimate with general SVR ",'SVR-general-estimate-3-dispatches.png')
    
    # graph TOP TEN data points
    dfs = trimToTopX(dfs, dispatchOrder, caseNos, "rank", 10) # destructive!!
    lookForTrendsDispatchCase(dfs,dispatchOrder,caseNos,titles,keysFLOPs,"hoodle","LFT-dispatches-kernel-time-x-axis-top-10.png")
    print("HASTA LUEGO")

def tryToFindOverHeadConstant(dfs, dispNo, caseNo,predName="tryToTuneOverheadCoef"):
    ranked = dfs[(dispNo,caseNo)]
    feature_names = ["Kernel Time Estimate","UnrollAndJam Outer Loops"]
    target_name = "Kernel Time"
    train = ranked[feature_names]
    X = np.array(train)
    y = np.array(ranked[target_name])
    # Build the model
    svm = SVR(kernel="linear", gamma=0.5, C=1.0)
    # Train the model
    svm.fit(X, y)
    return MySVM(svm=svm,predName=predName,featureNames=feature_names)

def tryGeneralSVM(dfs, dispNo, caseNo,predName="tryGeneralSVM"):
    ranked = dfs[(dispNo,caseNo)]
    feature_names = ["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim", "UnrollAndJam Factor"]
    target_name = "Kernel Time"
    train = ranked[feature_names]
    X = np.array(train)
    y = np.array(ranked[target_name])
    # Build the model
    svm = SVR(kernel="linear", gamma=0.5, C=1.0)
    # Train the model
    svm.fit(X, y)
    return MySVM(svm=svm,predName=predName,featureNames=feature_names)

def combineDispatchesWithContext(graphs,img_title,img_name):
    graphImages = []
    width = 0.0
    biggest = None
    # find graph with max width and use this to set the width of the canvas
    for g in graphs:
        gImg = Image.open(f'{g.imagePath}.png')  
        if gImg.size[0] > width:
            width = gImg.size[0]
            biggest = gImg
        graphImages.append(gImg)          
    # resize context header to fit canvas
    orig = Image.open('context2.png') # hard coded
    resized = orig.copy()
    resized.thumbnail(biggest.size, Image.Resampling.LANCZOS)
    resized.save("resized.png") 
    # place each of the three dispatches on the canvas
    top = Image.open('resized.png')
    g0 = graphImages[0] 
    g1 = graphImages[1]
    g2 = graphImages[2] 
    canvas = Image.new('RGBA', (biggest.size[0], top.size[1]+g0.size[1]+g1.size[1]+g2.size[1]), (255, 255, 255, 255))
    canvas.paste(top, (0, 0), top)
    canvas.paste(g0, (0, top.size[1]), g0)
    canvas.paste(g1, (0, top.size[1]+g0.size[1]), g1)
    canvas.paste(g2, (0, top.size[1]+g0.size[1]+g1.size[1]), g2)
    canvas.save(img_name)
    top.close()
    g0.close()
    g1.close()
    g2.close()

def lookForTrendsDispatchCase(dfs, dispatchNos, caseNos, titles, keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="FLOP Per Cycle",
            y_label="Throughput",
            y_unit="FLOPs per cycle",
        ),imgTitle="Looking for Trends",imgName='LFT-3-dispatches.png',):
    graphs = []
    for dispNo in dispatchNos:
        for caseNo in caseNos:
            title = titles[dispNo]    
            lftGraph = lookForTrends(dfs, dispNo, caseNo, title,keys)
            graphs.append(lftGraph)
            graphEmAll((1, 1), [lftGraph])
    combineDispatchesWithContext(graphs,imgTitle,imgName)

def actualVsPredictedTimeDispatchCase(dfs, dispatchNos, caseNos, titles,y1="Kernel Time",y2="Kernel Time Estimate",keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),imgTitle="Actual vs Predicted Time",imgName='ActualVsEstimate-3-dispatches.png'):
    graphs = []
    for dispNo in dispatchNos:
        for caseNo in caseNos:
            title = titles[dispNo]    
            lftGraph = actualVsPredictedTime(dfs, dispNo, caseNo, title,y1,y2,keys)
            graphs.append(lftGraph)
            graphEmAll((1, 1), [lftGraph])
    combineDispatchesWithContext(graphs,imgTitle,imgName)
    

def actualVsPredictedTime(dfs, dispNo, caseNo, title, y1="Kernel Time",y2="Kernel Time Estimate",keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )):
    tableData = dfs[(dispNo,caseNo)][["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J Factor","CC Outer Loops","Micro Runs","n","k"]
    defW = (1/(len(colLabels)*3)) # default width
    tableColWidths = [defW,defW,defW*1.5,defW*1.5,defW*1.25,defW*0.5,defW*0.5]#[defW]*len(colLabels)
    return Graph2D(
        imagePath=f'dispatch-{dispNo}-case-{1}',
        keys=keys,
        title=title,
        scatterSets=[
            (
                dfs[(dispNo,caseNo)],
                CustomMarker(
                    y=y1,
                    label= lambda y: f'    {y["JSON Name"]}', 
                    marker=lambda x: f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: 'Black',
                    fill=lambda x: 'Black'
                ),
            ),
            (
                dfs[(dispNo,caseNo)],
                CustomMarker(
                    y=y2,
                    label= lambda y: f'    {y["JSON Name"]}',                   
                    marker=lambda x: f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: "Purple",
                    fill=lambda x: "Purple",
                ),
            )
        ],
        legend = False,
        table = True,
        table_pos="right",
        table_bb=(1.01,0,1,1), 
        table_col_widths = tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData
    )

def lookForTrends(dfs, dispNo, caseNo, dispTitle, keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="FLOP Per Cycle",
            y_label="Throughput",
            y_unit="FLOPs per cycle",
        )):
    tableData = dfs[(dispNo,caseNo)][["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J","CC OLs","Micro Runs","RLs","Reused SLs","L1 Usage","n","k"]
    defW = (1/(len(colLabels)*3)) # default width
    tableColWidths = [defW*0.6,defW*0.5,defW*0.4,defW,defW*1.25,defW,defW*1.5,defW*1.6,defW*0.5,defW*0.5]
    return Graph2D(
        imagePath=f'LFT-dispatch-{dispNo}-case-{caseNo}',
        keys=keys,
        title=dispTitle,
        scatterSets=[
            (
                dfs[(dispNo,caseNo)],
                CustomMarker(
                    y="FLOP Per Cycle",
                    label= lambda y: f'    {y["JSON Name"]}', 
                    marker=lambda x: f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: 'Black',
                    fill=lambda x: 'Black'
                ),
            )
        ],
        legend = False,
        table = True,
        table_pos="right",
        table_bb=(1.01,0,1,1),
        table_col_widths = tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData
    )

if __name__ == "__main__":
    main()