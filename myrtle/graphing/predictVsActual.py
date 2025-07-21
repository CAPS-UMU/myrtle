from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, loadDFsDispatchCaseNo, deriveMoreData2,addSVMPrediction,trimToTopX, Graph2D, Keys2D, CustomMarker, MySVM
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product, islice
from PIL import Image
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, SVR

#
# # feature_file_df['RESULT'] = RESULT_df['RESULT'].to_numpy()
#    predictedPath = lambda m,n,k,mode,case: f"{mode}/{m}x{n}x{k}wm-n-k_case{case}_searchSpace_selection_{mode}.csv"
#     actualPath = lambda m,n,k: f"{m}x{n}x{k}wm-n-k-timed"

def actualTimeDispatchCase(dfs, dispatchNos, caseNos, titles,y1="Kernel Time",y2="Kernel Time Estimate",keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),imgTitle="Actual vs Predicted Time",imgName='ActualTime-3-dispatches.png'):
    graphs = []
    for dispNo in dispatchNos:
        for caseNo in caseNos:
            title = titles[dispNo]    
            lftGraph = actualTime(dfs, dispNo, caseNo, title,y1,y2,keys)
            graphs.append(lftGraph)
            graphEmAll((1, 1), [lftGraph])
    combineDispatchesWithContext(graphs,imgTitle,imgName)
 

def actualTime(dfs, dispNo, caseNo, title, y1="Kernel Time",y2="Kernel Time Estimate",keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )):
    tableData = dfs[(dispNo,caseNo)][["rankAsStr","Microkernel Row Dim","Kernel Time","UnrollAndJam Outer Loops","Microkernel Count","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","Kernel Time","CC Outer Loops","Micro Runs","n","k"]
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
            # (
            #     dfs[(dispNo,caseNo)],
            #     CustomMarker(
            #         y=y2,
            #         label= lambda y: f'    {y["JSON Name"]}',                   
            #         marker=lambda x: f'${x["rank"]}$',
            #         size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
            #         stroke=lambda x: "Purple",
            #         fill=lambda x: "Purple",
            #     ),
            # )
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

def loadDFsDispatchCaseNo(predicted, actual, dispatchNos, caseNos, inputSizes, mode):
    predictedPath = lambda m,n,k,mode,case: f"{predicted}/{mode}/{m}x{n}x{k}wm-n-k_case{case}_searchSpace_selection_{mode}.csv"
    actualPath = lambda m,n,k: f"{actual}/{m}x{n}x{k}wm-n-k-timed.csv"
    dfs = {}
    for d in dispatchNos:
        for c in caseNos:
            m,n,k = inputSizes[d]
            pred = pd.read_csv(predictedPath(m,n,k,mode,c))
            act = pd.read_csv(actualPath(m,n,k))
           # mergedCSV = pd.merge(firstCSV, secondCSV,on=sys.argv[3],how="inner")
            pred = pd.merge(pred, act[["JSON Name","Kernel Time"]],on="JSON Name",how="inner")
            pred = pd.merge(pred, act[["JSON Name","Kernel Name"]],on="JSON Name",how="inner")
            # pred['Kernel Time'] = act['Kernel Time'].to_numpy()
            # pred['Kernel Name'] = act['Kernel Name'].to_numpy()
            # pred['Kernel Time'] = act['Kernel Time'].to_numpy()
            # pred['Kernel Name'] = act['Kernel Name'].to_numpy()
            #Kernel Name
            dfs[(d, c)] = pred
    return dfs

# example run: python graphing-refactored.py "/home/hoppip/myrtle/estimated_cycles_no_overhead"
# example run: python graphing-refactored.py /home/emily/myrtle/estimated_cycles_overhead
#/home/hoppip/myrtle/accuracy
#

# svrcyc/1x161x600wm-n-k_case1_searchSpace_selection_svrcyc.csv
# 1x400x161wm-n-k-timed.csv
# /home/hoppip/myrtle/accuracy
# /home/hoppip/myrtle/sensitivity-analysis/holistic-data

# python3 predictVsActual.py /home/hoppip/myrtle/accuracy /home/hoppip/myrtle/sensitivity-analysis/holistic-data
def main():
    args = sys.argv[1:]
    if len(args) != 3:
      print("USAGE: python3 predictVsActual.py  <predicted> <actual> <predictionMode>")
      print("\twhere <predicted> <actual> are directories containing csv files")
      print("\tand <predictionMode> is either \"svrcyc\", \"ssyc\", or \"sflt\"")
      exit(1)
    print("HOLA")
    dispatcheSizes = {
        #0:(1,400,161),
        1:(1,1200,400),
        7:(1,600,400),
        8:(1,600,600),
       # 9:(1,161,600),
    }
    caseNos = [1] # We only graph case 1; no padding anywhere.
    dispatchOrder = [1,7,8] # All dispatches will be graphed in the order 0, 1, 7, 8, 9
    predictedPath = lambda m,n,k,mode,case: f"{mode}/{m}x{n}x{k}wm-n-k_case{case}_searchSpace_selection_{mode}.csv"
    actualPath = lambda m,n,k: f"{m}x{n}x{k}wm-n-k-timed"
    title = lambda d, m, n, k: f"Dispatch {d}\nmatvec: <{m}x{k}>, <{n}x{k}> -> <{m}x{n}>"
    titles = {}
    for d in dispatchOrder:
        m,n,k = dispatcheSizes[d]
        titles[d] = title(d,m,n,k)
    print(titles)
   # print(dfs)
    print(f"Reading data from {args[0]} and {args[1]}...")
    mode = args[2]
    #load dispatch data
    dfs = loadDFsDispatchCaseNo(args[0], args[1], dispatchOrder, caseNos, dispatcheSizes, mode)
    # derive more data about each dispatch
    deriveMoreData2(dfs,dispatchOrder,caseNos,mode)
    keysActual = Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","Predicted Kernel Time",keysActual,"Estimate with Microkernel Time * Microkernel Runs",'ActualTime-3-dispatches-rank-x-axis.png')
    keysActual = Keys2D(
            x="Space Needed in L1",
            x_label="L1 Usage",
            x_unit="bytes",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","Predicted Kernel Time",keysActual,"Estimate with Microkernel Time * Microkernel Runs",'ActualTime-3-dispatches-L1-usage-x-axis.png')
    keysActual = Keys2D(
            x="Row Dim",
            x_label="Row Dim",
            x_unit="elements",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","Predicted Kernel Time",keysActual,"Estimate with Microkernel Time * Microkernel Runs",'ActualTime-3-dispatches-row-dim-x-axis.png')
    keysActual = Keys2D(
            x="Reduction Dim",
            x_label="Reduction Dim",
            x_unit="elements",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","Predicted Kernel Time",keysActual,"Estimate with Microkernel Time * Microkernel Runs",'ActualTime-3-dispatches-reduction-dim-x-axis.png')
    keysActual = Keys2D(
            x="Microkernel Count",
            x_label="Microkernel Runs",
            x_unit="microkernel count",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","Predicted Kernel Time",keysActual,"Estimate with Microkernel Time * Microkernel Runs",'ActualTime-3-dispatches-micro-runs-x-axis.png')
 
    # graph TOP TEN data points
    # dfs = trimToTopX(dfs, dispatchOrder, caseNos, "rank", 10) # destructive!!
    # lookForTrendsDispatchCase(dfs,dispatchOrder,caseNos,titles,keysFLOPs,"hoodle","LFT-dispatches-kernel-time-x-axis-top-10.png")
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

def lookForTrendsDispatchCase(dfs, dispatchNos, caseNos, titles, keys, imgTitle="Looking for Trends",imgName='LFT-3-dispatches.png',):
    graphs = []
    for dispNo in dispatchNos:
        for caseNo in caseNos:
            title = titles[dispNo]    
            lftGraph = lookForTrends(dfs, dispNo, caseNo, title, keys)
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

def lookForTrends(dfs, dispNo, caseNo, dispTitle, keys):
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