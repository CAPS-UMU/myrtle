from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graphing.graph_utils import graphEmAll, deriveMoreData2,addSVMPrediction,trimToTopX, Graph2D, Keys2D, CustomMarker, MySVM
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product, islice
from PIL import Image
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, SVR

predictedKernelTime="Predicted Kernel Time"

# python3 predictVsActual.py /home/hoppip/myrtle/accuracy /home/hoppip/myrtle/sensitivity-analysis/holistic-data
def main():
    args = sys.argv[1:]
    if len(args) != 2:
      print("USAGE: python3 deliverableGraphs.py  <predicted-actual> <predictionMode>")
      print("\twhere <predicted-actual> is the directory containing csv files")
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
    dispatchOrder = [1,7,8] # All dispatches will be graphed in the order 1, 7, 8
    title = lambda d, m, n, k: f"Dispatch {d}\nmatvec: <{m}x{k}>, <{n}x{k}> -> <{m}x{n}>"
    titles = {}
    for d in dispatchOrder:
        m,n,k = dispatcheSizes[d]
        titles[d] = title(d,m,n,k)
    print(titles)
    mode = args[1]
    #load dispatch data
    dfs = loadDFsDispatchCaseNo(args[0], dispatchOrder, caseNos, dispatcheSizes, mode)
    # derive more data about each dispatch
    deriveMoreData2(dfs,dispatchOrder,caseNos,mode)

    # finally stary graphing
    tup=dfs,dispatchOrder,caseNos,titles
    # generate graphs of actual dispatch times
    graphActualTime('graphing/out/ActualTime-3-dispatches-rank-x-axis.png',"rank","Rank","fastest to slowest",tup)
    graphActualTime('graphing/out/ActualTime-3-dispatches-L1-usage-x-axis.png',"Space Needed in L1","L1 Usage","bytes",tup)
    graphActualTime('graphing/out/ActualTime-3-dispatches-micro-runs-x-axis.png',"Microkernel Count","Microkernel Runs","microkernel count",tup)
    graphActualTime('graphing/out/ActualTime-3-dispatches-regular-loads-axis.png',"Regular Loads","Regular Loads","from scratchpad to register",tup)
    
    # generate graph of actual vs predicted dispatch times
    graphActualVsPredictedTime(mode,f'graphing/out/ActualVsPredTime-3-dispatches-{mode}.png',tup)
    
    print("HASTA LUEGO")

def graphActualTime(path,x,x_label,x_unit,tup):
    dfs,dispatchOrder,caseNos,titles=tup
    keysActual = Keys2D(
            x=x,
            x_label=x_label,
            x_unit=x_unit,
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","",keysActual,"",path)

def graphActualVsPredictedTime(mode,path,tup):
    dfs,dispatchOrder,caseNos,titles=tup
    keys = Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        )
    y2 = "stage"
    if mode == "sflt":
        evilFilterHack = True
        actualTimeDispatchCase(dfs,dispatchOrder,caseNos,titles,"Kernel Time","",keys,"",path,evilFilterHack)
        return
    if mode == "scyc":
        y2 = "Kernel Time Estimate"
    if mode == "svrcyc":
        y2 = predictedKernelTime   
    actualVsPredictedTimeDispatchCase(dfs, dispatchOrder, caseNos, titles,y1=keys.y,y2=y2,keys=keys,imgTitle="",imgName=path)
    


def actualTimeDispatchCase(dfs, dispatchNos, caseNos, titles,y1="Kernel Time",y2="Kernel Time Estimate",keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),imgTitle="Actual vs Predicted Time",imgName='ActualTime-3-dispatches.png',evilFilterHack=False):
    graphs = []
    for dispNo in dispatchNos:
        for caseNo in caseNos:
            title = titles[dispNo]    
            lftGraph = actualTime(dfs, dispNo, caseNo, title,y1,y2,keys,evilFilterHack)
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
        ),evilFilterHack=False):
    stroke = lambda x: 'Black'
    if evilFilterHack:
        colors = ['Pink','Red','Blue','Purple']
        colors = ['Lavender','Thistle','Plum','Purple']
        colors = ['#c8acde','#b28bd0','#9c6ac3','#8649b6']
        colors = ['#98c1f2','#6fa8ec','#468fe7','Purple']
        stroke = lambda x: colors[x["stage"]]
    # tableData = dfs[(dispNo,caseNo)][["rankAsStr","Microkernel Row Dim","Kernel Time","UnrollAndJam Outer Loops","Microkernel Count","Row Dim","Reduction Dim"]]
    # colLabels = ["rank","n'","Kernel Time","CC Outer Loops","Micro Runs","n","k"]
    tableData = dfs[(dispNo,caseNo)][["rankAsStr","Row Dim","Reduction Dim","Microkernel Row Dim",]]
    if evilFilterHack:
        df = dfs[(dispNo,caseNo)]
        tableData = df[df["rank"] < 5][["rankAsStr","Row Dim","Reduction Dim","Microkernel Row Dim",]]

    colLabels = ["rank","n","k","n'"]
    defW = (1/(len(colLabels)*3)) # default width
    tableColWidths = [defW,defW,defW*1.5,defW*1.5,defW*1.25,defW*0.5,defW*0.5]#[defW]*len(colLabels)
    return Graph2D(
        imagePath=f'graphing/out/actualTime-dispatch-{dispNo}-case-{1}',
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
                    stroke=stroke,
                    fill=lambda x: 'Black'
                ),
            ),
        ],
        legend = False,
        table = True,
        table_pos="right", # Bbox or [xmin, ymin, width, height]
        table_bb=(1.01,0,1,1), 
        table_col_widths = tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData
    )

def loadDFsDispatchCaseNo(path, dispatchNos, caseNos, inputSizes, mode):
    myPath = lambda d,mode,case: f"{path}/dispatch_{d}_case{case}_everything-myrtle-{mode}-ranking.csv"
    dfs = {}
    for d in dispatchNos:
        for c in caseNos:
            m,n,k = inputSizes[d]
            print(f'reading from {myPath(d,mode,c)}')
            pred = pd.read_csv(myPath(d,mode,c))
            if mode == "svrcyc":
                # print(pred["JSON Name"])
                # print(pred[pred.columns[len(pred.columns)-1]])
                # print(pred["Predicted Kernel Time"])
                predictedKernelTime=pred.columns[len(pred.columns)-1]
            dfs[(d, c)] = pred
    return dfs

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
    orig = Image.open('graphing/context2.png') # hard coded
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

def actualVsPredictedTimeDispatchCase(dfs, dispatchNos, caseNos, titles,y1="actualColName",y2="predColName",keys=Keys2D(
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
    

def actualVsPredictedTime(dfs, dispNo, caseNo, title, y1="actualColName",y2="predColName",keys=Keys2D(
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
        imagePath=f'graphing/out/AcutualVsEst-dispatch-{dispNo}-case-{1}',
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
        table_bb=(1.01,0,1,1), # Bbox or [xmin, ymin, width, height]
        table_col_widths = tableColWidths,
        table_col_labels=colLabels,
        table_row_labels=[],
        table_data=tableData
    )


if __name__ == "__main__":
    main()