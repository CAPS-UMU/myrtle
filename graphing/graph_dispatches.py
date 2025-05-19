from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, getGraphXvsYrankedbyZtopQ

def shortcutToData():
    path = (
        lambda x, y: f"./streamingLoadDetail/dispatch_{x}_case{y}_everything.csv"
        #lambda x, y: f"./toGraph/dispatch_{x}_case{y}_everything.csv"
    )
    dispatches = [1, 7, 8]
    cases = [1, 2]
    dfs = {}
    for d in dispatches:
        for c in cases:
            df = pd.read_csv(path(d, c))
            df["Actual Streaming Loads"] = df["Other Streaming Loads"] + df["Start Reuse Streaming Loads"]
            dfs[(d, c)] = df
    return dfs

def main():
    dfs = shortcutToData()
    titles = ["Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>","Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>","Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>"]
    graphs = []
    for dispNo, dispTitle in zip([1,8,7], titles):
        g = getGraphXvsYrankedbyZtopQ(dfs,"Actual Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle,5)
        g.legend_pos="upper left"
        g.legend_bb=(0.5,1)
        graphs.append(g)
    for dispNo, dispTitle in zip([1,8,7], titles):
        g = getGraphXvsYrankedbyZtopQ(dfs,"Reused Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle,5)
        g.legend_pos="upper left"
        g.legend_bb=(0.5,1)
        graphs.append(g)
    graphEmAll((2,3),graphs)

    dispNo =1
    dispTitle = titles[0]
    justOne = getGraphXvsYrankedbyZtopQ(dfs,"Reused Streaming Loads","loads","Regular Loads","loads", "Kernel Time",True,dispNo,dispTitle,10)
    
    #graphEmAll((1,1),[justOne])
    

if __name__ == "__main__":
    main()

