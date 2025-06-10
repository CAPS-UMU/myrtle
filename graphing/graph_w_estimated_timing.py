from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, getBestXFrom, shortcutToData, rankBy, Graph2D, Keys2D, CustomMarker, generalGraph
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product, islice
from PIL import Image

# def graphEmAllImages(shape: tuple, graphs, context):
#     if shape[0] * shape[1] != len(graphs):
#         raise Exception("area of shape and graph count must be equal!")
#     fig = plt.figure()
#     image = plt.imread(context)
#     ax = plt.subplot2grid((shape[0]+1,shape[1]), (0,0),colspan=2)
#     ax.imshow(image)
#     ax.axis('off')
#     lhs = tuple(range(1,shape[0]+1))
#     rhs = tuple(range(0,shape[1]))
#     print(lhs)
#     print(rhs)
#     print(list(product(*[lhs,rhs])))
#     for tup, g in zip(list(product(*[lhs,rhs])), graphs):
#         print(f"{tup} {g}")
#         image = plt.imread(g)
#         ax = plt.subplot2grid((shape[0]+1,shape[1]),tup)
#         ax.imshow(image)
#         ax.axis('off')
#         #1187 × 265
#     print(fig.get_dpi())
#     plt.savefig("toad.png", bbox_inches='tight')
#     plt.show()

def unrollVsUnrollTable(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    ranked["rankAsStr"] = ranked.apply(lambda y: f'{y["rank"]}', axis=1)
    tableData = ranked[["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J Factor","CC Outer Loops","Micro Runs","n","k"]
    defW = (1/(len(colLabels)*3)) # default width
    print(defW)
    tableColWidths = [defW,defW,defW*1.5,defW*1.5,defW*1.25,defW,defW]#[defW]*len(colLabels)
    # [1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    #[1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    b = Graph2D(
        imagePath=f'dispatch-{dispNo}-case-{1}',
        keys=Keys2D(
            x="rank",
            x_label="Rank",
            x_unit="fastest to slowest",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                ranked,#dfs[(dispNo, 1)],
                CustomMarker(
                    y="Kernel Time",
                    label= lambda y: f'    {y["JSON Name"]}', 
                    # label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',                   
                    marker=lambda x: f'${x["rank"]}$',#f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: 'Black',#"Purple",
                    fill=lambda x: 'Black'#"Purple",
                ),
            ),
            (
                ranked,#dfs[(dispNo, 1)],
                CustomMarker(
                    y="Kernel Time Estimate",
                    label= lambda y: f'    {y["JSON Name"]}', 
                    # label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',                   
                    marker=lambda x: f'${x["rank"]}$',#f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: "Purple",
                    fill=lambda x: "Purple",
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
    return b

#  df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
# for index, row in data.iterrows():
#print(list(islice(it, 3)))

def printFiltering(df,by,num,lowIsGood):
    df = df.sort_values(by, ascending=lowIsGood)
    print("\t",end='')
    print(f'by {by}:',)
    for row in islice(df.iterrows(),num):
        print("\t\t",end='')
        print(f'{row[1]["Row Dim"]}x{row[1]["Reduction Dim"]}',end='')
        print("\t",end='')
        print(f'{row[1][by]}',end='')
        print("\t",end='')
        print(f'({row[1]["rankAsStr"]})')
    return df.head(num)

def naiveFiltering(df, dispatchNo, caseNo):
    tableData = df[["rankAsStr","Kernel Time","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]
    print(f'Naive Filtering of Dispatch {dispatchNo} case {caseNo}:')
    #tableData.sort_values("UnrollAndJam Factor", ascending=True)
    #unroll = printFiltering(tableData,"UnrollAndJam Factor",5,True)
    #byTime = printFiltering(tableData,"Kernel Time",5,True)
    # = printFiltering(tableData,"Space Needed in L1",5,False)
    byRuns = printFiltering(tableData,"Microkernel Count",4,True)
    byRegularLoads = printFiltering(byRuns,"Regular Loads",3,True)
    byStreamingLoads = printFiltering(byRegularLoads,"Reused Streaming Loads",2,False)
  

# Regular Loads,
# Total Streaming Loads,
# Other Streaming Loads,
# Start Reuse Streaming Loads,
# Reused Streaming Loads,
def lookAtLoads(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    ranked["rankAsStr"] = ranked.apply(lambda y: f'{y["rank"]}', axis=1)
    naiveFiltering(ranked,dispNo,1)
    ranked["Config Overhead"] = ranked.apply(lambda y: y["UnrollAndJam Outer Loops"] * y["Microkernel Count"], axis=1)
    tableData = ranked[["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J","CC OLs","Micro Runs","RLs","Reused SLs","L1 Usage","n","k"]
    defW = (1/(len(colLabels)*3)) # default width
    #print(defW)
    tableColWidths = [defW*0.6,defW*0.5,defW*0.4,defW,defW*1.25,defW,defW*1.5,defW*1.6,defW*0.5,defW*0.5]#[defW]*len(colLabels)
    # [1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    #[1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    b = Graph2D(
        imagePath=f'dispatch-{dispNo}-case-{1}',
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
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                ranked,#dfs[(dispNo, 1)],
                CustomMarker(
                    y="Kernel Time",
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
    return b

def main():
    titles = [
        "Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        "Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        "Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
    ]
    graphs = []
    # dfs = shortcutToData("../estimated_cycles_out")
    dfs = shortcutToData("../estimated_cycles_out_2")

    for dispNo, dispTitle in zip([1, 8, 7], titles):     
        # unrolled = unrollVsUnrollTable(dfs, dispNo, dispTitle)
        unrolled = lookAtLoads(dfs, dispNo, dispTitle)
        graphs.append(unrolled)
    #graphEmAll((3, 2), graphs)
    # print(type(plt.Figure.get_dpi()))
     # graphEmAllImages((1, 1), ["frog.png"],"context.png")
    #graphWContext((1, 1), [graphs[0]],"context.png")
    #graphWContext((2,1), [top[1],top[8]],"context.png")
   #1187 × 265
    for g in graphs:
        graphEmAll((1, 1), [g])
        top = Image.open('context2.png') # hard coded
        bot = Image.open(f'{g.imagePath}.png')    
        resized = top.copy()
        resized.thumbnail(bot.size, Image.Resampling.LANCZOS)
        resized.save("resized.png")
        top = Image.open('resized.png')
        canvas = Image.new('RGBA', (bot.size[0], bot.size[1]+top.size[1]), (0, 0, 0, 0))
        canvas.paste(resized, (0, 0), resized)
        canvas.paste(bot, (0, top.size[1]), bot)
        canvas.save('Image.png')
    # put all the graphs in one image
    top = Image.open('resized.png')
    g0 = Image.open(f'{graphs[0].imagePath}.png')  
    g1 = Image.open(f'{graphs[1].imagePath}.png')  
    g2 = Image.open(f'{graphs[2].imagePath}.png')  
    canvas = Image.new('RGBA', (g0.size[0], top.size[1]+g0.size[1]+g1.size[1]+g2.size[1]), (0, 0, 0, 0))
    canvas.paste(top, (0, 0), top)
    canvas.paste(g0, (0, top.size[1]), g0)
    canvas.paste(g1, (0, top.size[1]+g0.size[1]), g1)
    canvas.paste(g2, (0, top.size[1]+g0.size[1]+g1.size[1]), g2)
    canvas.save('3-dispatches.png')

  


if __name__ == "__main__":
    main()