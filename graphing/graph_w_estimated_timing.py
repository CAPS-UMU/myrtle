from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, getBestXFrom, shortcutToData, rankBy, Graph2D, Keys2D, CustomMarker, generalGraph
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
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
    tableData = ranked[["rankAsStr","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Row Dim","Reduction Dim"]]
    colLabels = ["rank","n'","U&J Factor","CC Outer Loops","n","k = k'"]
    defW = (1/(len(colLabels)*3)) # default width
    print(defW)
    tableColWidths = [defW,defW,defW*2,defW*2,defW,defW]#[defW]*len(colLabels)
    # [1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    #[1/(len(g.table_col_labels))*0.25]*len(g.table_col_labels)
    b = Graph2D(
        imagePath=f'dispatch-{dispNo}-case-{1}',
        keys=Keys2D(
            x="Row Dim",
            x_label="L1 Row Dim",
            x_unit="elements",
            y="Kernel Time",
            y_label="Kernel Time",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                ranked,#dfs[(dispNo, 1)],
                CustomMarker(
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
        unrolled = unrollVsUnrollTable(dfs, dispNo, dispTitle)
        graphs.append(unrolled)
    #graphEmAll((3, 2), graphs)
    # print(type(plt.Figure.get_dpi()))
    graphEmAll((1, 1), [graphs[0]])
    # graphEmAllImages((1, 1), ["frog.png"],"context.png")
    #graphWContext((1, 1), [graphs[0]],"context.png")
    #graphWContext((2,1), [top[1],top[8]],"context.png")
   #1187 × 265
    top = Image.open('context.png') # hard coded
    bot = Image.open(f'{graphs[0].imagePath}.png')    
    resized = top.copy()
    resized.thumbnail(bot.size, Image.Resampling.LANCZOS)
    resized.save("resized.png")
    top = Image.open('resized.png')
    canvas = Image.new('RGBA', (bot.size[0], bot.size[1]+top.size[1]), (0, 0, 0, 0))
    canvas.paste(resized, (0, 0), resized)
    canvas.paste(bot, (0, top.size[1]), bot)
    canvas.save('Image.png')
  


if __name__ == "__main__":
    main()