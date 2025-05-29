from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import pandas as pd
from graph_utils import graphEmAll, getBestXFrom, shortcutToData, rankBy, Graph2D, Keys2D, CustomMarker
import matplotlib as mpl
import matplotlib.pyplot as plt

def rowDimVsQuidditchTime(dfs, dispNo, dispTitle):
    # ranked = rankBy(dfs, (dispNo, 1), "Kernel Time Estimate", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    b = Graph2D(
        keys=Keys2D(
            x="Row Dim",
            x_label="Row Dim",
            x_unit="elements",
            y="Kernel Time Estimate",
            y_label="Kernel Time Estimate",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                dfs[(dispNo, 1)],
                CustomMarker(
                    label= lambda y: f'({y["Row Dim"]},{y["Reduction Dim"]})',
                    marker=lambda x: 'o',#f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2), #* 2,
                    stroke=lambda x: 'Black'#"Purple",
                ),
            )
        ],
        legend=True,
        legend_pos="upper left",
        legend_bb=(1,1)
    )
    return b

def unrollVsQuidditchTime(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    b = Graph2D(
        keys=Keys2D(
            x='UnrollAndJam Outer Loops',
            x_label='UnrollAndJam Outer Loops',
            x_unit="elements",
            y="Kernel Time Estimate",
            y_label="Kernel Time Estimate",
            y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                ranked,#dfs[(dispNo, 1)],
                CustomMarker(
                    label= lambda y: f'({y["Row Dim"]},{y["Reduction Dim"]})',
                    marker=lambda x: f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2), #* 2,
                    stroke=lambda x: 'Black'#"Purple",
                ),
            )
        ],
        legend=True,
        legend_pos="upper left",
        legend_bb=(1,1)
    )
    return b

def unrollVsUnroll(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    b = Graph2D(
        keys=Keys2D(
            x="Microkernel Row Dim",
            x_label="Microkernel Row Dim",
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
                    marker=lambda x: f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)**2,
                    stroke=lambda x: 'Black',#"Purple",
                    fill=lambda x: 'Black'#"Purple",
                ),
            )
        ],
        legend=True,
        legend_pos="upper left",
        legend_bb=(1,1)
    )
    return b

def unrollVsUnrollTable(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    b = Graph2D(
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
        table_pos="upper left",
        table_bb=(1,1),
        table_col_labels=["rank","CC Row Dim","U&J Factor","CC Outer Loops","L1 Row Dim","L1 Col Dim"],
        table_row_labels=[],
        table_data=ranked[["rank","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Row Dim","Reduction Dim"]]
    )
    return b

def unrollVsUnrollLegend(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    def getInfo(x):
        return f'{x["Microkernel Row Dim"]}{x["UnrollAndJam Factor"]}{x["UnrollAndJam Outer Loops"]}'
    b = Graph2D(
        keys=Keys2D(
            x="Reduction Dim",
            x_label="Reduction Dim",
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
                    #label= lambda y: f'    {y["JSON Name"]}', 
                    label= lambda y: f'{y["Microkernel Row Dim"]},     {y["UnrollAndJam Factor"]},     {y["UnrollAndJam Outer Loops"]}',                   
                    marker=lambda x: f'${x["rank"]}$',#f'$({x["UnrollAndJam Factor"]},{int(x["UnrollAndJam Outer Loops"])})$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2)*2,
                    stroke=lambda x: 'Black',#"Purple",
                    fill=lambda x: 'Black'#"Purple",
                ),
            )
        ],
        legend=True,
        legend_pos="upper left",
        legend_bb=(1,1),
   
        legend_title="         n'   U&J    OuterLoops",
      #  legend_pos='upper right',
        # legend_bb=(0,2),
        # table = False,
        # table_bb=(1,1),
        # table_col_labels=["rank","CC Row Dim","U&J Factor","CC Outer Loops","L1 Row Dim","L1 Col Dim"],
        # table_row_labels=[],
        # table_data=ranked[["rank","Microkernel Row Dim","UnrollAndJam Factor","UnrollAndJam Outer Loops","Row Dim","Reduction Dim"]]
    )
    return b

# important fields
# bbox, loc,rowLabels,colLabels, cellText

# matplotlib.pyplot.table(
#     cellText=None, 
#     cellColours=None, 
#     cellLoc='right', 
#     colWidths=None, 
#     rowLabels=None, 
#     rowColours=None, 
#     rowLoc='left', 
#     colLabels=None, 
#     colColours=None, 
#     colLoc='center', 
#     loc='bottom', 
#     bbox=None, 
#     edges='closed', **kwargs


def microKernelRowDimVsQuidditchTime(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    
    # mpl.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
    #                            'Lucida Grande', 'Verdana']

    b = Graph2D(
        keys=Keys2D(
            x="Microkernel Row Dim",
            x_label="Microkernel Row Dim",
            x_unit="elements",
            y="UnrollAndJam Outer Loops",
            y_label="OuterLoops Due to Unroll and Jam",
            y_unit="regular riscv loops",
            # y="Kernel Time",
            # y_label="Kernel Time",
            # y_unit="cycles",
        ),
        title=dispTitle,
        scatterSets=[
            (
                ranked,#dfs[(dispNo, 1)],
                CustomMarker(
                    label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',
                    marker=lambda x: f'${x["rank"]}$',#f'${x["UnrollAndJam Factor"]},{x["UnrollAndJam Outer Loops"]}$',#'o',#f'${x["rank"]}$',
                    size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2) * 2,
                    stroke=lambda x: 'Black'#"Purple",
                ),
            )
        ],
        legend=True,
        legend_title = "estimate timing",
        legend_pos="upper right",
        legend_bb=(0.5,0.5)
    )
    return b

def microKernelRowDimVsVsUnrollVsQuidditchTime(dfs, dispNo, dispTitle):
    ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    # best = getBestXFrom(ranked, "Kernel Time Estimate", 10, True)
    
    # mpl.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
    #                            'Lucida Grande', 'Verdana']

    # b = Graph2D(
    #     keys=Keys2D(
    #         x="Microkernel Row Dim",
    #         x_label="Microkernel Row Dim",
    #         x_unit="elements",
    #         y="UnrollAndJam Outer Loops",
    #         y_label="OuterLoops Due to Unroll and Jam",
    #         y_unit="regular riscv loops",
    #         # y="Kernel Time",
    #         # y_label="Kernel Time",
    #         # y_unit="cycles",
    #     ),
    #     title=dispTitle,
    #     scatterSets=[
    #         (
    #             ranked,#dfs[(dispNo, 1)],
    #             CustomMarker(
    #                 label= lambda y: f'{y["JSON Name"]} ({y["Microkernel Row Dim"]}, UaJ = {y["UnrollAndJam Factor"]}, {y["UnrollAndJam Outer Loops"]} outer loops)',
    #                 marker=lambda x: f'${x["rank"]}$',#f'${x["UnrollAndJam Factor"]},{x["UnrollAndJam Outer Loops"]}$',#'o',#f'${x["rank"]}$',
    #                 size=lambda y=0: (mpl.rcParams["lines.markersize"] ** 2) * 2,
    #                 stroke=lambda x: 'Black'#"Purple",
    #             ),
    #         )
    #     ],
    #     legend=True,
    #     legend_pos="upper left",
    #     legend_bb=(1,1)
    # )
    # return b
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for index, row in ranked.iterrows():
            ax.scatter(row["Microkernel Row Dim"], row["UnrollAndJam Outer Loops"], row["Kernel Time"],marker=f'${row["UnrollAndJam Outer Loops"]}$',c="YellowGreen",s=(mpl.rcParams["lines.markersize"] ** 2) * 2,edgecolors="Black")
    ax.set_xlabel("Microkernel Row Dim")
    ax.set_ylabel("UnrollAndJam Outer Loops")
    ax.set_zlabel("Cycles")
    plt.show()

# 3D reference:
# fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for index, row in df.iterrows():
    #         ax.scatter(row["CC Row Dim"], row["CC Reduction Dim"], row["linalg_xdsl"],marker="o",c="YellowGreen",edgecolors="Black")
    # ax.set_xlabel("Row Dim")
    # ax.set_ylabel("Reduction Dim")
    # ax.set_zlabel("Cycles")
    #plt.show()

def main():
    titles = [
        "Dispatch 1\nmatvec: <1x400>, <1200x400> -> <1x1200>",
        "Dispatch 8\nmatvec: <1x600>, <600x600> -> <1x600>",
        "Dispatch 7\nmatvec: <1x400>, <600x400> -> <1x600>",
    ]
    graphs = []
    # dfs = shortcutToData("../estimated_cycles_out")
    dfs = shortcutToData("../estimated_cycles_out_2")
    top10 = {}
    for dispNo, dispTitle in zip([1, 8, 7], titles):
        leg = unrollVsUnrollLegend(dfs, dispNo, dispTitle)
        quid = unrollVsQuidditchTime(dfs, dispNo, dispTitle)
        est = rowDimVsQuidditchTime(dfs, dispNo, dispTitle)
        combo = microKernelRowDimVsQuidditchTime(dfs, dispNo, dispTitle)
        unrolled = unrollVsUnrollTable(dfs, dispNo, dispTitle)
        graphs.append(leg)
        graphs.append(quid)
        graphs.append(est)
        graphs.append(combo)
        top10[dispNo] = (leg,quid,est,combo,unrolled)
    #graphEmAll((3, 2), graphs)

    g = top10[7]
    for i in [1]:#[1, 8, 7]:
        graphEmAll((1,1), [top10[i][0]])

    #microKernelRowDimVsVsUnrollVsQuidditchTime(dfs,7,titles[2])

    # graphEmAll((1,2), (g[0],g[1]))
    # data = g[0].scatterSets[0][0]
    # print(data)
    # top5=getBestXFrom(data, "Kernel Time", 5, True)
    # print(top5)
    # print("hooodle")
    # for i in range(0,5):
    #     better = top5.iloc[i]["Kernel Time"]
    #     print(top5.iloc[i]["JSON Name"],end=" compared to\n")
    #     for j in range(i+1,5):  
    #         worse = top5.iloc[j]["Kernel Time"]
    #         percentage = ((worse + 0.0) - (better + 0.0) )/ (worse + 0.0)   * 100.0   
    #         print("\t",end=f'{top5.iloc[j]["JSON Name"]} is {percentage:.2f} % better\n')


if __name__ == "__main__":
    main()