from graphing.graph_utils import Graph2D, Keys2D, CustomMarker
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rankBy(df, by, lowIsGood):
        df_sorted = df.sort_values(by=by, ascending=lowIsGood)
        df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
        df_sorted["rankAsStr"] = df_sorted.apply(lambda y: f'{y["rank"]}', axis=1)
        return df_sorted

def graphEmAll(shape: tuple, graphs, x_inch,y_inch):
    if shape[0] * shape[1] != len(graphs):
        raise Exception("area of shape and graph count must be equal!")
    fig = plt.figure()
    #fig.set_size_inches(4, 2) # for the deliverable graphs
    #fig.set_size_inches(1098/72.0,476/72.0) # 15.25 x 6.61
    fig.set_size_inches(x_inch,y_inch)
    for i in range(0, len(graphs)):
        ax = fig.add_subplot(shape[0], shape[1], i + 1)
        generalGraph(ax, graphs[i])
        plt.savefig(f"{graphs[i].imagePath}", bbox_inches='tight')

def graphWPatch(graph, x_inch, y_inch, patch_func):
    fig = plt.figure()
    #fig.set_size_inches(4, 2) # for the deliverable graphs
    #fig.set_size_inches(1098/72.0,476/72.0) # 15.25 x 6.61
    table_bb = graph.table_bb
    table_bb_height = table_bb[3]
    print(table_bb)
    print(table_bb_height)
    fig.set_size_inches(x_inch,y_inch)
    print(f'I think the table_bb_width in inches is {table_bb[2]*x_inch}')
    print(f'I think the table_bb_height in inches is {table_bb_height*y_inch}')
    ax = fig.add_subplot(1,1, 1)
    ax = generalGraph(ax, graph)
    patch_func(ax)
    plt.savefig(f"{graph.imagePath}", bbox_inches='tight')
    return ax
    

def generalGraph(ax, g: Graph2D):
    ax.set_title(g.title)
    for data, cm in g.scatterSets:
        for index, row in data.iterrows():
            ax.scatter(
                row[g.keys.x],
                row[cm.y],
                c=cm.fill(row),
                edgecolors=cm.stroke(row),
                s=cm.size(row),
                label=cm.label(row),
                marker=cm.marker(row),
            )
    ax.set_xlabel(f"{g.keys.x_label} ({g.keys.x_unit})")
    ax.set_ylabel(f"{g.keys.y_label} ({g.keys.y_unit})")
    if len(g.curves):
        labels = []
        lines = []
        for curve in g.curves:
            line = ax.plot(
                curve.data,
                curve.func(curve.data),
                label=curve.label,
                color=curve.color,
                linestyle="-",
                linewidth=2.0
            )
    if g.legend:
        leg = ax.legend(loc=g.legend_pos, bbox_to_anchor=g.legend_bb,title=g.legend_title)
        leg._legend_box.align = "left"      
    if g.table:#bbox, loc,rowLabels,colLabels, cellText#colWidths
        t=plt.table(loc='right',bbox=g.table_bb,colLabels=g.table_col_labels,colWidths=g.table_col_widths,cellText=g.table_data.values.tolist())
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        ax.add_table(t)
    return ax
