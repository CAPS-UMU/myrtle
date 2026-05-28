import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def saveFigsInHTML(special_figs, more_figs, titleOfWebpage):
    # --- Convert each figure to HTML div ---
    special_divs = []
    for f in special_figs:
        special_divs.append(pio.to_html(f, include_plotlyjs="cdn", full_html=False))
    moreDivs = []
    for f in more_figs:
        moreDivs.append(pio.to_html(f, include_plotlyjs="cdn", full_html=False))
    # --- Concatenate divs into single string ---
    specialDivsAsHTML = ""
    for d in special_divs:
        specialDivsAsHTML = specialDivsAsHTML + f"""<div class="plot-box">{d}</div>"""
    moreDivsAsHTML = ""
    for d in moreDivs:
        moreDivsAsHTML = moreDivsAsHTML + f"""<div class="plot-box">{d}</div>"""
    # --- Combine into HTML page with grid layout ---
    html = f"""
    <html>
    <head>
    <title>Graphing Tiling Schemes</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
        font-family: Arial, sans-serif;
        margin: 30px;
        background-color: #f7f7f7;
        }}
        .dashboard {{
        display: flex;
        flex-direction: column;
        gap: 30px; /* space between charts */
        }}
        .plot-box {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
    </head>
    <body>
    <h1 style="text-align:center;">{titleOfWebpage}</h1>
    <a href="index.html" >Back to Landing Page</a>
    <div class="dashboard">
      {specialDivsAsHTML} 
        {"<div>More Experiments</div>"}
        {moreDivsAsHTML}        
        </div>
        </body>
        </html>
        """
    return html


def scatterWithColor(df, x_col, y_col, color, hover_data, title, maerker=""):
    fig8 = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b>",
    )
    return fig8

def scatterWithColorSymbol(df, x_col, y_col, color, hover_data, title, marker=""):
    fig8 = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        symbol=marker,
        symbol_sequence=["circle", "triangle-up", "triangle-up"],
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b>",
    )
    return fig8


def hoverTemplateString(df, hover_data, title):
    idx = {col: i for i, col in enumerate(df[hover_data].columns.values)}
    str = f"{title} "
    for k in hover_data:
        str = str + "<br>" + f"{k}" + ": %{" + "customdata" + f"[{idx[k]}]" + "}"
    str = str + "<extra></extra>"
    return str


def addScatterFlatColorMarker(fig, df, x_col, y_col, color, marker, hover_data, title):
    customData = df[hover_data].to_numpy()
    fig.add_scatter(
        x=df[x_col],
        y=df[y_col],
        mode="markers",
        marker=dict(color=color, symbol=marker),
        showlegend=False,
        name=title,
        customdata=customData,
        hovertemplate=(hoverTemplateString(df, hover_data, title)),
    )
    # mark
def ssr_prune(df):
    unique_ssr_configs = list(
        set(df["SSR Configs"].values.tolist())
    )  # remove duplicates
    unique_ssr_configs.sort()  # sort least to greateset
    prunePoint = unique_ssr_configs[1]  # prune to two smallest groups of ssr_configs
    # why did I prune to under 24576 for 384 cube???
   # print(f'4 smallest SSR config values: {unique_ssr_configs[0]} {unique_ssr_configs[1]} {unique_ssr_configs[2]}')
   # print(unique_ssr_configs)
    return prunePoint

def prunedScatter(df, x_col, y_col, color, hover_data, title, prunePoint, marker=""):
    df_mod = df
    if marker == "":
        df["symbolMarker"] = "O"
    fig14 = px.scatter(
        df_mod,
        x=x_col,
        y=y_col,
        symbol="symbolMarker",
        symbol_sequence=["circle", "triangle-up", "triangle-up"],
        color=color,  # "regPerStream",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b> with SSR Configs > {prunePoint} separated with vertical dotted line.",
    )
    fig14.add_vline(x=prunePoint, line_width=2, line_dash="dash", line_color="green")
    # turn the pruned points gray?
    return fig14


def generateExperimentalPruningGraphs(timed, analyzed, titleOfWebpage):
    timed["Total CC Tiles"] = timed["SSR Config Count"]
    timed["FMADDsMULsPerCore"] = timed["FMADDsMULs"] / timed["Total CC Tiles"]
    # timed["Raw Compute / Overlap Stall"] = timed["Raw Compute Time Total"] / timed["Overlap Stall Time Total"]
    # timed["(Raw Compute / Overlap Stall) Per Core"] = timed["Raw Compute Time Total"] / timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
    # timed["Overlap Stall Time Per Core"] = timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
    # timed["Y/X"]=timed["Avg n'_sz / k_size"] * timed["Avg A'"]
    timed["mRem"] = timed["M"] % timed["m"]
    timed["niceMRem"] = timed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
    x_col = "SSR Config Count"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        "absoluteRank",
        "dma",
        "HW Loops",
        "SSR Configs",
        #"FMADDsMULs",
        "FMADDsMULsPerCore",
        "SSR Loads per HW Loop",
      #  "HW Loops / SSR Loads per HW Loop",
      #  "myRegPerStream",
        "remainderTiles",
        # "Overlap Stall Time Total",
        # "Raw Compute Time Total",
        # "Global Sim E2E_dma",
        "Total CC Tiles",
        # "Overlap Stall Time Per Core",
        # "Avg A''",
        # "Avg B'",
        # "Avg C''",
        "timeout",
        "Avg CC Tile Size",
        "Avg A''/ B'",
        "Avg (A''+ B') / C''",
        "Avg A'",
        # "L3 Loads",
        # "L3 Stores",
        "Avg L3 Loads",
        "Avg L3 Stores",
        "Avg m'_sz / k_size",
        "Avg n'_sz / k_size"
    ]
    analyzed["mRem"] = analyzed["M"] % analyzed["m"]
    analyzed["niceMRem"] = analyzed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
    # analyzed["Overlap Stall Time Total"] = -1
    analyzed["Total CC Tiles"] = analyzed["SSR Config Count"]
    analyzed["FMADDsMULsPerCore"] = analyzed["FMADDsMULs"] / analyzed["Total CC Tiles"]
    # analyzed["Overlap Stall Time Per Core"] = -1
    # analyzed["Raw Compute / Overlap Stall"] = -1
    # analyzed["(Raw Compute / Overlap Stall) Per Core"] = -1
    analyzed["Y/X"]=timed["Avg n'_sz / k_size"] * timed["Avg A'"]
    
#     print(analyzed.columns)
#     print(analyzed[["Overlap Stall Time Total"]])

    timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
    timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
    timed = timed.sort_values(by="symbolMarker", ascending=True)
    timed["flatColor"] = "pink"

    prunePoint = ssr_prune(analyzed)

    ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    #print(ut)
    # combine timed points into single DF, then create absolute rank
    
    

    # special figures
    special_figs = []
    x_col = "SSR Configs"
    y_col = "dma"
    special_figs.append(
        prunedScatter(
            timed,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "OLD RATIO: timed divisors and (some) timed remainders",
            prunePoint,
            "symbolMarker",
        )
    )
    x_col = "Avg n'_sz / k_size"
    y_col = "dma"
    pruned = timed[timed["SSR Configs"]<= prunePoint]
    special_figs.append(
        scatterWithColor(
            pruned,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "OLD RATIO: timed divisors and (some) timed remainders",
            "symbolMarker",
        )
    )

    # more figures
    more_figs = []

    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)

def addFakeTime(df_ut, df_t):
    avgTime = sum(df_t["Kernel Time"].values) / len(df_t["Kernel Time"].values)
    df_ut["Kernel Time"] = avgTime
    avgTime = sum(df_t["Global Sim E2E_dma"].values) / len(df_t["Global Sim E2E_dma"].values)
    df_ut["Global Sim E2E_dma"] = avgTime
    df_ut["dma"] = avgTime
    df_ut["absoluteRank"] = -1
    df_ut["Overlap Stall Time Total"] = -1
    df_ut["Raw Compute Time Total"] = -1
    return df_ut

def generateExperimentalPruningGraphsStalls(timed, analyzed, titleOfWebpage):
    timed["Total CC Tiles"] = timed["SSR Config Count"]
    timed["FMADDsMULsPerCore"] = timed["FMADDsMULs"] / timed["Total CC Tiles"]
    # timed["Raw Compute / Overlap Stall"] = timed["Raw Compute Time Total"] / timed["Overlap Stall Time Total"]
    # timed["(Raw Compute / Overlap Stall) Per Core"] = timed["Raw Compute Time Total"] / timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
    timed["Overlap Stall Time Per Core"] = timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
    # timed["Y/X"]=timed["Avg n'_sz / k_size"] * timed["Avg A'"]
    timed["mRem"] = timed["M"] % timed["m"]
    timed["niceMRem"] = timed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
    x_col = "SSR Config Count"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        "timeout",
        "absoluteRank",
        "dma",
        "HW Loops",
        "SSR Configs",
        "FMADDsMULs",
        "FMADDsMULsPerCore",
        "SSR Loads per HW Loop",
        "HW Loops / SSR Loads per HW Loop",
      #  "myRegPerStream",
        "remainderTiles",
        # "Overlap Stall Time Total",
        # "Raw Compute Time Total",
        "Global Sim E2E_dma",
        "Total CL Tiles",
        "Total CC Tiles",
        "Overlap Stall Time Per Core",
        # "Avg A''",
        # "Avg B'",
        # "Avg C''",
        "L1 Usage",
        "Avg CC Tile Size",
        "Avg A''/ B'",
        "Avg (A''+ B') / C''",
        "Avg A'",
        # "L3 Loads",
        # "L3 Stores",
        "Avg L3 Loads",
        "Avg L3 Stores",
        "Avg m'_sz / k_size",
        "Avg n'_sz / k_size"
    ]
    analyzed["mRem"] = analyzed["M"] % analyzed["m"]
    analyzed["niceMRem"] = analyzed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
    #analyzed["Overlap Stall Time Total"] = -1
    analyzed["Total CC Tiles"] = analyzed["SSR Config Count"]
    analyzed["FMADDsMULsPerCore"] = analyzed["FMADDsMULs"] / analyzed["Total CC Tiles"]
    analyzed["Overlap Stall Time Per Core"] = -1
    analyzed = addFakeTime(analyzed,timed)
    # analyzed["Raw Compute / Overlap Stall"] = -1
    # analyzed["(Raw Compute / Overlap Stall) Per Core"] = -1
    analyzed["Y/X"]=timed["Avg n'_sz / k_size"] * timed["Avg A'"]
    
#     print(analyzed.columns)
#     print(analyzed[["Overlap Stall Time Total"]])

    timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
    timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
    timed = timed.sort_values(by="symbolMarker", ascending=True)
    timed["flatColor"] = "pink"

    #prunePoint = ssr_prune(analyzed)
    prunePoint = 24576
    ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    #print(ut)
    # combine timed points into single DF, then create absolute rank
    
    

    # special figures
    special_figs = []
    x_col = "Global Sim E2E_dma"
    y_col = "Global Sim E2E_dma"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "timed and untimed remainders",
            "symbolMarker",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "untimed remainders"
    )
    x_col = "FakeNN JSON Name"
    y_col = "Global Sim E2E_dma"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "timed data only",
            "symbolMarker",
        )
    )

    x_col = "Global Sim E2E_dma"
    y_col = "FMADDsMULsPerCore"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "timed data only",
            "symbolMarker",
        )
    )

    x_col = "FMADDsMULsPerCore"
    y_col = "Global Sim E2E_dma"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "timed data only",
            "symbolMarker",
        )
    )

    # size and number not same
    x_col = "Avg CC Tile Size"
    y_col = "SSR Configs"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "Global Sim E2E_dma",
            hover_data,
            "Tile Size vs Arithmetic Intensity",
            "symbolMarker",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "untimed remainders"
    )

    x_col = "SSR Configs"
    y_col = "Global Sim E2E_dma"
    special_figs.append(
        prunedScatter(
            timed,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "timed remainders",
            prunePoint,
            "symbolMarker",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "untimed remainders",
    )
    
    x_col = "SSR Configs"
    y_col = "Global Sim E2E_dma"
   # prunePoint = 15984
    pruned = timed[timed["SSR Configs"]<= prunePoint]
    special_figs.append(
        scatterWithColor(
            pruned,
            x_col,
            y_col,
            "Avg n'_sz / k_size",#"mRem",
            hover_data,
            f"After pruning to <= {prunePoint} SSR Configs",
            "symbolMarker",
        )
    )

    x_col = "SSR Configs"
    y_col = "Avg CC Tile Size"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "absoluteRank",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; How does CC tile size relate to FMADDs per core?",
            "niceMRem",
        )
    special_figs.append(myFig)

    x_col = "SSR Configs"
    y_col = "FMADDsMULsPerCore"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "absoluteRank",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; How do SSR configs relate to FMADDs per core?",
            "niceMRem",
        )
    special_figs.append(myFig)

    x_col = "Avg CC Tile Size"
    y_col = "FMADDsMULsPerCore"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "absoluteRank",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; How does CC tile size relate to FMADDs per core?",
            "niceMRem",
        )
    special_figs.append(myFig)

    
    x_col = "FMADDsMULsPerCore"
    y_col = "Avg CC Tile Size"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "absoluteRank",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; How does CC tile size relate to FMADDs per core?",
            "niceMRem",
        )
    special_figs.append(myFig)

    x_col = "FakeNN JSON Name"
    y_col = "Avg n'_sz / k_size"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "FMADDsMULsPerCore",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; prune out triangles, then take leftmost. THEN can we maximize by FMADDS?",
            "niceMRem",
        )
    special_figs.append(myFig)

    x_col = "Avg n'_sz / k_size"
    y_col = "Global Sim E2E_dma"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned[pruned["niceMRem"]== True],
            x_col,
            y_col,
            "FMADDsMULsPerCore",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; myrtle after non-squares: prune out triangles, then take leftmost. THEN can we maximize by FMADDS?",
             "mRem",
        )
    special_figs.append(myFig) #pruned[pruned["niceMRem"]== True]

    

    # print(pruned[pruned["FakeNN JSON Name"]== "384x384x384w21-24-39"][["Avg n'_sz / k_size","mRem"]]) #0.626573
    # print(pruned[pruned["FakeNN JSON Name"]== "384x384x384w21-24-39"][["Avg n'_sz / k_size","mRem"]].iloc(0))

    #               #,"Avg n'_sz / k_size"]])
    # print(pruned[pruned["Avg n'_sz / k_size"] == "0.626573"][["FakeNN JSON Name","Avg n'_sz / k_size","mRem"]])
 

    x_col = "Avg n'_sz / k_size"
    y_col = "Global Sim E2E_dma"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "mRem",
            hover_data,
            f"pruned to SSR configs <= {prunePoint}; myrtle: take leftmost, then the tie break by first preferring circle over triangle",
            "niceMRem",
        )
    special_figs.append(myFig)

    x_col = "Overlap Stall Time Per Core"
    y_col = "Global Sim E2E_dma"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "does ordering by overlap stall time help?",
            "niceMRem",
        )
    special_figs.append(myFig)

    x_col = "mRem"
    y_col = "Overlap Stall Time Per Core"
    pruned.sort_values(by="niceMRem",ascending=True)
    myFig = scatterWithColorSymbol(
            pruned,
            x_col,
            y_col,
            "Global Sim E2E_dma",
            hover_data,
            "how is mRem size related to overlap stall time?",
            "niceMRem",
        )
    special_figs.append(myFig)
    

    # more figures
    more_figs = []
    more_figs.append(
        px.bar(
            timed,
            x="absoluteRank",
            y=["dma"],  # Pass both column names here
            # barmode='group',         # Keeps them side-by-side
            title="Effect of Remainder Tiles divisible by 8 on Execution Time?",
            color="niceMRem",
            labels={
                "value": "Time (cycles)",
                "variable": "Metric",
            },  # 'value' and 'variable' are default labels for lists
            # template='plotly_dark'
        )
    )

    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)