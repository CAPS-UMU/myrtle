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


def scatterWithColor(df, x_col, y_col, color, hover_data, title, marker=""):
    fig8 = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
      #  symbol="symbolMarker",
     #   symbol_sequence=["circle", "triangle-up", "triangle-up"],
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


def prunedScatter(df, x_col, y_col, color, hover_data, title, marker=""):
    df_mod = df
    if marker == "":
        df["symbolMarker"] = "O"
    # df_mod["color"] = df_mod["k/n"]
    # print(df_mod["L3 Loads"].values)
    # top =  max(df_mod["L3 Loads"].values)
    # bot =  min(df_mod["L3 Loads"].values)
    # mid = (top - bot) / 2.0
    # prunePointL3 = bot + mid#2686976
    unique_ssr_configs = list(
        set(df_mod["SSR Configs"].values.tolist())
    )  # remove duplicates
    unique_ssr_configs.sort()  # sort least to greateset
    prunePoint = unique_ssr_configs[1]  # prune to two smallest groups of ssr_configs
    prunePoint = 1024
    fig14 = px.scatter(
        df_mod,
        x=x_col,
        y=y_col,
        symbol="symbolMarker",
        symbol_sequence=["circle", "triangle-up", "triangle-up"],
        color=color,  # "regPerStream",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b> with SSR Configs >= {prunePoint} separated with vertical dotted line.",
    )
    fig14.add_vline(x=prunePoint, line_width=2, line_dash="dash", line_color="green")
    # turn the pruned points gray?
    return fig14


def generateInteractiveBarAndScatterGraphs(timed, analyzed, titleOfWebpage):
    timed["FMADDsMULsPerCore"] = timed["FMADDsMULs"] / timed["Total CC Tiles"]
    timed["Raw Compute / Overlap Stall"] = timed["Raw Compute Time Total"] / timed["Overlap Stall Time Total"]
    timed["(Raw Compute / Overlap Stall) Per Core"] = timed["Raw Compute Time Total"] / timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
    timed["Overlap Stall Time Per Core"] = timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
    x_col = "SSR Config Count"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        "absoluteRank",
        "dma",
        "HW Loops",
        "FMADDsMULs",
        "FMADDsMULsPerCore",
        "SSR Loads per HW Loop",
        "HW Loops / SSR Loads per HW Loop",
      #  "myRegPerStream",
        "remainderTiles",
        "Overlap Stall Time Total",
        "Raw Compute Time Total",
        "Global Sim E2E_dma",
        "Total CC Tiles",
        "Overlap Stall Time Per Core",
        "Avg A''",
        "Avg B'",
        "Avg C''",
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
    analyzed["Overlap Stall Time Total"] = -1
    analyzed["Raw Compute Time Total"] = -1
    analyzed["Global Sim E2E_dma"] = -1
    analyzed["Total CC Tiles"] = analyzed["SSR Config Count"]
    analyzed["FMADDsMULsPerCore"] = analyzed["FMADDsMULs"] / analyzed["Total CC Tiles"]
    analyzed["Overlap Stall Time Per Core"] = -1
    analyzed["Raw Compute / Overlap Stall"] = -1
    analyzed["(Raw Compute / Overlap Stall) Per Core"] = -1
    
#     print(analyzed.columns)
#     print(analyzed[["Overlap Stall Time Total"]])

    timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
    timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
    timed = timed.sort_values(by="symbolMarker", ascending=True)
    timed["flatColor"] = "pink"

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
            "regPerStream",
            hover_data,
            "OLD RATIO: timed divisors and (some) timed remainders",
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
        "timed divisors and (some) timed remainders timed divisors and (some) timed remainders",
    )

    special_figs.append(
        prunedScatter(
            timed,
            x_col,
            y_col,
            "HW Loops / SSR Loads per HW Loop",
            hover_data,
            "Summation of new ratio: ",
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
        "timed divisors and (some) timed remainders",
    )

    x_col = "Overlap Stall Time Total"
    y_col = "dma"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "HW Loops / SSR Loads per HW Loop",
            hover_data,
            "stall time vs e2e time",
            "symbolMarker",
        )
    )


    # more figures
    more_figs = []

    # let's try pruning by SSR configs FIRST
    timed_sorted = timed.sort_values(by="SSR Configs", ascending=True)
    timed_pruned=timed_sorted[timed_sorted["SSR Configs"] <= 1024]#timed_sorted.head(50)
    x_col = "Overlap Stall Time Total"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "HW Loops / SSR Loads per HW Loop",
            hover_data,
            "pruned to SSR configs <= 1024",
            "symbolMarker",
        )
    )
    # THEN pruning by bottom third based on stall time
    timed_sorted = timed_pruned.sort_values(by="Overlap Stall Time Total", ascending=True)
    print(timed_sorted.shape)
    print(int(timed_sorted.shape[0]/3))
    timed_pruned=timed_sorted.head(int(timed_sorted.shape[0]/3))
    x_col = "Overlap Stall Time Total"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "HW Loops / SSR Loads per HW Loop",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col = "Overlap Stall Time Total"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col = "Overlap Stall Time Total"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg A'",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col = "Avg m'_sz / k_size"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col = "Avg n'_sz / k_size"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg m'_sz / k_size",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col =  "Avg A'"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col =  "Avg A'"
    y_col = "Avg n'_sz / k_size"
    fig=scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Global Sim E2E_dma",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time - best solution so far",
            "symbolMarker",
        )
    more_figs.append(fig)
    
    special_figs= [fig] + special_figs

    # pruned_out = timed[~timed["FakeNN JSON Name"].isin(timed_pruned["FakeNN JSON Name"])]
    # addScatterFlatColorMarker(
    #     more_figs[-1],
    #     pruned_out,
    #     x_col,
    #     y_col,
    #     "gray",
    #     "triangle-up",
    #     hover_data,
    #     "pruned out, timed points",
    # )

    # ut_pruned = analyzed[~analyzed["FakeNN JSON Name"].isin(timed_pruned["FakeNN JSON Name"])]
    # addScatterFlatColorMarker(
    #     more_figs[-1],
    #     ut_pruned,
    #     x_col,
    #     y_col,
    #     "gray",
    #     "triangle-up",
    #     hover_data,
    #     "untimed timed remainders",
    # )

   

    x_col =  "Avg A''"
    y_col = "Avg n'_sz / k_size"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Global Sim E2E_dma",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col = "Avg n'_sz / k_size"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg A''",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )

    x_col = "Avg n'_sz / k_size"
    y_col = "Global Sim E2E_dma"
    more_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "Avg L3 Stores",
            hover_data,
            "pruned to SSR configs <= 1024, THEN pruned to bottom third based on overlap stall time",
            "symbolMarker",
        )
    )
    




    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)
