import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def generateInteractiveGraphsTuples(divisors, remainders, title, titleOfWebpage):
    df = divisors[0]
    df["divisorRank"] = df["absoluteRank"]
    df["symbolMarker"] = "O"
    df["remainderTiles"] = df["remainderTiles"].apply(lambda x: f"{x}")
    rm_ut = remainders[1]
    rm_ut["symbolMarker"] = "^"
    rm_ut["divisorRank"] = -1
    rm_ut["remainderTiles"] = rm_ut["remainderTiles"].apply(lambda x: f"{x}")
    x_col = "Regular Loads"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        x_col,
        y_col,
        "Reused / Total SSR Loads",
        "SSR Config Count",
        "absoluteRank",
        "divisorRank",
        "L3 Loads",
        "HW Loops / SSR Loads",
        "mk/n",
        "L3 Loads Timed",
        "L1 Usage",
        "CC L1 Footprint",
        "dma",
        #  "oldRegPerStream",
        "regPerStream",
        #     "myRegPerStream",
        "FMADDsPerCore",
        "CC L1 / L1",
        "k/n",
        "fmaddsPerCore",
    ]

    if divisors[1] is not None:
        print("I can't handle timed AND untimed divisor tiles right now!")
    if remainders[0] is not None:
        # print("I can't handle timed remainder tiles right now!")
        rm = remainders[0]
        rm["divisorRank"] = -1
        rm["symbolMarker"] = "^"
        rm["remainderTiles"] = rm["remainderTiles"].apply(lambda x: f"{x}")
        # remove timed points from the untimed set
        # C = A[~A['ID'].isin(B['ID'])]
        rm_ut = rm_ut[~rm_ut["FakeNN JSON Name"].isin(rm["FakeNN JSON Name"])]
        # combine timed points into single DF, then create absolute rank
        timed = pd.concat([df, rm], join="inner", ignore_index=True)
        timed_sorted = timed.sort_values(by="dma", ascending=True)
        timed_sorted["absoluteRank"] = range(1, int(timed_sorted.shape[0] + 1))
        timed = timed_sorted.sort_values(by="symbolMarker", ascending=True)
        timed["flatColor"] = "pink"
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
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "timed divisors and (some) timed remainders",
        )

        x_col = "regPerStream"
        y_col = "dma"
        timed_pruned = timed[timed["SSR Config Count"] <= 1024]
        rm_ut_pruned = rm_ut[rm_ut["SSR Config Count"] <= 1024]
        special_figs.append(
            scatterWithColor(
                timed_pruned,
                x_col,
                y_col,
                "dma",
                hover_data,
                "OLD RATIO + pruned to SSR Configs <= 1024",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            special_figs[-1],
            rm_ut_pruned,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "timed divisors and (some) timed remainders",
        )

        # y_col = "HW Loops / SSR Loads"
        # x_col = "regPerStream"
        # color = "dma"
        # special_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"OLD RATIO vs. UPDATED SUMMATION OF RATIOS (untimed points omitted)","symbolMarker"))
        # #addScatterFlatColorMarker(special_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")

        x_col = "SSR Configs"
        y_col = "dma"
        color = "HW Loops / SSR Loads"
        special_figs.append(
            prunedScatter(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "UPDATED, SCALED SUMMATION OF RATIO: timed divisors and (some) timed remainders ",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            special_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        x_col = "HW Loops / SSR Loads"
        y_col = "dma"
        color = "dma"
        special_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "UPDATED, SCALED SUMMATION OF RATIO: timed divisors and (some) timed remainders ",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            special_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        # more figures
        more_figs = []
        x_col = "dma"
        y_col = "A SSR Reuse Loads"
        color = "A SSR Reuse Loads"
        # color = "HW Loops / SSR Loads"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )
        # timed["remainderTiles"] = "" + timed["remainderTiles"]

        x_col = "FMADDsPerCore"
        y_col = "dma"
        color = "L3 Loads"
        # color = "HW Loops / SSR Loads"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )
        # timed["remainderTiles"] = "" + timed["remainderTiles"]

        # fig = px.bar(
        #         timed,
        #         x="remainderTiles",
        #         y="dma",
        #         color="remainderTiles",
        #         barmode="group", # Use "group" for side-by-side, or "relative" for stacked
        #         title="dimensions with remainder tiles vs. kernel execution time",
        #         labels={"remainderTiles": "MNK dimensions", "dma": "cycles"},
        #         template="plotly_white"
        #         )
        # more_figs.append(fig)
        #         x_col = "remainderTiles"
        #         y_col = "dma"
        #         color = "remainderTiles"
        #        # color = "HW Loops / SSR Loads"
        #         more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"effect of dimensions using remainder tiles","symbolMarker"))
        #         #

        #         x_col = "remainderTiles"
        #         y_col = "A SSR Reuse Loads"
        #         color = "remainderTiles"
        #        # color = "HW Loops / SSR Loads"
        #         more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"effect of dimensions using remainder tiles <b>(overcounting??!)</b>","symbolMarker"))
        #         #addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")

        #         x_col = "remainderTiles"
        #         y_col = "HW Loops / SSR Loads"
        #         color = "remainderTiles"
        #        # color = "HW Loops / SSR Loads"
        #         more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"effect of dimensions using remainder tiles <b>(overcounting??!)</b>","symbolMarker"))
        #         #addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")

        #         x_col = "remainderTiles"
        #         y_col = "L1 Usage"
        #         color = "dma"
        #         more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"which remainder tiles have I timed, wrt L1 Usage?","symbolMarker"))
        #         #addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")

        #         x_col = "remainderTiles"
        #         y_col = "L1 Usage"
        #         color = "dma"
        #         more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"which remainder tiles have I timed, wrt L1 Usage?","symbolMarker"))
        #         addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")

        x_col = "L1 Usage"
        y_col = "dma"
        color = "FMADDsPerCore"
        # color = "HW Loops / SSR Loads"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        x_col = "L1 Usage"
        y_col = "L3 Loads"
        color = "dma"
        # color = "HW Loops / SSR Loads"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        y_col = "dma"
        x_col = "L3 Loads"
        color = "A SSR Reuse Loads"
        # color = "HW Loops / SSR Loads"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        x_col = "dma"
        y_col = "regPerStream"
        color = "SSR Configs"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        x_col = "dma"
        y_col = "HW Loops / SSR Loads"
        color = "SSR Configs"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        x_col = "dma"
        y_col = "A SSR Reuse Loads"
        color = "HW Loops / SSR Loads"
        more_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                color,
                hover_data,
                "timed divisors and (some) timed remainders",
                "symbolMarker",
            )
        )
        addScatterFlatColorMarker(
            more_figs[-1],
            rm_ut,
            x_col,
            y_col,
            "gray",
            "triangle-up",
            hover_data,
            "Remainders Untimed",
        )

        return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)

    # specialized graphs
    special_figs = []
    x_col = "SSR Configs"
    y_col = "dma"
    special_figs.append(
        prunedScatter(
            df,
            x_col,
            y_col,
            "regPerStream",
            hover_data,
            "timed divisors and untimed remainders",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        rm_ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "Remainders Untimed",
    )

    dfPruned = df[df["SSR Configs"] <= 1024]
    dfPrunedPoints = dfPruned["SSR Configs"].values.tolist()
    #  print(f"there are {len(dfPrunedPoints)} timed, pruned points to graph are: {dfPrunedPoints}")
    rm_utPruned = rm_ut[rm_ut["SSR Configs"] <= 1024]
    rm_utPruned.to_csv("./out/remaindertilesWithFewerThan1024.csv", index=False)
    fewerPoints = rm_utPruned["SSR Configs"].values.tolist()
    unique_ssr_configs = list(set(rm_ut["SSR Configs"].values.tolist()))
    unique_ssr_configs.sort()
    # print(f"There are {len(fewerPoints)} points with fewer than 1024 SSR configs: {fewerPoints}")

    x_col = "SSR Configs"
    y_col = "regPerStream"

    #    special_figs.append(scatterWithColor(dfPruned,x_col,y_col,"dma",hover_data,"timed divisors and untimed remainders"))
    #    addScatterFlatColorMarker(special_figs[-1],rm_utPruned,x_col,y_col,"lightpink","triangle-up",hover_data,"remainders untimed")
    special_figs.append(
        scatterWithColor(
            rm_utPruned,
            x_col,
            y_col,
            "regPerStream",
            hover_data,
            "untimed remainders and timed divisors with SSR Configs <= 1024",
        )
    )
    special_figs[-1].update_traces(marker_symbol="triangle-up")
    addScatterFlatColorMarker(
        special_figs[-1],
        dfPruned,
        x_col,
        y_col,
        "black",
        "circle",
        hover_data,
        "divisors timed",
    )

    x_col = "SSR Configs"
    y_col = "dma"
    special_figs.append(
        scatterWithColor(
            rm_utPruned,
            x_col,
            y_col,
            "regPerStream",
            hover_data,
            "untimed remainders and timed divisors with SSR Configs <= 1024",
        )
    )
    special_figs[-1].update_traces(marker_symbol="triangle-up")
    addScatterFlatColorMarker(
        special_figs[-1],
        dfPruned,
        x_col,
        y_col,
        "black",
        "circle",
        hover_data,
        "divisors timed",
    )

    # more experiments
    more_figs = []
    x_col = "SSR Configs"
    y_col = "regPerStream"
    more_figs.append(
        scatterWithColor(
            rm_ut,
            x_col,
            y_col,
            "HW Loops / SSR Loads",
            hover_data,
            "untimed remainders",
        )
    )

    x_col = "regPerStream"
    y_col = "HW Loops / SSR Loads"
    #    more_figs.append(scatterWithColor(rm_ut,x_col,y_col,"dma",hover_data,"untimed remainders"))
    #    more_figs[-1].update_traces(marker_symbol='triangle-up')
    more_figs.append(
        scatterWithColor(
            df, x_col, y_col, "dma", hover_data, "divisors timed and remainders untimed"
        )
    )
    #   more_figs[-1].add_traces(list(fig.data))
    addScatterFlatColorMarker(
        more_figs[-1],
        rm_ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "remainder untimed",
    )

    x_col = "HW Loops / SSR Loads"
    y_col = "oldRegPerStream"
    more_figs.append(
        scatterWithColor(df, x_col, y_col, "dma", hover_data, "divisors timed")
    )

    x_col = "SSR Configs"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            df, x_col, y_col, "HW Loops / SSR Loads", hover_data, "divisors timed"
        )
    )

    x_col = "m"
    y_col = "SSR Configs"
    more_figs.append(
        scatterWithColor(rm_ut, x_col, y_col, "m", hover_data, "untimed remainders")
    )

    x_col = "n"
    y_col = "SSR Configs"
    more_figs.append(
        scatterWithColor(rm_ut, x_col, y_col, "n", hover_data, "untimed remainders")
    )

    x_col = "k"
    y_col = "SSR Configs"
    more_figs.append(
        scatterWithColor(rm_ut, x_col, y_col, "k", hover_data, "untimed remainders")
    )

    x_col = "SSR Configs"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(df, x_col, y_col, "CC L1 / L1", hover_data, "divisors timed")
    )

    x_col = "SSR Configs"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            df,
            x_col,
            y_col,
            "regPerStream",
            hover_data,
            f"{title} (using old metric regPerStream)",
        )
    )

    # export to HTML
    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)


def generateInteractiveGraphs(df, title, titleOfWebpage):
    x_col = "Regular Loads"
    y_col = "Kernel Time"
    hover_data = [
        "JSON Name",
        x_col,
        y_col,
        "SSR Config Count",
        "absoluteRank",
        "L3 Loads",
        "HW Loops / SSR Loads",
        "mk/n",
        "L3 Loads Timed",
        "L1 Usage",
        "CC L1 Footprint",
        "tileC",
        "regPerStream",
        "CC L1 / L1",
        "sumSSRsRegs",
        "k/n",
        "fmaddsPerCore",
    ]
    small_hover_data = ["JSON Name", x_col, y_col, "HW Loops / SSR Loads", "L3 Loads"]

    # specialized graphs
    special_figs = []
    x_col = "SSR Configs"
    y_col = "Kernel Time"
    special_figs.append(
        prunedScatter(df, x_col, y_col, "HW Loops / SSR Loads", hover_data, "")
    )

    # more experiments
    more_figs = []
    x_col = "SSR Configs"
    y_col = "Kernel Time"
    more_figs.append(
        scatterWithColor(df, x_col, y_col, "CC L1 / L1", hover_data, title)
    )

    x_col = "SSR Configs"
    y_col = "Kernel Time"
    more_figs.append(
        scatterWithColor(
            df,
            x_col,
            y_col,
            "regPerStream",
            hover_data,
            f"{title} (using old metric regPerStream)",
        )
    )

    # export to HTML
    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)


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
        symbol="symbolMarker",
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


def generateInteractiveGraphsKernelVsDMA(
    rem_timed, rem_retimed, divisors_timed, title, titleOfWebpage
):
    # convert remainderTiles to string
    # rank points by dma time
    rem_timed["remainderTiles"] = rem_timed["remainderTiles"].apply(lambda x: f"{x}")
    rem_timed = rem_timed.sort_values(by="dma", ascending=True)
    rem_timed["absoluteRank"] = range(1, int(rem_timed.shape[0] + 1))
    # rem_timed["symbolMarker"] = 'O'

    rem_retimed["remainderTiles"] = rem_retimed["remainderTiles"].apply(
        lambda x: f"{x}"
    )
    rem_retimed = rem_retimed.sort_values(by="dma", ascending=True)
    rem_retimed["absoluteRank"] = range(1, int(rem_retimed.shape[0] + 1))
    # rem_retimed["symbolMarker"] = '^'

    x_col = "Regular Loads"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        x_col,
        y_col,
        "Reused / Total SSR Loads",
        "SSR Config Count",
        "absoluteRank",
        "divisorRank",
        "L3 Loads",
        "HW Loops / SSR Loads",
        "mk/n",
        "L3 Loads Timed",
        "L1 Usage",
        "CC L1 Footprint",
        "Kernel Time",
        "dma",
        #  "oldRegPerStream",
        "regPerStream",
        #     "myRegPerStream",
        "FMADDsPerCore",
        "CC L1 / L1",
        "k/n",
        "fmaddsPerCore",
    ]

    special_figs = []
    more_figs = []
    #         x_col = "dma"
    #         y_col = "A SSR Reuse Loads"
    #
    df_merged = pd.merge(
        rem_timed,
        rem_retimed,
        on="FakeNN JSON Name",
        how="inner",
        suffixes=("_withBug", "_noBug"),
    )
    df_merged["Kernel Time Difference"] = (
        df_merged["Kernel Time_noBug"] - df_merged["Kernel Time_withBug"]
    )
    # print(df_merged[['FakeNN JSON Name','Kernel Time Difference',"Kernel Time_withBug","Kernel Time_noBug"]] )
    df_merged["% Kernel Time Change"] = (
        df_merged["Kernel Time Difference"] / df_merged["Kernel Time_withBug"]
    ) * 100
    df_merged["symbolMarker"] = "O"
    # df_merged['RankDiff'] = df_merged['absoluteRank_withBug'] - df_merged['absoluteRank_noBug']
    # df_merged['Status'] = df_merged['Difference'].apply(
    # lambda x: 'Slow Down' if x < 0 else 'Same or Better'
    # )

    fig2 = px.bar(
        df_merged,
        x="FakeNN JSON Name",
        y="Kernel Time Difference",
        title="Change in Kernel Time After Fixing Parsing Bug",
        color="remainderTiles_withBug",
        # color_continuous_scale='RdBu', # Red for negative, Blue for positive
        hover_name="FakeNN JSON Name",
        hover_data={
            "Kernel Time Difference": ":.2f",  # Format to 2 decimal places
            "% Kernel Time Change": ":.2f",
            "Kernel Time_withBug": True,  # Show the raw value from File A
            "Kernel Time_noBug": True,  # Show the raw value from File B
            "FakeNN JSON Name": False,  # Hide Category if it's already on the X-axis
        },
        labels={
            "Kernel Time Difference": "Kernel Time Diff noBug - withBug (cycles)",
            "value": "Kernel Time (cycles)",
            "variable": "Source File",
        },
    )
    # labels={'Difference': 'Redundant - No Redundant (cycles)','value': 'Kernel Time (cycles)', 'variable': 'Source File'})
    special_figs.append(fig2)

    y_col = "Kernel Time_withBug"
    x_col = "dma_withBug"
    color = "dma_withBug"
    # color = "HW Loops / SSR Loads"
    hover_data = [
        "Kernel Time Difference",  # Format to 2 decimal places
        "Kernel Time_withBug",  # Show the raw value from File A
        "Kernel Time_noBug",  # Show the raw value from File B
        "dma_withBug",
        "dma_noBug",
        "FakeNN JSON Name",
    ]  # Hide Category if it's already on the X-axis

    special_figs.append(
        scatterWithColor(
            df_merged,
            x_col,
            y_col,
            color,
            hover_data,
            "Black Triangle = Kernel Time AFTER BUG FIXED;",
            "symbolMarker",
        )
    )
    y_col = "Kernel Time_noBug"
    addScatterFlatColorMarker(
        special_figs[-1],
        df_merged,
        x_col,
        y_col,
        "black",
        "triangle-up",
        hover_data,
        "No Bug",
    )

    df_merged["DMA - Kernel Time"] = (
        df_merged["dma_noBug"] - df_merged["Kernel Time_noBug"]
    )
    # more_figs.append(px.bar(
    #         df_merged,
    #         x='FakeNN JSON Name',
    #         y=['dma_noBug', 'Kernel Time_noBug'], # Pass both column names here
    #         barmode='group',         # Keeps them side-by-side
    #         title='DMA vs Kernel Time',
    #        # color='remainderTiles_withBug',
    #         labels={'value': 'Time (cycles)', 'variable': 'Metric'}, # 'value' and 'variable' are default labels for lists
    #         #template='plotly_dark'
    #         ))

    cols = [
        "FakeNN JSON Name",
        "DMA - Kernel Time",
        "remainderTiles",
        "symbolMarker",
        "Kernel Time",
        "dma",
    ]
    divisors_timed["DMA - Kernel Time"] = (
        divisors_timed["dma"] - divisors_timed["Kernel Time"]
    )
    df = divisors_timed
    df["symbolMarker"] = "O"
    df["remainderTiles"] = df["remainderTiles"].apply(lambda x: f"{x}")
    # print(divisors_timed[["FakeNN JSON Name","DMA - Kernel Time","remainderTiles"]])
    left = divisors_timed[cols]
    df_merged["remainderTiles"] = df_merged["remainderTiles_withBug"]
    df_merged["Kernel Time"] = df_merged["Kernel Time_noBug"]
    df_merged["dma"] = df_merged["dma_noBug"]
    # print(df_merged[["FakeNN JSON Name","DMA - Kernel Time","remainderTiles"]])
    # print(f"{df.keys()} and then {df_merged.keys()}")
    right = df_merged[cols]

    df_merged = pd.concat([left, right])
    # print(df_merged.keys())
    # print(df_merged[cols])
    # print(f"left: {left.shape}, right: {right.shape}, together: {df_merged.shape}")
    df_sorted = df_merged.sort_values(by="dma", ascending=True)
    df_sorted["absoluteRank"] = range(1, int(df_sorted.shape[0] + 1))
    df = df_sorted
    more_figs.append(
        px.bar(
            df,
            x="FakeNN JSON Name",
            y=["DMA - Kernel Time"],  # Pass both column names here
            # barmode='group',         # Keeps them side-by-side
            title="Difference between End to End Execution time (dma) and Computation time (Kernel Time)",
            color="remainderTiles",
            labels={
                "value": "Time (cycles)",
                "variable": "Metric",
            },  # 'value' and 'variable' are default labels for lists
            # template='plotly_dark'
        )
    )

    y_col = "Kernel Time"
    x_col = "dma"
    color = "remainderTiles"
    # color = "HW Loops / SSR Loads"
    hover_data = [
        "DMA - Kernel Time",  # Format to 2 decimal places
        "FakeNN JSON Name",
        "dma",
        "Kernel Time",
        "absoluteRank",
    ]  # Hide Category if it's already on the X-axis

    more_figs.append(
        scatterWithColor(
            df,
            x_col,
            y_col,
            color,
            hover_data,
            "dma vs kernel time, all 128 cube points",
            "symbolMarker",
        )
    )

    y_col = "DMA - Kernel Time"
    x_col = "dma"
    color = "remainderTiles"
    more_figs.append(
        scatterWithColor(
            df,
            x_col,
            y_col,
            color,
            hover_data,
            "dma vs kernel-dma time diff, all 128 cube points",
            "symbolMarker",
        )
    )

    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)


def generateInteractiveBarGraphs(divisors, remainders, title, titleOfWebpage):
    x_col = "Regular Loads"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        x_col,
        y_col,
        "Reused / Total SSR Loads",
        "SSR Config Count",
        "absoluteRank",
        "divisorRank",
        "L3 Loads",
        "HW Loops / SSR Loads",
        "mk/n",
        "L3 Loads Timed",
        "L1 Usage",
        "CC L1 Footprint",
        "dma",
        #  "oldRegPerStream",
        "regPerStream",
        #     "myRegPerStream",
        "FMADDsPerCore",
        "CC L1 / L1",
        "k/n",
        "fmaddsPerCore",
    ]

    if divisors[1] is not None:
        raise Exception("I can't handle timed AND untimed divisor tiles right now!")
    if remainders[0] is None or remainders[1] is None:
        raise Exception("I require both TIMED and UNTIMED remainder tiles!")

    def sumProloguesEpilogues(df):
        for cat in ["Before Computation", "After Computation"]:
            cat_total = f"{cat} Total"
            df[cat_total] = 0
            for c in range(0, 8):
                suffix = f"_cc_{c}"
                df[cat_total] = df[cat_total] + df[f"{cat}{suffix}"]
        return df

    def otherTime(df):
        for cat in [
            "Before Computation",
            "After Computation",
            "Overlap Stall Time",
            "Raw Compute Time",
        ]:
            cat_total = "Time Accounted For"
            df[cat_total] = 0
        for c in range(0, 8):
            total = f"Time Accounted For_cc_{c}"
            df[total] = 0
            for cat in [
                "Before Computation",
                "After Computation",
                "Overlap Stall Time",
                "Raw Compute Time",
            ]:
                col = f"{cat}_cc_{c}"
                df[total] = df[total] + df[col]
            otherTotal = f"Time Unaccounted For_cc_{c}"
            df[otherTotal] = df["dma"] - df[total]
        return df

    def sumTimeAccountedFor(df):
        for cat in ["Time Accounted For", "Time Unaccounted For"]:
            cat_total = f"{cat} Total"
            df[cat_total] = 0
            for c in range(0, 8):
                suffix = f"_cc_{c}"
                df[cat_total] = df[cat_total] + df[f"{cat}{suffix}"]
        return df

    # def otherTime(df):
    #         df["other"] = df["dma"]-df['Overlap Stall Time Total']- df['Raw Compute Time Total'] - df['Before Computation Total'] - df['After Computation Total']
    #         return df
    # preprocess data
    df = divisors[0]
    df["divisorRank"] = df["absoluteRank"]
    df["symbolMarker"] = "O"
    df["remainderTiles"] = df["remainderTiles"].apply(lambda x: f"{x}")
    df = sumProloguesEpilogues(df)
    df = otherTime(df)
    df = sumTimeAccountedFor(df)
    rm = remainders[0]
    rm["divisorRank"] = -1
    rm["symbolMarker"] = "^"
    rm["remainderTiles"] = rm["remainderTiles"].apply(lambda x: f"{x}")
    rm = sumProloguesEpilogues(rm)
    rm = otherTime(rm)
    rm = sumTimeAccountedFor(rm)
    rm_ut = remainders[1]
    rm_ut["symbolMarker"] = "^"
    rm_ut["divisorRank"] = -1
    rm_ut["remainderTiles"] = rm_ut["remainderTiles"].apply(lambda x: f"{x}")
    rm_ut["Before Computation Total"] = -1
    rm_ut["After Computation Total"] = -1

    rm_ut = rm_ut[~rm_ut["FakeNN JSON Name"].isin(rm["FakeNN JSON Name"])]

    # combine timed points into single DF, then create absolute rank
    timed = pd.concat([df, rm], join="inner", ignore_index=True)
    timed_sorted = timed.sort_values(by="dma", ascending=True)
    timed_sorted["absoluteRank"] = range(1, int(timed_sorted.shape[0] + 1))
    timed = timed_sorted.sort_values(by="symbolMarker", ascending=True)
    timed["flatColor"] = "pink"

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
        rm_ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "timed divisors and (some) timed remainders",
    )

    x_col = "regPerStream"
    y_col = "dma"
    timed_pruned = timed[timed["SSR Config Count"] <= 1024]
    rm_ut_pruned = rm_ut[rm_ut["SSR Config Count"] <= 1024]
    special_figs.append(
        scatterWithColor(
            timed_pruned,
            x_col,
            y_col,
            "dma",
            hover_data,
            "OLD RATIO + pruned to SSR Configs <= 1024",
            "symbolMarker",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        rm_ut_pruned,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "timed divisors and (some) timed remainders",
    )

    x_col = "SSR Configs"
    y_col = "dma"
    color = "HW Loops / SSR Loads"
    special_figs.append(
        prunedScatter(
            timed,
            x_col,
            y_col,
            color,
            hover_data,
            "UPDATED, SCALED SUMMATION OF RATIO: timed divisors and (some) timed remainders ",
            "symbolMarker",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        rm_ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "Remainders Untimed",
    )

    x_col = "HW Loops / SSR Loads"
    y_col = "dma"
    color = "dma"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            color,
            hover_data,
            "UPDATED, SCALED SUMMATION OF RATIO: timed divisors and (some) timed remainders ",
            "symbolMarker",
        )
    )
    addScatterFlatColorMarker(
        special_figs[-1],
        rm_ut,
        x_col,
        y_col,
        "gray",
        "triangle-up",
        hover_data,
        "Remainders Untimed",
    )

    # more figures
    more_figs = []

    # df = pd.DataFrame(data)
    #   print(timed.columns)
    #  print(timed[['Overlap Stall Time Total', 'Raw Compute Time Total','Before Compute Time Total','After Compute Time Total']])
    timed["8*dma"] = timed["dma"] * 8
    timed.sort_values(by="absoluteRank", ascending=True)
    # # 2. Create the stacked bar graph
    fig = px.bar(
        timed,
        x="absoluteRank",
        y=[
            "Overlap Stall Time Total",
            "Raw Compute Time Total",
            "Before Computation Total",
            "After Computation Total",
        ],  # Each column is a trace
        title="End-to-End Execution Time Breakdown (divisors AND remainders)",
        barmode="stack",  # This stacks the traces on top of each other
        hover_data={
            "FakeNN JSON Name": True,  # Hide Category if it's already on the X-axis
            "remainderTiles": True,
            "dma": ":.2f",  # Format to 2 decimal places
            "Kernel Time": True,  # Show the raw value from File B
            "L3 Loads": True,
            "HW Loops / SSR Loads": True,
            "mk/n": True,
            "L1 Usage": True,
            "Total CC Tiles": True,
        },
        labels={"value": "Cycles", "variable": "Absolute Rank; smaller is faster"},
        template="presentation",
    )
    more_figs.append(fig)

    pruned = timed[timed["absoluteRank"] < 50]
    pruned.sort_values(by="absoluteRank", ascending=True)
    fig = px.bar(
        pruned,
        x="absoluteRank",
        y=[
            "Overlap Stall Time Total",
            "Raw Compute Time Total",
            "Before Computation Total",
            "After Computation Total",
        ],  # Each column is a trace
        title="End-to-End Execution Time Breakdown (divisors AND remainders)",
        barmode="stack",  # This stacks the traces on top of each other
        hover_data={
            "absoluteRank": True,
            "FakeNN JSON Name": True,  # Hide Category if it's already on the X-axis
            "remainderTiles": True,
            "dma": ":.2f",  # Format to 2 decimal places
            "Kernel Time": True,  # Show the raw value from File B
            "L3 Loads": True,
            "HW Loops / SSR Loads": True,
            "mk/n": True,
            "L1 Usage": True,
            "Total CC Tiles": True,
            "SSR Config Count": True,
        },
        labels={"value": "Cycles", "variable": "Absolute Rank; smaller is faster"},
        template="presentation",
    )
    more_figs.append(fig)

    # pruned = timed[timed["absoluteRank"] < 5]
    # print(pruned.columns)
    # print(pruned[['dma', 'Raw Compute Time_cc_1',"Overlap Stall Time_cc_1"]])
    fig = px.bar(
        timed,
        x="absoluteRank",
        y=[
            "8*dma",
            "Overlap Stall Time Total",
            "Raw Compute Time Total",
        ],  # Each column is a trace
        title="Reality Check: 8*(dma core end-to-end time) >=  (raw compute + overlap stall over all compute cores)",
        barmode="group",  # This stacks the traces on top of each other
        hover_data={
            "FakeNN JSON Name": True,  # Hide Category if it's already on the X-axis
            "remainderTiles": True,
            "dma": ":.2f",  # Format to 2 decimal places
            "Kernel Time": True,  # Show the raw value from File B
            "L3 Loads": True,
            "HW Loops / SSR Loads": True,
            "mk/n": True,
            "L1 Usage": True,
            "Total CC Tiles": True,
        },
        labels={"value": "Cycles", "variable": "Absolute Rank; smaller is faster"},
        template="presentation",
    )
    more_figs.append(fig)

    fig = px.bar(
        timed,
        x="absoluteRank",
        y=[
            "8*dma",
            "Overlap Stall Time Total",
            "Raw Compute Time Total",
        ],  # Each column is a trace
        title="Reality Check: 8*(dma core end-to-end time) >=  (raw compute + overlap stall over all compute cores)",
        barmode="group",  # This stacks the traces on top of each other
        hover_data={
            "FakeNN JSON Name": True,  # Hide Category if it's already on the X-axis
            "remainderTiles": True,
            "dma": ":.2f",  # Format to 2 decimal places
            "Kernel Time": True,  # Show the raw value from File B
            "L3 Loads": True,
            "HW Loops / SSR Loads": True,
            "mk/n": True,
            "L1 Usage": True,
            "Total CC Tiles": True,
        },
        labels={"value": "Cycles", "variable": "Absolute Rank; smaller is faster"},
        template="presentation",
    )
    more_figs.append(fig)
    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)


def checkStallTimeCorrectness(df):
    # overcounting = []
    # undercounting = []
    # same = []
    df = df.reset_index()  # Make sure indexes pair with number of rows
    for index, row in df.iterrows():
        overcounting = []
        undercounting = []
        same = []
        e2e = row["dma"]
        for c in range(0, 8):
            sum = 0
            for cat in [
                "Before Computation",
                "After Computation",
                "Overlap Stall Time",
                "Raw Compute Time",
            ]:
                sum = sum + row[f"{cat}_cc_{c}"]
            diff = e2e - sum
            if diff < 0:
                overcounting.append(diff)
                # raise Exception(f"sum of times less than total time!!{e2e}-{sum}={diff}")
            else:
                if diff == 0:
                    same.append(diff)
                else:
                    undercounting.append(diff)
        print(
            f"{row['FakeNN JSON Name']}: overcounted: {len(overcounting)} undercounted: {len(undercounting)} Exactly right: {len(same)}"
        )
        print("\t", end="")
        if len(overcounting) != 0:
            print(f"max:min overcount:{max(overcounting)}:{min(overcounting)}")
        if len(undercounting) != 0:
            print(f"max:min undercount:{max(undercounting)}:{min(undercounting)}")
    # print(f"max overcount:{max(overcounting)} max undercount: {max(undercounting)}")
    return True


def generateInteractiveBarAndScatterGraphs(timed, analyzed, titleOfWebpage):
    x_col = "SSR Config Count"
    y_col = "dma"
    hover_data = [
        "JSON Name",
        "absoluteRank",
        "dma",
        "SSR Loads",
        "HW Loops",
        "FMADDs",
        "MULs",
        "FMADDsMULs",
        "SSR Loads per HW Loop",
        "HW Loops / SSR Loads per HW Loop",
        "myRegPerStream",
        "remainderTiles",
        "Overlap Stall Time Total",
        "Raw Compute Time Total",
    ]
    analyzed["Overlap Stall Time Total"] = -1
    analyzed["Raw Compute Time Total"] = -1
    print(analyzed.columns)
    print(analyzed[["Overlap Stall Time Total"]])
    timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
    timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
    timed = timed.sort_values(by="symbolMarker", ascending=True)
    timed["flatColor"] = "pink"

    ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    print(ut)
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
        "timed divisors and (some) timed remainders",
    )

    special_figs.append(
        prunedScatter(
            timed,
            x_col,
            y_col,
            "HW Loops / SSR Loads per HW Loop",
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
    
    x_col = "SSR Configs"
    y_col = "Overlap Stall Time Total"
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

        # x_col = "regPerStream"
        # y_col = "dma"
        # timed_pruned = timed[timed["SSR Config Count"] <= 1024]
        # rm_ut_pruned = rm_ut[rm_ut["SSR Config Count"] <= 1024]
        # special_figs.append(
        #     scatterWithColor(
        #         timed_pruned,
        #         x_col,
        #         y_col,
        #         "dma",
        #         hover_data,
        #         "OLD RATIO + pruned to SSR Configs <= 1024",
        #         "symbolMarker",
        #     )
        # )

    # more figures
    more_figs = []

    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)
