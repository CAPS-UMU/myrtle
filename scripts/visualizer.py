import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import math
# theColorBar="ylorrd_r"
theColorBar="haline"

def jugaadTitle(df):
    M=int(df["M"][0])
    N=int(df["N"][0])
    K=int(df["K"][0])
    dims=f"{M}x{N}x{K}"
    t="Transformer"
    if (M == 128) and (N == 128) and (K == 128):
        t="BertTiny" 
    if (M == 128) and (N == 768) and (K == 768):
        t="Roberta" 
    if (M == 192) and (N == 384) and (K == 384):
        t="BGESmall" 
    if (M == 384) and (N == 384) and (K == 384):
        t="MiniLM" 
    if (M == 256) and (N == 256) and (K == 256):
        t="BertMini" 
    if (M == 512) and (N == 512) and (K == 512):
        t="Bert" 
    return f"{t} Matmul {dims}"

def genResultGraphPDF(title,timed, untimed, recentlyPruned,hover_data):
    has_low_values = (timed["mRem"] < 0).any()
    print(f"Are there values below -0.5 in timed? {has_low_values}")
    has_low_values = (untimed["mRem"] < 0).any()
    print(f"Are there values below -0.5 untimed? {has_low_values}")
    has_low_values = (recentlyPruned["mRem"] < 0).any()
    print(f"Are there values below -0.5 in recentlyPruned? {has_low_values}")
    if len(timed)<5:
        rp_max=recentlyPruned["Time (cycles)"].max()
        tm_max=timed["Time (cycles)"].max()
        newFakeTime = max(tm_max,rp_max)
        if newFakeTime==rp_max:
            newFakeTime = rp_max*1.25
    else:
        newFakeTime = timed["Time (cycles)"].max()
    # customize the height of the untimed points
    untimed = untimed.copy(deep=True)
    untimed["Time (cycles)"]=newFakeTime    
    x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
    y_col = "Time (cycles)"#"Global Sim E2E_dma"
    fig=scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "testing short tile",
            "timeout",
            ["circle","cross"]
    )
    addScatterFlatColorMarker(
        fig,
        untimed,
        x_col,
        y_col,
        "gray",
        "square",
        hover_data,
        "untimed w/ nice m remainder, n/k < 1"
        )
    if(len(timed)<5):
        addScatterFlatColorMarker(
        fig,
        recentlyPruned,
        x_col,
        y_col,
        "gray",
        "circle-open",
        hover_data,
        "untimed w/ nice m remainder, n/k < 1"
        )
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,             # Center point on a scale from 0 to 1
            xanchor="center"   # Anchor the title string by its exact middle
        )
    )
    fig.update_layout(
    font=dict(
       # family="CMU Serif",  # Tells Plotly to search your system for Computer Modern
        family="CMU Serif, Computer Modern, Latin Modern Roman, Serif",
        size=12,
        color="black"
    ),
    )
    fig.update_traces(showlegend=False)


    #fig.write_image(f"out/{title}.pdf", width=1200, height=800, scale=3)
    # I have a 7x10 paper, so 1/3 of the width is approx 2.3 inches
    # let's try 600 dpi for the scale
    # plotly graph is 7 wide and 8 tall
    dpi = 72#300
    ratio=4/3.2
    heightPx=4*ratio*dpi#(3.2/8*7)*dpi
    widthPx=6*ratio*dpi#3.2*dpi
  
    fig.update_layout(
    # 1. Maintain your physical 6x4 inch PDF aspect ratio
    width=widthPx,  
    height=heightPx,
    
    # 2. Aggressively reduce the outer canvas padding
    margin=dict(
        l=30,  # Left margin (space for Y-axis titles/labels)
        r=20,  # Right margin (space near your legend)
        t=35,  # Top margin (just enough room for your centered title)
        b=30   # Bottom margin (space for X-axis titles/labels)
    ),
    
    # 3. Tell the axes to automatically expand only what they need
    xaxis=dict(automargin=True),
    yaxis=dict(automargin=True),
    
    # 4. Your clean, smaller font settings
    font=dict(
        family="CMU Sans Serif Demi Condensed,CMU Typewriter Text", 
        size=14,
        color="black"
    ),
    template="plotly_white"
)
    # Scale it by 3x upon export to achieve 300 DPI crispness.
    # This keeps the text, lines, and markers perfectly proportioned!
    fig.update_layout(
    coloraxis=dict(
        cmin=0,         # Force the scale to start exactly at 0
        # cmax=8        # Optional: You can also hardcode the maximum if you want
    )
    )

    fig.write_image(f"out/{title}.pdf", scale=1)
    #fig.write_image(f"out/{title}.pdf", width=widthPx, height=heightPx)
    return fig


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
    <h1 id="top" style="text-align:center;">{titleOfWebpage}</h1>
    <a href="index.html" >Back to Landing Page</a>
    <div class="dashboard">
      {specialDivsAsHTML} 
        {"<div>More Experiments</div>"}
        {moreDivsAsHTML}        
        </div>
        <span id="bottom"><a href="#top" >Back to Top</a></span>
        </body>
        </html>
        """
    return html

def scatterWithColor(df, x_col, y_col, color, hover_data, title, maerker=""):
    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        color_continuous_scale=theColorBar,
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b>",
    )

#color_discrete_sequence=["gray"]
def scatterWithFlatColor(df, x_col, y_col, color, hover_data, title, unused):
    fig8 = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[color],
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b>",
    )
    return fig8

def scatterWithColorSymbol(df, x_col, y_col, color, hover_data, title, marker, markerSequence=["circle", "triangle-up", "triangle-up"]):
    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        color_continuous_scale=theColorBar,
        symbol=marker,
        symbol_sequence=markerSequence,
        hover_data=hover_data,  # Show these columns on hover
        title=title#f"{title} <b>{x_col} vs {y_col}</b>",
    )

def scatterWithFlatColorSymbol(df, x_col, y_col, color, hover_data, title, marker,markerSeq=["circle", "triangle-up", "triangle-up"]):
    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[color],
        symbol=marker,
        symbol_sequence=markerSeq,
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b>",
    )

def addFakeTime(df_ut, df_t):
    avgTime = sum(df_t["Kernel Time"].values) / len(df_t["Kernel Time"].values)
    df_ut["Kernel Time"] = avgTime
    # avgTime = sum(df_t["Global Sim E2E_dma"].values) / len(df_t["Global Sim E2E_dma"].values)
    maxTime = df_t["Global Sim E2E_dma"].max()*1.5
    df_ut["Global Sim E2E_dma"] = maxTime
    df_ut["dma"] = maxTime
    df_ut["absoluteRank"] = -1
    df_ut["Overlap Stall Time Total"] = -1
    df_ut["Raw Compute Time Total"] = -1
    return df_ut

# return the SSR config value that is the largest of the bottom frac
def ssr_prune_frac(df,frac):
        unique_ssr_configs = list(
            set(df["SSR Config Count"].values.tolist())
        )  # remove duplicates
        unique_ssr_configs.sort()  # sort least to greateset
        third = unique_ssr_configs[0:int(len(unique_ssr_configs)/frac)]
        #print(f"{unique_ssr_configs} with len {len(unique_ssr_configs)} and bottom third {third}")
        prunePoint = unique_ssr_configs[int(len(unique_ssr_configs)/frac)]  # prune to smallest frac of ssr_configs   
        return prunePoint

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


def visualizePruning(timed, analyzed, full, titleOfWebpage):
     timed["Total CC Tiles"] = timed["SSR Config Count"]
     timed["FMADDsMULsPerCore"] = timed["FMADDsMULs"] / timed["Total CC Tiles"]
     timed["1/FMADDS"]=1/timed["FMADDsMULsPerCore"]
     timed["Overlap Stall Time Per Core"] = timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
     timed["mRem"] = timed["M"] % timed["m"]
     timed["1/mRem"]=1/timed["mRem"]
     timed["niceMRem"] = timed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
     timed["howNice"] = timed["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
     timed["hypotenuse"] = timed[["mRem","1/FMADDS"]].apply(lambda x: math.sqrt(x["mRem"]*x["mRem"]+x["1/FMADDS"]*x["1/FMADDS"]),axis=1)
     timed["Time (cycles)"]=timed["Global Sim E2E_dma"]
    # print(timed[["JSON Name","mRem","1/FMADDS","hypotenuse"]])
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
        "Time (cycles)",#"SSR Loads per HW Loop",
        "HW Loops / SSR Loads per HW Loop",
        "remainderTiles",
        "Global Sim E2E_dma",
        "Total CL Tiles",
        "Total CC Tiles",
        #"Overlap Stall Time Per Core",
        "1/FMADDS",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
        "Avg L3 Loads",
        "Avg L3 Stores",
        "Avg n'_sz / k_size",
        "timedData",
     ]
     minimal_hover = [
        "JSON Name",       
        "SSR Configs",
        "L1 Usage",       
     ]
     analyzed["mRem"] = analyzed["M"] % analyzed["m"]
     analyzed["1/mRem"]=1/analyzed["mRem"]
     analyzed["niceMRem"] = analyzed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
     analyzed["howNice"] = analyzed["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
     analyzed["Total CC Tiles"] = analyzed["SSR Config Count"]
     analyzed["FMADDsMULsPerCore"] = analyzed["FMADDsMULs"] / analyzed["Total CC Tiles"]
     analyzed["1/FMADDS"]=1/analyzed["FMADDsMULsPerCore"]
     analyzed["hypotenuse"] = analyzed[["1/mRem","FMADDsMULsPerCore"]].apply(lambda x: math.sqrt(x["1/mRem"]*x["1/mRem"]+x["FMADDsMULsPerCore"]*x["FMADDsMULsPerCore"]),axis=1)
     analyzed["Overlap Stall Time Per Core"] = -1
     analyzed = addFakeTime(analyzed,timed)
     analyzed["Y/X"]=timed["Avg n'_sz / k_size"] * timed["Avg A'"]
     analyzed["timedData"] = False
     analyzed["hypotenuse"] = analyzed[["mRem","1/FMADDS"]].apply(lambda x: math.sqrt(x["mRem"]*x["mRem"]+x["1/FMADDS"]*x["1/FMADDS"]),axis=1)
     analyzed["Time (cycles)"]=analyzed["Global Sim E2E_dma"]
     
     timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
     timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
     timed = timed.sort_values(by="symbolMarker", ascending=True)
     timed["flatColor"] = "pink"
     timed["timedData"] = True

     prunePoint = ssr_prune_frac(full,3)
     full["SSR Configs"] = full["SSR Config Count"]
     full["L1 Usage"] = full["Space Needed in L1"]
     full = addFakeTime(full,timed)

    # we assume untimed points are a subset of the pruned search space
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
     pruned = full[full["SSR Config Count"] < prunePoint]

    # special figures
     special_figs = []
     # more figs
     more_figs = []

     x_col = "Global Sim E2E_dma"#"Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "Reality Check. Make sure fastest point is ranked 1.",
            "timedData",
            ["circle","circle"]
     ))

     # step 0: full search space
     x_col = "L1 Usage"
     y_col = "SSR Configs"
     special_figs.append(
         scatterWithFlatColor(
            full,
            x_col,
            y_col,
            "gray",
            minimal_hover,
            "0) full search space",
            "symbolMarker",
        )
     )


    # step 1: pruned search space
     x_col = "L1 Usage"
     y_col = "SSR Configs"
     special_figs.append(
        scatterWithColor(
            analyzed,
            x_col,
            y_col,
            "SSR Configs",
            minimal_hover,
            "0.1) Full search space (multicolor points are analyzed by our model)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        full,
        x_col,
        y_col,
        "gray",
        "circle",
        minimal_hover,
        "full search space"
    )
     
     # step 3: timed vs untimed points
     x_col = "SSR Configs"
     y_col = "L1 Usage"
     special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            minimal_hover,
            "1) Pruned Search Space (multicolor points are timed)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        ut,
        x_col,
        y_col,
        "gray",
        "circle",
        minimal_hover,
        "untimed"
    )
     
     # step 4: order by n'/k
     x_col = "Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"
     special_figs.append(
        scatterWithFlatColor(
            timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "1.1) Pruned Search Space (black points are timed); sort by n'/k ratio",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        ut,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed"
    )
     
     # step 5: identify nice m remainders
     nice_timed=timed[timed["niceMRem"]]
     nice_ut=ut[ut["niceMRem"]]
     mean_timed=timed[timed["niceMRem"]==False]
     mean_ut=ut[ut["niceMRem"]==False]
     x_col = "Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"
     special_figs.append(
        scatterWithFlatColor(
            nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "1.2) Pruned Search Space (black points are timed); identify worst case CL boundary tiles (marked with red x)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        nice_ut,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed w/ nice m remainder"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        mean_ut,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "untimed w/ worst case m remainder"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        mean_timed,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "timed w/ worst case m remainder"
    )
     
     # step 5: Keep nice m-remainders (prune out CL boundary tiles with 0 < m-dim < 8)
     x_col = "Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"
     special_figs.append(scatterWithFlatColorSymbol(
         nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "2) prune out worst-case CL boundary tiles; only nice ones remain.",
            "mRem",
     ))
     mRem_shape_map = {"zero": "circle", "divisBy8": "triangle-up","mean":"diamond"}
     for status_name, group_df in nice_ut.groupby("howNice"):
          special_figs[-1].add_scatter(
               x=group_df[x_col],
               y=group_df[y_col],
               mode="markers",
               name=status_name,  # Sets the legend label
               marker=dict(
                    symbol=mRem_shape_map[status_name], color="gray"  
               ),
          )
     # step 6: tie-break with FMADDMULS per Core
#      x_col = "Avg n'_sz / k_size"
#      y_col = "Global Sim E2E_dma"
#      special_figs.append(scatterWithColorSymbol(
#          nice_timed,
#             x_col,
#             y_col,
#             "FMADDsMULsPerCore",
#             hover_data,
#             "Take left most. Tie break by maximizing FMADDS per core. SQUARES are untimed.",
#             "mRem",
#      ))
#      addScatterFlatColorMarker(
#         special_figs[-1],
#         nice_ut,
#         x_col,
#         y_col,
#         "gray",
#         "square",
#         hover_data,
#         "untimed"
#     )
     

     nice_timed_reduced = nice_timed[hover_data]
     nice_ut_reduced = nice_ut[hover_data]     
     combined = pd.concat([nice_timed_reduced,nice_ut_reduced],axis=0, ignore_index=True)
   
     x_col = "Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         combined,
            x_col,
            y_col,
            "FMADDsMULsPerCore",
            hover_data,
            "3) Take left most. Tie break by maximizing FMADDS per core. SQUARES are untimed.",
            "timedData",
            ["circle","square"]
     ))

     # more figs

     x_col = "Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"
     more_figs.append(scatterWithFlatColorSymbol(
         nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "2) Worst-case CL boundary tiles pruned out; only nice ones remain. RED LINE marks n/k = 1",
            "mRem",
     ))
     mRem_shape_map = {"zero": "circle", "divisBy8": "triangle-up","mean":"diamond"}
     for status_name, group_df in nice_ut.groupby("howNice"):
          more_figs[-1].add_scatter(
               x=group_df[x_col],
               y=group_df[y_col],
               mode="markers",
               name=status_name,  # Sets the legend label
               marker=dict(
                    symbol=mRem_shape_map[status_name], color="gray"  
               ),
          )
     more_figs[-1].add_vline(x=1.0, line_width=2, line_dash="dash", line_color="red")

     # prune to less than n/k = 1
     #combined = combined[combined["Avg n'_sz / k_size"] < 1.0]
     nice_timed_reduced_lt1 = nice_timed_reduced[nice_timed_reduced["Avg n'_sz / k_size"] < 1.0]
     nice_timed_reduced_gte1=nice_timed_reduced[nice_timed_reduced["Avg n'_sz / k_size"] >= 1.0]
     nice_ut_reduced_lt1 = nice_ut_reduced[nice_ut_reduced["Avg n'_sz / k_size"] < 1.0]
     #jugaad
     more_figs.append(genResultGraphPDF(jugaadTitle(timed),nice_timed_reduced_lt1,nice_ut_reduced_lt1,nice_timed_reduced_gte1,hover_data))



     return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)

#      more_figs[-1].update_layout(
#     # 1. Target the Title specifically
#     title=dict(
#         font=dict(
#         family="CMU Serif",  # Tells Plotly to search your system for Computer Modern
#         size=12,
#         color="black"
#     )
#     ),
#     # 2. Target the X-Axis Title
#     xaxis=dict(
#         title=dict(
#             font=dict(
#         family="CMU Serif",  # Tells Plotly to search your system for Computer Modern
#         size=12,
#         color="black"
#     )
#         )
#     ),
#     yaxis=dict(
#         title=dict(
#             font=dict(
#         family="CMU Serif",  # Tells Plotly to search your system for Computer Modern
#         size=12,
#         color="black"
#     )
#         )
#     ),
#     # 3. Target the Legend text
#     legend=dict(
#         font=dict(
#         family="CMU Serif",  # Tells Plotly to search your system for Computer Modern
#         size=6,
#         color="white"
#     )
#     ),
#     template="plotly_white", 
#     width=600, 
#     height=400
# )

# def genResultGraphPDF(title,timed, untimed, recentlyPruned,hover_data):
#     if len(timed)<5:
#         rp_max=recentlyPruned["Time (cycles)"].max()
#         tm_max=timed["Time (cycles)"].max()
#         newFakeTime = max(tm_max,rp_max)
#         if newFakeTime==rp_max:
#             newFakeTime = rp_max*1.25
#     else:
#         newFakeTime = timed["Time (cycles)"].max()
#     # customize the height of the untimed points
#     untimed = untimed.copy(deep=True)
#     untimed["Time (cycles)"]=newFakeTime    
#     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
#     y_col = "Time (cycles)"#"Global Sim E2E_dma"
#     fig=scatterWithColorSymbol(
#          timed,
#             x_col,
#             y_col,
#             "mRem",
#             hover_data,
#             f"<b>{title}</b>",
#             "timeout",
#             ["circle","cross"]
#     )
#     addScatterFlatColorMarker(
#         fig,
#         untimed,
#         x_col,
#         y_col,
#         "gray",
#         "square",
#         hover_data,
#         "untimed w/ nice m remainder, n/k < 1"
#         )
#     if(len(timed)<5):
#         addScatterFlatColorMarker(
#         fig,
#         recentlyPruned,
#         x_col,
#         y_col,
#         "gray",
#         "circle-open",
#         hover_data,
#         "untimed w/ nice m remainder, n/k < 1"
#         )

#     fig.update_traces(showlegend=False)

#     #fig.write_image(f"out/{title}.pdf", width=1200, height=800, scale=3)
#     # I have a 7x10 paper, so 1/3 of the width is approx 2.3 inches
#     # let's try 600 dpi for the scale
#     # plotly graph is 7 wide and 8 tall
#     dpi = 72#300
#     heightPx=4*dpi#(3.2/8*7)*dpi
#     widthPx=6*dpi#3.2*dpi
#     # 1. Apply your base template
#     fig.update_layout(
#         template="plotly_white",
#         title=dict(x=0.5, xanchor="center"),
#         font=dict(family="CMU Serif, Computer Modern, Serif", size=10)
#     )

#     # 2. Add the border lines to the axes
#     fig.update_xaxes(
#         showline=True,       # Turn on the axis line
#         linewidth=1,         # Thickness of the border
#         linecolor="black",   # Color of the border (matches standard academic plots)
#         mirror=True,         # CRITICAL: Mirrors the line to the top of the graph box
#         gridcolor="lightblue" # Keeps your light blue grid lines intact
#     )

#     fig.update_yaxes(
#         showline=True,
#         linewidth=1,
#         linecolor="black",
#         mirror=True,         # CRITICAL: Mirrors the line to the right side of the graph box
#         gridcolor="lightblue"
#     )

# # Adjust your physical PDF size and export
#     fig.update_layout(width=widthPx, height=heightPx, margin=dict(l=30, r=20, t=35, b=30))
#     # Scale it by 3x upon export to achieve 300 DPI crispness.
#     # This keeps the text, lines, and markers perfectly proportioned!

#     fig.write_image(f"out/{title}.pdf", scale=4)
#     #fig.write_image(f"out/{title}.pdf", width=widthPx, height=heightPx)
#     return fig