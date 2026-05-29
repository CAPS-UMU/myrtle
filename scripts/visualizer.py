import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

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
        symbol=marker,
        symbol_sequence=markerSequence,
        hover_data=hover_data,  # Show these columns on hover
        title=f"{title} <b>{x_col} vs {y_col}</b>",
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
    avgTime = sum(df_t["Global Sim E2E_dma"].values) / len(df_t["Global Sim E2E_dma"].values)
    df_ut["Global Sim E2E_dma"] = avgTime
    df_ut["dma"] = avgTime
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
     timed["Overlap Stall Time Per Core"] = timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
     timed["mRem"] = timed["M"] % timed["m"]
     timed["niceMRem"] = timed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
     timed["howNice"] = timed["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
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
        "remainderTiles",
        "Global Sim E2E_dma",
        "Total CL Tiles",
        "Total CC Tiles",
        "Overlap Stall Time Per Core",
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
     analyzed["niceMRem"] = analyzed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
     analyzed["howNice"] = analyzed["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
     analyzed["Total CC Tiles"] = analyzed["SSR Config Count"]
     analyzed["FMADDsMULsPerCore"] = analyzed["FMADDsMULs"] / analyzed["Total CC Tiles"]
     analyzed["Overlap Stall Time Per Core"] = -1
     analyzed = addFakeTime(analyzed,timed)
     analyzed["Y/X"]=timed["Avg n'_sz / k_size"] * timed["Avg A'"]
     analyzed["timedData"] = False
     
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
     nice_ut_reduced_lt1 = nice_ut_reduced[nice_ut_reduced["Avg n'_sz / k_size"] < 1.0]

     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"#"Global Sim E2E_dma"
     more_figs.append(scatterWithColorSymbol(
         nice_timed_reduced_lt1,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "3) Prune to n/k < 1;  Maximize FMADDS per core (take right most). TIE BREAK with smaller mRem size.",
            "timedData",
            ["circle","circle"]
     ))
     addScatterFlatColorMarker(
        more_figs[-1],
        nice_ut_reduced_lt1,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed w/ nice m remainder, n/k < 1"
    )
 

     # x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     # y_col = "Avg n'_sz / k_size"#"Global Sim E2E_dma"
     # more_figs.append(scatterWithColorSymbol(
     #     combined,
     #        x_col,
     #        y_col,
     #        "Global Sim E2E_dma",
     #        hover_data,
     #        "Another view: Take left most. Tie break by maximizing FMADDS per core",
     #        "timedData",
     #        ["circle","square"]
     # ))

#      special_figs[-1].update_layout(
#     legend=dict(
#         orientation="v",  # Forces vertical layout
#         itemwidth=30,  # Gives the markers more breathing room from the text
#         tracegroupgap=10,  # Adds vertical space between different trace groups
#         yanchor="top",
#         y=1,  # Keeps it aligned at the top right
#         xanchor="left",
#         x=1.02,  # Pushes the legend slightly outside the plot area so it doesn't overlap the grid
#     )
# )
     



#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned[pruned["niceMRem"]== True],
#             x_col,
#             y_col,
#             "FMADDsMULsPerCore",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; myrtle after non-squares: prune out triangles, then take leftmost. THEN can we maximize by FMADDS?",
#              "mRem",
#         )

#     # here we prune out bad mrems
#     x_col = "Avg n'_sz / k_size"
#     y_col = "Global Sim E2E_dma"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned[pruned["niceMRem"]== True],
#             x_col,
#             y_col,
#             "FMADDsMULsPerCore",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; myrtle after non-squares: prune out triangles, then take leftmost. THEN can we maximize by FMADDS?",
#              "mRem",
#         )
   
#      x_col = "L1 Usage"
#      y_col = "SSR Configs"
#      special_figs.append(
#         scatterWithColor(
#             timed,
#             x_col,
#             y_col,
#             "SSR Configs",
#             hover_data,
#             "Reality Check: timed and untimed points",
#             "symbolMarker",
#         )
#     )
#      addScatterFlatColorMarker(
#         special_figs[-1],
#         ut,
#         x_col,
#         y_col,
#         "gray",
#         "triangle-up",
#         hover_data,
#         "untimed points"
#     )
#      addScatterFlatColorMarker(
#         special_figs[-1],
#         pruned,
#         x_col,
#         y_col,
#         "gray",
#         "square",
#         minimal_hover,
#         "pruned out by SSR Config pruning"
#     )

#      x_col = "FakeNN JSON Name"
#      y_col = "Global Sim E2E_dma"
#      special_figs.append(
#         scatterWithColor(
#             timed,
#             x_col,
#             y_col,
#             "SSR Configs",
#             hover_data,
#             "timed data only",
#             "symbolMarker",
#         )
#     )

#     x_col = "Global Sim E2E_dma"
#     y_col = "FMADDsMULsPerCore"
#     special_figs.append(
#         scatterWithColor(
#             timed,
#             x_col,
#             y_col,
#             "SSR Configs",
#             hover_data,
#             "timed data only",
#             "symbolMarker",
#         )
#     )

#     x_col = "FMADDsMULsPerCore"
#     y_col = "Global Sim E2E_dma"
#     special_figs.append(
#         scatterWithColor(
#             timed,
#             x_col,
#             y_col,
#             "SSR Configs",
#             hover_data,
#             "timed data only",
#             "symbolMarker",
#         )
#     )

#     # size and number not same
#     x_col = "Avg CC Tile Size"
#     y_col = "SSR Configs"
#     special_figs.append(
#         scatterWithColor(
#             timed,
#             x_col,
#             y_col,
#             "Global Sim E2E_dma",
#             hover_data,
#             "Tile Size vs Arithmetic Intensity",
#             "symbolMarker",
#         )
#     )
#     addScatterFlatColorMarker(
#         special_figs[-1],
#         ut,
#         x_col,
#         y_col,
#         "gray",
#         "triangle-up",
#         hover_data,
#         "untimed remainders"
#     )

#     x_col = "SSR Configs"
#     y_col = "Global Sim E2E_dma"
#     special_figs.append(
#         prunedScatter(
#             timed,
#             x_col,
#             y_col,
#             "Avg n'_sz / k_size",
#             hover_data,
#             "timed remainders",
#             prunePoint,
#             "symbolMarker",
#         )
#     )
#     addScatterFlatColorMarker(
#         special_figs[-1],
#         ut,
#         x_col,
#         y_col,
#         "gray",
#         "triangle-up",
#         hover_data,
#         "untimed remainders",
#     )
    
#     x_col = "SSR Configs"
#     y_col = "Global Sim E2E_dma"
#    # prunePoint = 15984
#     pruned = timed[timed["SSR Configs"]<= prunePoint]
#     ut_pruned = ut[ut["SSR Configs"]<= prunePoint]
#     special_figs.append(
#         scatterWithColor(
#             pruned,
#             x_col,
#             y_col,
#             "Avg n'_sz / k_size",#"mRem",
#             hover_data,
#             f"After pruning to <= {prunePoint} SSR Configs",
#             "symbolMarker",
#         )
#     )
#     addScatterFlatColorMarker(
#         special_figs[-1],
#         ut_pruned,
#         x_col,
#         y_col,
#         "gray",
#         "square",
#         hover_data,
#         "untimed remainders",
#     )

#     x_col = "SSR Configs"
#     y_col = "Avg CC Tile Size"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "absoluteRank",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; How does CC tile size relate to FMADDs per core?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

#     x_col = "SSR Configs"
#     y_col = "FMADDsMULsPerCore"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "absoluteRank",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; How do SSR configs relate to FMADDs per core?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

#     x_col = "Avg CC Tile Size"
#     y_col = "FMADDsMULsPerCore"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "absoluteRank",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; How does CC tile size relate to FMADDs per core?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

    
#     x_col = "FMADDsMULsPerCore"
#     y_col = "Avg CC Tile Size"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "absoluteRank",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; How does CC tile size relate to FMADDs per core?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

#     x_col = "FakeNN JSON Name"
#     y_col = "Avg n'_sz / k_size"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "FMADDsMULsPerCore",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; prune out triangles, then take leftmost. THEN can we maximize by FMADDS?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

#     # here we prune out bad mrems
#     x_col = "Avg n'_sz / k_size"
#     y_col = "Global Sim E2E_dma"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned[pruned["niceMRem"]== True],
#             x_col,
#             y_col,
#             "FMADDsMULsPerCore",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; myrtle after non-squares: prune out triangles, then take leftmost. THEN can we maximize by FMADDS?",
#              "mRem",
#         )
#     special_figs.append(myFig) #pruned[pruned["niceMRem"]== True]
#     addScatterFlatColorMarker(
#         special_figs[-1],
#         ut_pruned[ut_pruned["niceMRem"]== True],
#         x_col,
#         y_col,
#         "gray",
#         "square",
#         hover_data,
#         "untimed remainders"
#     )

    

#     # print(pruned[pruned["FakeNN JSON Name"]== "384x384x384w21-24-39"][["Avg n'_sz / k_size","mRem"]]) #0.626573
#     # print(pruned[pruned["FakeNN JSON Name"]== "384x384x384w21-24-39"][["Avg n'_sz / k_size","mRem"]].iloc(0))

#     #               #,"Avg n'_sz / k_size"]])
#     # print(pruned[pruned["Avg n'_sz / k_size"] == "0.626573"][["FakeNN JSON Name","Avg n'_sz / k_size","mRem"]])
 

#     x_col = "Avg n'_sz / k_size"
#     y_col = "Global Sim E2E_dma"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "mRem",
#             hover_data,
#             f"pruned to SSR configs <= {prunePoint}; myrtle: take leftmost, then the tie break by first preferring circle over triangle",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

#     x_col = "Overlap Stall Time Per Core"
#     y_col = "Global Sim E2E_dma"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "Avg n'_sz / k_size",
#             hover_data,
#             "does ordering by overlap stall time help?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)

#     x_col = "mRem"
#     y_col = "Overlap Stall Time Per Core"
#     pruned.sort_values(by="niceMRem",ascending=True)
#     myFig = scatterWithColorSymbol(
#             pruned,
#             x_col,
#             y_col,
#             "Global Sim E2E_dma",
#             hover_data,
#             "how is mRem size related to overlap stall time?",
#             "niceMRem",
#         )
#     special_figs.append(myFig)
    

#     # more figures
     
#     more_figs.append(
#         px.bar(
#             timed,
#             x="absoluteRank",
#             y=["dma"],  # Pass both column names here
#             # barmode='group',         # Keeps them side-by-side
#             title="Effect of Remainder Tiles divisible by 8 on Execution Time?",
#             color="niceMRem",
#             labels={
#                 "value": "Time (cycles)",
#                 "variable": "Metric",
#             },  # 'value' and 'variable' are default labels for lists
#             # template='plotly_dark'
#         )
#     )

     return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)