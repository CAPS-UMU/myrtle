import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import math
import re
# theColorBar="ylorrd_r"
theColorBar="haline"

subfigEpi=r"""
            \bottomrule
        \end{tabular}
         \Description{todo}
        \label{subfig:table_a}
    \end{subfigure}
"""
def subFigPro(title):
    beg=r"""
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \textbf{"""
    end=r"""} \\[0.5ex] % Title row over the table
        \begin{tabular}{ccc}
            \toprule
            m-n-k & Cycles & \% from Best \\
            \midrule
        """
    return beg + title + end
def tableRow(mnk,cycles, diff):
    return f"    {mnk}&         {cycles}&  {diff:.2f} \\\\"
def df_to_latex_rows(df):
    str = ""
    for row in df[["JSON Name","Time (cycles)","diff"]].iterrows():
        # print(row[1]["JSON Name"])
        # print(row[1]["Time (cycles)"])
        # print(row[1]["diff"])
        str = str + tableRow(row[1]["JSON Name"],row[1]["Time (cycles)"],row[1]["diff"]) + "\n"
    return str


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

def jugaadTitleQ(df):
    df = df.reset_index(drop=True)
    M=int(df["M"][0])
    N=int(df["N"][0])
    K=int(df["K"][0])
    dims=f"{M}x{N}x{K}"
    return f"VecMatT {dims}"


def genResultGraphPDF(title,timed, untimed, recentlyPruned,hover_data,color="n / k"):
    colorCol=color
    colorMin=timed[colorCol].min()
    colorMax=timed[colorCol].max()
    # print(f"color min is {colorMin} with type{type(colorMin)}")
    # print(f"color max is {colorMax} with type{type(colorMax)}")
    if len(timed)<5:
        rp_max=recentlyPruned["Time (cycles)"].max()
        tm_max=timed["Time (cycles)"].max()
        newFakeTime = max(tm_max,rp_max)
        if newFakeTime==rp_max:
            newFakeTime = rp_max*1.1
        colorMin=min(colorMin,recentlyPruned[colorCol].min())
        colorMax=max(colorMax,recentlyPruned[colorCol].max())
    else:
        newFakeTime = timed["Time (cycles)"].max()*1.01
    # customize the height of the untimed points
    untimed = untimed.copy(deep=True)
    untimed["Time (cycles)"]=newFakeTime    
    x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
    y_col = "Time (cycles)"#"Global Sim E2E_dma"
    fig=scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            colorCol,
            hover_data,
            "testing short title",
            "timeout",
            ["circle","cross"]    )
    matching_rows = timed.loc[timed["timeout"] == True, "dma"]
    if not matching_rows.empty:
        # plot timeout threshold
        dma=matching_rows.values[0]
        fig.add_hline(y=dma, line_width=0.75, line_dash="dash", line_color="black",layer="below")
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
            text=title,
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
    dpi = 72 #300
    widthPx=6*dpi
    heightPx=4*dpi
  
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
    template="plotly_white")
    # Scale it by 3x upon export to achieve 300 DPI crispness.
    # This keeps the text, lines, and markers perfectly proportioned!
    fig.update_layout(
    coloraxis=dict(
        cmin=colorMin,         # Force the scale to start exactly at 0
        cmax=colorMax        # Optional: You can also hardcode the maximum if you want
    )
    )

    fig.write_image(f"out/{title}.pdf", scale=1)
    #fig.write_image(f"out/{title}.pdf", width=widthPx, height=heightPx)
    return fig

def genResultGraphQPDF(title,timed, recentlyPruned,hover_data,color="fmaddsPerCore"):
    colorCol=color
    colorMin=timed[colorCol].min()
    colorMax=timed[colorCol].max()
    x_col = "Regular Loads"#"Avg n'_sz / k_size"
    y_col = "Time (cycles)"#"Global Sim E2E_dma"
    # only graph bottom half of regular loads to improve visibility in left corner
    sorted =timed.sort_values("Regular Loads", ascending=True)
   # print(sorted[["FakeNN JSON Name","Regular Loads"]])
    pruned = sorted.iloc[range(0, int(len(sorted)*0.75))]
   # print(pruned[["FakeNN JSON Name","Regular Loads"]])
    #l1_thresh = min(best_half_l1["L1 Usage"].values)

    fig=scatterWithColorSymbol(
         timed,#pruned,
            x_col,
            y_col,
            colorCol,
            hover_data,
            "testing short title",
            "timeout",
            ["circle","cross"]
    )
    # addScatterFlatColorMarker(
    #     fig,
    #     recentlyPruned,
    #     x_col,
    #     y_col,
    #     "gray",
    #     "circle-open",
    #     hover_data,
    #     "pruned out by SSR config threshold"
    #     )
    fig.update_layout(
        title=dict(
            text=title,
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
    dpi = 72 #300
    widthPx=6*dpi
    heightPx=4*dpi
  
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
    template="plotly_white")
    # Scale it by 3x upon export to achieve 300 DPI crispness.
    # This keeps the text, lines, and markers perfectly proportioned!
    fig.update_layout(
    coloraxis=dict(
        cmin=colorMin,         # Force the scale to start exactly at 0
        cmax=colorMax        # Optional: You can also hardcode the maximum if you want
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

def printFinalRanking(title,df,colorCol):
    print("\tFinal ranking:")
    df=df.sort_values("Time (cycles)",ascending=True)
    df = df.reset_index(drop=True)
    best=df["Time (cycles)"][0]
    print(f"best observed: {best} cycles")
    df=df.sort_values("FMADDsMULsPerCore",ascending=False)
    df["diff"] = df["Time (cycles)"].apply(lambda x: (x - best)/best * 100)
 #   print(df[["JSON Name","FMADDsMULsPerCore","timeout","Time (cycles)","diff"]][0:9])
    print("--------------------")
    latexList=[subFigPro(title)]
    pointsPrinted = 0
    for fmadds, group_df in df.groupby("FMADDsMULsPerCore",sort=False):
          if pointsPrinted < 5:
               print(f"FMADDS: {fmadds} w/ len {len(group_df)}")
               pointsPrinted = pointsPrinted + len(group_df)
               sorted = group_df.sort_values(colorCol,ascending=True)
               print(sorted[["JSON Name","timeout","timed",colorCol,"Time (cycles)","diff"]])
               latexList.append(df_to_latex_rows(sorted))
    
    print("-------------------- FOR LATEX")
    latexList.append(subfigEpi)
    print(''.join(latexList))
#     pointsPrinted = 0
#     for fmadds, group_df in df.groupby("FMADDsMULsPerCore",sort=False):
#           if pointsPrinted < 5:
#                print(f"FMADDS: {fmadds} w/ len {len(group_df)}")
#                pointsPrinted = pointsPrinted + len(group_df)
#                sorted = group_df.sort_values(colorCol,ascending=True)
#                print(sorted[["JSON Name","Time (cycles)","diff"]])
    print("-------------- ^^^^ ------------\n")

def printFinalRankingQ(title,df,colorCol):
    print("\tFinal ranking:")
    df=df.sort_values("Time (cycles)",ascending=True)
    df = df.reset_index(drop=True)
    best=df["Time (cycles)"][0]
    print(f"best observed: {best} cycles")
    df=df.sort_values("Regular Loads",ascending=True)
    df["diff"] = df["Time (cycles)"].apply(lambda x: (x - best)/best * 100)
 #   print(df[["JSON Name","FMADDsMULsPerCore","timeout","Time (cycles)","diff"]][0:9])
    print("--------------------")
    latexList=[subFigPro(title)]
    pointsPrinted = 0
    for regLoads, group_df in df.groupby("Regular Loads",sort=False):
          if pointsPrinted < 5:
               print(f"REGULAR LDS: {regLoads} w/ len {len(group_df)}")
               pointsPrinted = pointsPrinted + len(group_df)
               sorted = group_df.sort_values(colorCol,ascending=False)
               print(sorted[["JSON Name",colorCol,"Time (cycles)","diff"]])               
               latexList.append(df_to_latex_rows(sorted))
    
    print("-------------------- FOR LATEX")
    latexList.append(subfigEpi)
    print(''.join(latexList))
#     pointsPrinted = 0
#     for fmadds, group_df in df.groupby("FMADDsMULsPerCore",sort=False):
#           if pointsPrinted < 5:
#                print(f"FMADDS: {fmadds} w/ len {len(group_df)}")
#                pointsPrinted = pointsPrinted + len(group_df)
#                sorted = group_df.sort_values(colorCol,ascending=True)
#                print(sorted[["JSON Name","Time (cycles)","diff"]])
    print("-------------- ^^^^ ------------\n")

def visualizePruning(timed, analyzed, full, titleOfWebpage):
     timed["Total CC Tiles"] = timed["SSR Config Count"]
     timed["FMADDsMULsPerCore"] = timed["FMADDsMULs"] / timed["Total CC Tiles"]
     timed["1/FMADDS"]=1/timed["FMADDsMULsPerCore"]
     timed["Overlap Stall Time Per Core"] = timed["Overlap Stall Time Total"] / timed["Total CC Tiles"]
     timed["mRem"] = timed["M"] % timed["m"]
     timed["1/mRem"]=1/timed["mRem"]
     timed["niceM"] = timed["m"].apply(lambda r: True if r % 8 == 0 else False)
     timed["niceMRem"] = timed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
     timed["bothNice"]=timed[["niceM", "niceMRem"]].apply(lambda r: True if r["niceM"] & r["niceMRem"] else False,axis=1)
     timed["howNice"] = timed["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
     timed["hypotenuse"] = timed[["mRem","1/FMADDS"]].apply(lambda x: math.sqrt(x["mRem"]*x["mRem"]+x["1/FMADDS"]*x["1/FMADDS"]),axis=1)
     timed["Time (cycles)"]=timed["Global Sim E2E_dma"]
     timed["n / k"]=timed["Avg n'_sz / k_size"]
     def parseDimM(nm):
         if nm=="timeout":
            return -1
         else:
            expNameRegex = re.compile(
                r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
            )
         return expNameRegex.search(nm).groups()[0]
     def parseDimN(nm):
         if nm=="timeout":
            return -1
         else:
            expNameRegex = re.compile(
                r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
            )
         return expNameRegex.search(nm).groups()[1]
     def parseDimK(nm):
        if nm=="timeout":
            return -1
        else:
            expNameRegex = re.compile(
                r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
            )
            return expNameRegex.search(nm).groups()[2]

     timed["M"] = timed["FakeNN JSON Name"].apply(parseDimM)
     timed["N"] = timed["FakeNN JSON Name"].apply(parseDimN)
     timed["K"] = timed["FakeNN JSON Name"].apply(parseDimK)    
   
     analyzed["mRem"] = analyzed["M"] % analyzed["m"]
     analyzed["1/mRem"]=1/analyzed["mRem"]
     analyzed["niceM"] = timed["m"].apply(lambda r: True if r % 8 == 0 else False)
     analyzed["niceMRem"] = analyzed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
     analyzed["howNice"] = analyzed["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
     analyzed["bothNice"]=analyzed[["niceM", "niceMRem"]].apply(lambda r: True if r["niceM"] and r["niceMRem"] else False,axis=1)
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
     analyzed["n / k"]=analyzed["Avg n'_sz / k_size"]
     
     timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
     timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
     timed = timed.sort_values(by="symbolMarker", ascending=True)
     timed["flatColor"] = "pink"
     timed["timedData"] = True

     full["SSR Configs"] = full["SSR Config Count"]
     full["L1 Usage"] = full["Space Needed in L1"]
     full = addFakeTime(full,timed)

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
        "n / k",
        "Avg n'_sz / k_size",
        "timedData",
     ]
     minimal_hover = [
        "JSON Name",       
        "SSR Configs",
        "L1 Usage",       
     ]
     
     special_figs = []
     more_figs = []

     special_figs,more_figs = pruneApproach1(timed,analyzed,full)
    # special_figs=special_figs+more_figs

     #more_figs = pruneApproach2(timed,analyzed,full)
     
     return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)

def pruneApproach2(timed, analyzed, full):
     prunePoint = ssr_prune_frac(full,3)
    # we assume untimed points are a subset of the pruned search space
   #  print("Missing values in M:", analyzed['M'].isna().sum())
   #  print("Missing values in m:", analyzed['m'].isna().sum())
   #  print("analyzed: NaN for NiceM:", analyzed['niceM'].isna())
     #print(analyzed[["FakeNN JSON Name","M","m","niceM"]])
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
   #  print("ut: NaN for NiceM:", ut['niceM'].isna())
    # print(ut[["FakeNN JSON Name","M","m","niceM"]])
     pruned = full[full["SSR Config Count"] < prunePoint]
     print("prune approach 2 statistics:")
     print(f"full: {len(full)} pruned:{len(pruned)} % analyzed:{len(pruned)/len(full)}")
     print(f"ann: {len(analyzed)} pruned: {len(pruned)} timed: {len(timed)} % timed:{len(timed)/len(pruned)}")
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
        "n / k",
        "Avg n'_sz / k_size",
        "timedData",
     ]
     minimal_hover = [
        "JSON Name",       
        "SSR Configs",
        "L1 Usage",       
     ]
     special_figs=[]
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
            pruned,
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
     
     # step 5: identify nice m remainders and keep 'em, also keep nice m sizes

     niceM_timed=timed[timed["niceM"]==True]
     niceMrem_timed=timed[timed["niceMRem"]==True] 
     niceM_ut=ut[ut["niceM"]==True]
     niceMrem_ut=ut[ut["niceMRem"]==True]

     nice_timed=timed[timed["bothNice"]==True]
     nice_ut=ut[ut["bothNice"]==True]

     meanM_timed=timed[timed["niceM"]==False]
     meanMRem_timed=timed[timed["niceMRem"]==False] 
     meanM_ut=ut[ut["niceM"]==False]
     meanMRem_ut=niceM_ut[niceM_ut["niceMRem"]==False]

     
     x_col = "Global Sim E2E_dma" #"Avg n'_sz / k_size"
     y_col = "mRem"
     special_figs.append(
        scatterWithFlatColor(
            nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "1.2) Pruned Search Space (black points are timed); prune out all m boundary tiles (blue x); worst case CL boundary tiles (red x)",
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
        "untimed w/ nice m "
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        meanM_ut,
        x_col,
        y_col,
        "blue",
        "x",
        hover_data,
        "untimed w/ m not evenly divided by 8"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        meanM_timed,
        x_col,
        y_col,
        "blue",
        "x",
        hover_data,
        "timed w/ m not evenly divided by 8"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        meanMRem_ut,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "untimed w/ worst case m remainder"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        meanMRem_timed,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "timed w/ worst case m remainder"
    )
     
     x_col = "Time (cycles)"
     y_col = "Avg n'_sz / k_size"#"Avg n'_sz / k_size"
     special_figs.append(
        scatterWithColor(
            nice_timed,
            x_col,
            y_col,
            "Avg n'_sz / k_size",
            hover_data,
            "1.2) Pruned by n/k < 1 (threshold line in green)",
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
        "untimed w/ nice m "
    )
     special_figs[-1].add_hline(y=1.0, line_width=2, line_dash="dash", line_color="green")

     nice_timed_reduced = nice_timed[hover_data]
     nice_ut_reduced = nice_ut[hover_data]  

     nice_timed_lt1=nice_timed_reduced[nice_timed_reduced["Avg n'_sz / k_size"]<1.0]
     nice_ut_lt1=nice_ut_reduced[nice_ut_reduced["Avg n'_sz / k_size"]<1]
     nice_timed_gte1 = nice_timed_reduced[nice_timed_reduced["Avg n'_sz / k_size"]<=1.0]
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"
     special_figs.append(
        scatterWithColor(
            nice_timed_lt1,
            x_col,
            y_col,
            "mRem",#"Avg n'_sz / k_size",
            hover_data,
            "1.2) Pruned Search Space (black points are timed); identify worst case CL boundary tiles (marked with red x)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        nice_ut_lt1,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed w/ nice m "
    )
     
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         nice_timed_lt1,
            x_col,
            y_col,
            "n / k",
            hover_data,
            "web version of result graph",
            "timeout",
            ["circle","cross"]
     ))
     addScatterFlatColorMarker(
        special_figs[-1],
        nice_ut_lt1,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed w/ nice m AND mRem "
    )
     
     
     #result graph
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
     resultGraph = genResultGraphPDF(jugaadTitle(timed),nice_timed_lt1,nice_ut_lt1,nice_timed_gte1,hover_data,"n / k")
     printFinalRanking(jugaadTitle(timed),nice_timed_lt1,"n / k")
     special_figs.append(resultGraph)     
     return special_figs

def printMethodologyStats(full, pruned, timed):
    print(f"full ss has size {len(full)}")
    print(f"pruned has size {len(pruned)}")
    print(f"timed has size {len(timed)}")
    print("what percentage of SPM used in timed points vs pruned points?")

def pruneApproach1(timed, analyzed, full):
     prunePoint = ssr_prune_frac(full,3)
    # we assume untimed points are a subset of the pruned search space
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
     pruned = full[full["SSR Config Count"] < prunePoint]
     printMethodologyStats(full, pruned, timed)
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
     special_figs=[]
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
            pruned,
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
     nice_timed_reduced = nice_timed[hover_data].copy()
     nice_ut_reduced = nice_ut[hover_data].copy()
     nice_timed_reduced["timed"]=True
     nice_ut_reduced["timed"] = False
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
     more_figs = []



     # more figs
     y_col = "mRem"
     x_col = "Global Sim E2E_dma"
     timed.sort_values("howNice")
     more_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "Is it ever worth it to includle boundary tiles? Emily, fix marker symbols (not working right)",
            "howNice",
            ["circle","triangle-up","x",]
     ))
     addScatterFlatColorMarker(
        more_figs[-1],
        ut,
        x_col,
        y_col,
        "gray",
        "square",
        hover_data,
        "untimed"
    )

    # print(f"before appending: len of more_figs is {len(more_figs)}")
     x_col = "Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"
     mRem_shape_map = {"zero": "circle", "divisBy8": "triangle-up","mean":"diamond"}
     more_figs.append(scatterWithFlatColorSymbol(
         nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "2) Worst-case CL boundary tiles pruned out; only nice ones remain. RED LINE marks n/k = 1",
            "howNice",
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
     for status_name, group_df in nice_timed.groupby("howNice"):
          more_figs[-1].add_scatter(
               x=group_df[x_col],
               y=group_df[y_col],
               mode="markers",
               name=status_name,  # Sets the legend label
               marker=dict(
                    symbol=mRem_shape_map[status_name], color="black"  
               ),
          )
     more_figs[-1].add_vline(x=1.0, line_width=2, line_dash="dash", line_color="red")
    
     # prune to less than n/k = 1
     #combined = combined[combined["Avg n'_sz / k_size"] < 1.0]
     nice_timed_reduced_lt1 = nice_timed_reduced[nice_timed_reduced["Avg n'_sz / k_size"] < 1.0]
     nice_timed_reduced_gte1=nice_timed_reduced[nice_timed_reduced["Avg n'_sz / k_size"] >= 1.0]
     nice_ut_reduced_lt1 = nice_ut_reduced[nice_ut_reduced["Avg n'_sz / k_size"] < 1.0]
     
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
    # print(f"before appending: len of more_figs is {len(more_figs)}")
     more_figs.append(scatterWithColorSymbol(
         nice_timed_reduced_lt1,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "testing short title",
            "timeout",
            ["circle","cross"]
     ))
     
     #result graph
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
     resultGraph = genResultGraphPDF(jugaadTitle(timed),nice_timed_reduced_lt1,nice_ut_reduced_lt1,nice_timed_reduced_gte1,hover_data,"mRem")
     print(len(nice_timed_reduced_lt1.columns))
     print(len(nice_ut_reduced_lt1.columns))
     combined=pd.concat([nice_timed_reduced_lt1,nice_ut_reduced_lt1])
     printFinalRanking(jugaadTitle(timed),combined,"mRem")
     more_figs.append(resultGraph)   
     return special_figs,more_figs


def unrollAndJamFactor(tobeUnrolledDim):
        options = [7, 6, 5, 4, 3, 2]
        factor = 1
        for option in options:
            if tobeUnrolledDim / 8 % option == 0:
                factor = option
                break
        return factor

def pruneApproachQ(timed,noPrune=False):
    # prunePoint = ssr_prune_frac(timed,3)
    # pruned = timed[timed["SSR Config Count"] < prunePoint]
    #print(timed[["M"]])
    #timed["niceMRem"] = timed["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
    timed["UaJ"]=timed["n"].apply(unrollAndJamFactor)
    sorted_ssr =timed.sort_values("SSR Configs", ascending=True)
    best_third_ssr = sorted_ssr.iloc[range(0, len(sorted_ssr)//3)]

    hover_data = [
        "JSON Name",
        "timeout",
        "absoluteRank",
        "dma",
        "fmaddsPerCore",
        "n/k",
        "SSR Configs",
        "UaJ",
        "Regular Loads"
    ]

    if noPrune:
        special_figs=[]
        x_col = "dma"#"Avg n'_sz / k_size"
        y_col = "dma"#"Global Sim E2E_dma"
        special_figs.append(scatterWithColorSymbol(
            timed,
                x_col,
                y_col,
                "SSR Config Count",
                hover_data,
                "Reality Check. Make sure fastest point is ranked 1.",
                "timeout",
                ["circle","circle"]
        ))
        x_col = "SSR Configs"
        y_col = "dma"
        special_figs.append(
            scatterWithColor(
                timed,
                x_col,
                y_col,
                "SSR Configs",
                hover_data,
                "1) Full search space (divisor tiles only).",
                "symbolMarker",
            )
        )
        more_figs=[]
        return special_figs,more_figs

    
    ssr_thresh = max(best_third_ssr["SSR Configs"].values)
    recentlyPruned =  timed[~timed["FakeNN JSON Name"].isin(best_third_ssr["FakeNN JSON Name"])]
   
    pruned_sorted_l1 =best_third_ssr.sort_values("L1 Usage", ascending=False)
    best_half_l1 = pruned_sorted_l1.iloc[range(0, len(pruned_sorted_l1)//2)]
    l1_thresh = min(best_half_l1["L1 Usage"].values)
  
    
    # print("prune approach 2 statistics:")
    # print(f"full: {len(full)} pruned:{len(pruned)} % analyzed:{len(pruned)/len(full)}")
    # print(f"ann: {len(analyzed)} pruned: {len(pruned)} timed: {len(timed)} % timed:{len(timed)/len(pruned)}")
   
    special_figs=[]
    x_col = "dma"#"Avg n'_sz / k_size"
    y_col = "dma"#"Global Sim E2E_dma"
    special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "SSR Config Count",
            hover_data,
            "Reality Check. Make sure fastest point is ranked 1.",
            "timeout",
            ["circle","circle"]
     ))
    x_col = "SSR Configs"
    y_col = "dma"
    special_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "1) Full search space (divisor tiles only). Vertical line is SSR Config pruning threshold.",
            "symbolMarker",
        )
    )
    special_figs[-1].add_vline(x=ssr_thresh, line_width=2, line_dash="dash", line_color="green")


    x_col = "L1 Usage"
    y_col = "dma"
    special_figs.append(
        scatterWithColor(
            best_third_ssr,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "2) Pruned by SSR Configs. Vertical line is L1 Usage pruning threshold",
            "symbolMarker",
        )
    )
    special_figs[-1].add_vline(x=l1_thresh, line_width=2, line_dash="dash", line_color="green")

    x_col = "Regular Loads"
    y_col = "dma"
    special_figs.append(
        scatterWithColor(
            best_half_l1,
            x_col,
            y_col,
            "fmaddsPerCore",
            hover_data,
            "3) Pruned by L1 Usage. Minimize by Regular Loads",
            "symbolMarker",
        )
    )
    # more experiments
    more_figs=[]

    x_col = "Regular Loads"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            best_third_ssr,
            x_col,
            y_col,
            "fmaddsPerCore",
            hover_data,
            "3) What if we DON'T prune by L1 usage, just order by regular loads and tie break by FMADDs??",
            "symbolMarker",
        )
    )

    fig = genResultGraphQPDF(jugaadTitleQ(best_third_ssr),best_third_ssr, recentlyPruned,hover_data,color="FMADDs/core")
    more_figs.append(fig)
    x_col = "fmaddsPerCore"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            best_third_ssr,
            x_col,
            y_col,
            "Regular Loads",
            hover_data,
            "What if we DON'T prune by L1 usage??",
            "symbolMarker",
        )
    )

    x_col = "Regular Loads"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            best_third_ssr,
            x_col,
            y_col,
            "UaJ",
            hover_data,
            "3) What if we DON'T prune by L1 usage, just order by regular loads and tie break by UaJ factor?",
            "symbolMarker",
        )
    )

    x_col = "Regular Loads"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "fmaddsPerCore",
            hover_data,
            "Completely unpruned",
            "symbolMarker",
        )
    )

    x_col = "Regular Loads"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "UaJ",
            hover_data,
            "Completely unpruned, tie break with uaJ?",
            "symbolMarker",
        )
    )
    x_col = "Regular Loads"
    y_col = "UaJ"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "dma",
            hover_data,
            "reg loads = n_tiles * n / UaJ",
            "symbolMarker",
        )
    )

    x_col = "fmaddsPerCore"
    y_col = "SSR Configs"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "dma",
            hover_data,
            "Reality Check: SSR Configs and FmaddsPerCore (for Quidditch) are linear to each other (NOT TRUE!)",
            "symbolMarker",
        )
    )

    x_col = "L1 Usage"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "fmaddsPerCore",
            hover_data,
            "L1 Usage vs time",
            "symbolMarker",
        )
    )
    x_col = "SSR Configs"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "fmaddsPerCore",
            hover_data,
            "SSR Configs vs time",
            "symbolMarker",
        )
    )


    x_col = "n/k"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            timed,
            x_col,
            y_col,
            "SSR Configs",
            hover_data,
            "0.1) Full search space (multicolor points are analyzed by our model)",
            "symbolMarker",
        )
    )

    # PRUNE OUT points with n/k >= 1
    pruned_for_n_k=best_half_l1.copy()
    n_k_ge_one = pruned_for_n_k[pruned_for_n_k["n/k"]>=1.0]
    n_k_lt_one = pruned_for_n_k[pruned_for_n_k["n/k"]<1.0]

    x_col = "n/k"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            n_k_lt_one,
            x_col,
            y_col,
            "Regular Loads",
            hover_data,
            "Prune out n/k > 1, THEN order by n/k and tie-break with Regular Loads??",
            "symbol-marker",
        )
    )
    addScatterFlatColorMarker(
        more_figs[-1],
        n_k_ge_one,
        x_col,
        y_col,
        "gray",
        "circle-open",
        hover_data,
        "n/k > 1"
        )

    x_col = "fmaddsPerCore"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            n_k_lt_one,
            x_col,
            y_col,
            "Regular Loads",
            hover_data,
            "Prune out n/k > 1, THEN order by Fmadds and tie-break with Regular Loads??",
            "symbol-marker",
        )
    )
    addScatterFlatColorMarker(
        more_figs[-1],
        n_k_ge_one,
        x_col,
        y_col,
        "gray",
        "circle-open",
        hover_data,
        "n/k > 1"
        )

    x_col = "Regular Loads"
    y_col = "dma"
    more_figs.append(
        scatterWithColor(
            n_k_lt_one,
            x_col,
            y_col,
            "fmaddsPerCore",
            hover_data,
            "Prune out n/k > 1, THEN order by reg loads configs and tie-break with Fmadds Per Core??",
            "symbol-marker",
        )
    )
    addScatterFlatColorMarker(
        more_figs[-1],
        n_k_ge_one,
        x_col,
        y_col,
        "gray",
        "circle-open",
        hover_data,
        "n/k > 1"
        )
  
    #  resultGraph = genResultGraphPDF(jugaadTitle(timed),nice_timed_lt1,nice_ut_lt1,nice_timed_gte1,hover_data,"n / k")
    #printFinalRankingQ(jugaadTitleQ(best_third_ssr),best_third_ssr,"fmaddsPerCore")
    #  special_figs.append(resultGraph)     
    return special_figs,more_figs

def visualizePruningQ(timed,titleOfWebpage):
    timed["Total CC Tiles"] = timed["SSR Config Count"]
    timed["Time (cycles)"]=timed["dma"]       
    special_figs = []
    more_figs = []

    #jugaad
    if titleOfWebpage == "1x400x161-myrtle-pruning":
        special_figs,more_figs = pruneApproachQ(timed,noPrune=True)
    else:
        special_figs,more_figs = pruneApproachQ(timed,noPrune=False)
    # special_figs=special_figs+more_figs

     #more_figs = pruneApproach2(timed,analyzed,full)
     
    return saveFigsInHTML(special_figs, more_figs, titleOfWebpage)

  