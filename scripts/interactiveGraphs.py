import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def generateInteractiveGraphsTuples(divisors,remainders, title, titleOfWebpage):
        df = divisors[0]
        df["divisorRank"]=df["absoluteRank"]
        df["symbolMarker"] = 'O'
        df["remainderTiles"] = df["remainderTiles"].apply(lambda x: f"{x}")
        rm_ut = remainders[1]
        rm_ut["symbolMarker"] = '^'
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
                #C = A[~A['ID'].isin(B['ID'])]
                rm_ut = rm_ut[~rm_ut['FakeNN JSON Name'].isin(rm['FakeNN JSON Name'])]
                # combine timed points into single DF, then create absolute rank
                timed = pd.concat([df, rm], join='inner', ignore_index=True)   
                timed_sorted = timed.sort_values(by="dma", ascending=True)
                timed_sorted["absoluteRank"] = range(1, int(timed_sorted.shape[0] + 1))
                timed = timed_sorted.sort_values(by="symbolMarker", ascending=True) 
                timed ["flatColor"] = "pink" 
                # special figures
                special_figs = []
                x_col = "SSR Configs"
                y_col = "dma"   
                special_figs.append(prunedScatter(timed,x_col,y_col,"regPerStream",hover_data,"OLD RATIO: timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(special_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"timed divisors and (some) timed remainders")
                
                x_col = "regPerStream"
                y_col = "dma"
                timed_pruned = timed[timed["SSR Config Count"]<=1024]   
                rm_ut_pruned = rm_ut[rm_ut["SSR Config Count"]<=1024]
                special_figs.append(scatterWithColor(timed_pruned,x_col,y_col,"dma",hover_data,"OLD RATIO + pruned to SSR Configs <= 1024","symbolMarker"))
                addScatterFlatColorMarker(special_figs[-1],rm_ut_pruned,x_col,y_col,"gray","triangle-up",hover_data,"timed divisors and (some) timed remainders")
                
                # y_col = "HW Loops / SSR Loads" 
                # x_col = "regPerStream"
                # color = "dma"
                # special_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"OLD RATIO vs. UPDATED SUMMATION OF RATIOS (untimed points omitted)","symbolMarker"))
                # #addScatterFlatColorMarker(special_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")

                x_col = "SSR Configs"
                y_col = "dma" 
                color = "HW Loops / SSR Loads"
                special_figs.append(prunedScatter(timed,x_col,y_col,color,hover_data,"UPDATED, SCALED SUMMATION OF RATIO: timed divisors and (some) timed remainders ","symbolMarker"))
                addScatterFlatColorMarker(special_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                
                x_col = "HW Loops / SSR Loads"
                y_col = "dma" 
                color = "dma"
                special_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"UPDATED, SCALED SUMMATION OF RATIO: timed divisors and (some) timed remainders ","symbolMarker"))
                addScatterFlatColorMarker(special_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                
                # more figures
                more_figs = []
                x_col = "dma" 
                y_col = "A SSR Reuse Loads"
                color = "A SSR Reuse Loads"
               # color = "HW Loops / SSR Loads"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
               # timed["remainderTiles"] = "" + timed["remainderTiles"]
                
                 
                x_col = "FMADDsPerCore"
                y_col = "dma"
                color = "L3 Loads"
               # color = "HW Loops / SSR Loads"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
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
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                

                x_col = "L1 Usage" 
                y_col = "L3 Loads"
                color = "dma"
               # color = "HW Loops / SSR Loads"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                
                y_col = "dma" 
                x_col = "L3 Loads"
                color = "A SSR Reuse Loads"
               # color = "HW Loops / SSR Loads"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")


                x_col = "dma" 
                y_col = "regPerStream"
                color = "SSR Configs"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                

                x_col = "dma" 
                y_col = "HW Loops / SSR Loads"
                color = "SSR Configs"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                
                x_col = "dma" 
                y_col = "A SSR Reuse Loads"
                color = "HW Loops / SSR Loads"
                more_figs.append(scatterWithColor(timed,x_col,y_col,color,hover_data,"timed divisors and (some) timed remainders","symbolMarker"))
                addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
                
                return saveFigsInHTML(special_figs,more_figs,titleOfWebpage)
       

        # specialized graphs
        special_figs = []
        x_col = "SSR Configs"
        y_col = "dma"        
        special_figs.append(prunedScatter(df,x_col,y_col,"regPerStream",hover_data,"timed divisors and untimed remainders"))
        addScatterFlatColorMarker(special_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"Remainders Untimed")
        
        dfPruned = df[df["SSR Configs"]<= 1024]
        dfPrunedPoints = dfPruned["SSR Configs"].values.tolist()
      #  print(f"there are {len(dfPrunedPoints)} timed, pruned points to graph are: {dfPrunedPoints}")
        rm_utPruned=rm_ut[rm_ut["SSR Configs"]<= 1024]
        rm_utPruned.to_csv("./out/remaindertilesWithFewerThan1024.csv",index=False)
        fewerPoints = rm_utPruned["SSR Configs"].values.tolist()
        unique_ssr_configs = list(set(rm_ut["SSR Configs"].values.tolist()))
        unique_ssr_configs.sort()
       # print(f"There are {len(fewerPoints)} points with fewer than 1024 SSR configs: {fewerPoints}")
        
        x_col = "SSR Configs"
        y_col = "regPerStream"
    
     #    special_figs.append(scatterWithColor(dfPruned,x_col,y_col,"dma",hover_data,"timed divisors and untimed remainders"))
     #    addScatterFlatColorMarker(special_figs[-1],rm_utPruned,x_col,y_col,"lightpink","triangle-up",hover_data,"remainders untimed")
        special_figs.append(scatterWithColor(rm_utPruned,x_col,y_col,"regPerStream",hover_data,"untimed remainders and timed divisors with SSR Configs <= 1024"))
        special_figs[-1].update_traces(marker_symbol='triangle-up')
        addScatterFlatColorMarker(special_figs[-1],dfPruned,x_col,y_col,"black","circle",hover_data,"divisors timed")

        x_col = "SSR Configs"
        y_col = "dma"  
        special_figs.append(scatterWithColor(rm_utPruned,x_col,y_col,"regPerStream",hover_data,"untimed remainders and timed divisors with SSR Configs <= 1024"))
        special_figs[-1].update_traces(marker_symbol='triangle-up')
        addScatterFlatColorMarker(special_figs[-1],dfPruned,x_col,y_col,"black","circle",hover_data,"divisors timed")

        # more experiments
        more_figs = []
        x_col = "SSR Configs"
        y_col = "regPerStream"
        more_figs.append(scatterWithColor(rm_ut,x_col,y_col,"HW Loops / SSR Loads",hover_data,"untimed remainders"))

     
        x_col = "regPerStream"
        y_col = "HW Loops / SSR Loads"
     #    more_figs.append(scatterWithColor(rm_ut,x_col,y_col,"dma",hover_data,"untimed remainders"))
     #    more_figs[-1].update_traces(marker_symbol='triangle-up')
        more_figs.append(scatterWithColor(df,x_col,y_col,"dma",hover_data,"divisors timed and remainders untimed"))
     #   more_figs[-1].add_traces(list(fig.data))
        addScatterFlatColorMarker(more_figs[-1],rm_ut,x_col,y_col,"gray","triangle-up",hover_data,"remainder untimed")
        
       
        x_col = "HW Loops / SSR Loads"
        y_col = "oldRegPerStream"
        more_figs.append(scatterWithColor(df,x_col,y_col,"dma",hover_data,"divisors timed"))

        x_col = "SSR Configs"
        y_col = "dma"
        more_figs.append(scatterWithColor(df,x_col,y_col,"HW Loops / SSR Loads",hover_data,"divisors timed"))

        x_col = "m"
        y_col = "SSR Configs"
        more_figs.append(scatterWithColor(rm_ut,x_col,y_col,"m",hover_data,"untimed remainders"))

        x_col = "n"
        y_col = "SSR Configs"
        more_figs.append(scatterWithColor(rm_ut,x_col,y_col,"n",hover_data,"untimed remainders"))

        x_col = "k"
        y_col = "SSR Configs"
        more_figs.append(scatterWithColor(rm_ut,x_col,y_col,"k",hover_data,"untimed remainders"))

        x_col = "SSR Configs"
        y_col = "dma"
        more_figs.append(scatterWithColor(df,x_col,y_col,"CC L1 / L1",hover_data,"divisors timed"))
        
        x_col = "SSR Configs"
        y_col = "dma"
        more_figs.append(scatterWithColor(df,x_col,y_col,"regPerStream",hover_data,f"{title} (using old metric regPerStream)"))

        # export to HTML
        return saveFigsInHTML(special_figs,more_figs,titleOfWebpage)

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
        small_hover_data = [
            "JSON Name",
            x_col,
            y_col,
            "HW Loops / SSR Loads",
            "L3 Loads"
        ]

        # specialized graphs
        special_figs = []
        x_col = "SSR Configs"
        y_col = "Kernel Time"        
        special_figs.append(prunedScatter(df,x_col,y_col,"HW Loops / SSR Loads",hover_data,""))

        # more experiments
        more_figs = []
        x_col = "SSR Configs"
        y_col = "Kernel Time"
        more_figs.append(scatterWithColor(df,x_col,y_col,"CC L1 / L1",hover_data,title))
        
        x_col = "SSR Configs"
        y_col = "Kernel Time"
        more_figs.append(scatterWithColor(df,x_col,y_col,"regPerStream",hover_data,f"{title} (using old metric regPerStream)"))

        # export to HTML
        return saveFigsInHTML(special_figs,more_figs,titleOfWebpage)

def saveFigsInHTML(special_figs,more_figs,titleOfWebpage):
     # --- Convert each figure to HTML div ---
     special_divs = []
     for f in special_figs:
               special_divs.append(pio.to_html(f, include_plotlyjs="cdn", full_html=False))
     moreDivs=[]
     for f in more_figs:
               moreDivs.append(pio.to_html(f, include_plotlyjs="cdn", full_html=False))
     # --- Concatenate divs into single string ---
     specialDivsAsHTML = ""
     for d in special_divs:
               specialDivsAsHTML = specialDivsAsHTML + f"""<div class="plot-box">{d}</div>"""
     moreDivsAsHTML=""
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
       

def scatterWithColor(df,x_col,y_col,color,hover_data,title,marker=""):
     fig8 = px.scatter(
          df,
          x=x_col,
          y=y_col,
          color=color,
          symbol="symbolMarker",
          symbol_sequence=['circle','triangle-up','triangle-up'],
          hover_data=hover_data,  # Show these columns on hover
          title=f"{title} <b>{x_col} vs {y_col}</b>",
     )  
     return fig8

# def addScatterWithColor(fig, df,x_col,y_col,color,hover_data):
#      fig.add_scatter(
#           df,
#           x=x_col,
#           y=y_col,
#           color=color,
#           hover_data=hover_data,  # Show these columns on hover
#      )

def hoverTemplateString(df,hover_data,title):
        idx = {col: i for i, col in enumerate(df[hover_data].columns.values)}
        str = f"{title} "
        for k in hover_data:
                str = str + "<br>" +f"{k}"+ ": %{" + "customdata"+ f"[{idx[k]}]" +"}"
        str = str + "<extra></extra>" 
        return str

def addScatterFlatColorMarker(fig,df, x_col, y_col, color, marker, hover_data, title):
        customData=df[hover_data].to_numpy()
        fig.add_scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(color=color,symbol=marker),
            showlegend=False,
            name = title,
            customdata=customData,
            hovertemplate=(hoverTemplateString(df,hover_data,title) 
            ),
        )
        #mark
     
def prunedScatter(df,x_col,y_col,color,hover_data,title,marker=""):
  
     df_mod = df
     if marker == "":
             df["symbolMarker"] = "O"
     # df_mod["color"] = df_mod["k/n"]
     # print(df_mod["L3 Loads"].values)
     # top =  max(df_mod["L3 Loads"].values)
     # bot =  min(df_mod["L3 Loads"].values)
     # mid = (top - bot) / 2.0
     #prunePointL3 = bot + mid#2686976
     unique_ssr_configs = list(set(df_mod["SSR Configs"].values.tolist())) # remove duplicates
     unique_ssr_configs.sort() # sort least to greateset
     prunePoint = unique_ssr_configs[1] # prune to two smallest groups of ssr_configs
     prunePoint = 1024
     fig14 = px.scatter(
          df_mod,
          x=x_col,
          y=y_col,
          symbol="symbolMarker",
          symbol_sequence=['circle','triangle-up','triangle-up'],
          color=color,#"regPerStream",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
          hover_data=hover_data,  # Show these columns on hover
          title=f"{title} <b>{x_col} vs {y_col}</b> with SSR Configs >= {prunePoint} separated with vertical dotted line.",
     )
     fig14.add_vline(x=prunePoint, line_width=2, line_dash="dash", line_color="green")
     # turn the pruned points gray?
     return fig14
        

def generateInteractiveGraphsKernelVsDMA(rem_timed,rem_retimed, title, titleOfWebpage):
        # convert remainderTiles to string
        # rank points by dma time
        rem_timed["remainderTiles"] = rem_timed["remainderTiles"].apply(lambda x: f"{x}")
        rem_timed= rem_timed.sort_values(by="dma", ascending=True)
        rem_timed["absoluteRank"] = range(1, int(rem_timed.shape[0] + 1))

        rem_retimed["remainderTiles"] = rem_retimed["remainderTiles"].apply(lambda x: f"{x}")
        rem_retimed= rem_retimed.sort_values(by="dma", ascending=True)
        rem_retimed["absoluteRank"] = range(1, int(rem_retimed.shape[0] + 1))
        
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
        df_merged = pd.merge(rem_timed, rem_retimed, on='FakeNN JSON Name', how="inner", suffixes=('_withBug', '_noBug'))
        df_merged['Kernel Time Difference'] = df_merged['Kernel Time_withBug'] - df_merged['Kernel Time_noBug']
        print(df_merged[['FakeNN JSON Name','Kernel Time Difference',"Kernel Time_withBug","Kernel Time_noBug"]] )
        df_merged['% Kernel Time Change'] = (df_merged['Kernel Time Difference'] / df_merged['Kernel Time_withBug']) * 100
        #df_merged['RankDiff'] = df_merged['absoluteRank_withBug'] - df_merged['absoluteRank_noBug']
        # df_merged['Status'] = df_merged['Difference'].apply(
        # lambda x: 'Slow Down' if x < 0 else 'Same or Better'
        # )

        fig2 = px.bar(df_merged, 
        x='FakeNN JSON Name', 
        y='Kernel Time Difference',
        title='Fixing Kernel Time Parsing Bug',
        color='Kernel Time Difference',
        color_continuous_scale='RdBu', # Red for negative, Blue for positive
        hover_name='FakeNN JSON Name',
        hover_data={
                'Kernel Time Difference': ':.2f',    # Format to 2 decimal places
                'Kernel Time_withBug': True,         # Show the raw value from File A
                'Kernel Time_noBug': True,         # Show the raw value from File B
                'FakeNN JSON Name': False       # Hide Category if it's already on the X-axis
        },
        labels={'Kernel Time Difference': 'withBug - No Bug (cycles)','value': 'Kernel Time (cycles)', 'variable': 'Source File'})
        #labels={'Difference': 'Redundant - No Redundant (cycles)','value': 'Kernel Time (cycles)', 'variable': 'Source File'})
        special_figs.append(fig2)
        return saveFigsInHTML(special_figs,more_figs,titleOfWebpage)
