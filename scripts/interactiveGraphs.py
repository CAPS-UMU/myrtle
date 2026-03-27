import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

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
        myTitle=f"{title} {x_col} vs {y_col}, special title"
        more_figs.append(scatterWithColor(df,x_col,y_col,"regPerStream",hover_data,myTitle))

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
    <a href="results.html" >Back to Landing Page</a>
    <div class="dashboard">
      {specialDivsAsHTML} 
        {"<div>More Experiments</div>"}
        {moreDivsAsHTML}        
        </div>
        </body>
        </html>
        """
     return html
       

def scatterWithColor(df,x_col,y_col,color,hover_data,title):
     fig8 = px.scatter(
          df,
          x=x_col,
          y=y_col,
          color=color,
          hover_data=hover_data,  # Show these columns on hover
          title=f"{title} {x_col} vs {y_col}",
     )
     return fig8
     
def prunedScatter(df,x_col,y_col,color,hover_data,title):
     x_col = "SSR Configs"
     y_col = "Kernel Time"
     df_mod = df
     # df_mod["color"] = df_mod["k/n"]
     # print(df_mod["L3 Loads"].values)
     # top =  max(df_mod["L3 Loads"].values)
     # bot =  min(df_mod["L3 Loads"].values)
     # mid = (top - bot) / 2.0
     #prunePointL3 = bot + mid#2686976
     unique_ssr_configs = list(set(df_mod["SSR Configs"].values.tolist())) # remove duplicates
     unique_ssr_configs.sort() # sort least to greateset
     prunePoint = unique_ssr_configs[1] # prune to two smallest groups of ssr_configs
     fig14 = px.scatter(
          df_mod,
          x=x_col,
          y=y_col,
          color="SSRconfigsXregPerStream",#"regPerStream",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
          hover_data=hover_data,  # Show these columns on hover
          title=f"{title} {x_col} vs {y_col} with SSR Configs > {prunePoint} pruned away.",
     )
     fig14.add_vline(x=prunePoint, line_width=2, line_dash="dash", line_color="green")
     # turn the pruned points gray?
     return fig14
        