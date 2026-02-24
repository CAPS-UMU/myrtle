import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def generateInteractiveGraphs(
    df,
    title,
    titleOfWebpage,
    shapeMetric,
    cmFeatures,
    coeffs,
    features,
    df_w_prediction,
):
    x_col = "Regular Loads"
    y_col = "Kernel Time"
    df["Tile Dims"] = df["JSON Name"]
    hover_data = [
        "Tile Dims",
        "absoluteRank",
        x_col,
        y_col,
        "CC L1 Footprint",
        "fmaddsPerCore",
        "Regular Loads",
        "Total SSR Loads",
        "SSR Config Count",
        "L3 Loads",
        "B SSR Loads",
        "A SSR Reuse Loads",
        "A Not Reused SSR Loads",
        "Hardware Loops",
        "L3 Loads Timed",
        "tileB",
        "tileC",
        "tileC_cc"
    ]

    # df_sorted = df.sort_values("L1 Usage", ascending=False)
    df_sorted = df.sort_values("Kernel Time", ascending=True)
    df_best_5 = df_sorted.iloc[range(0, 4)]

    x_col = "L1 Usage"
    y_col = "Kernel Time"
    fig1 = px.scatter(
        df_best_5,
        x=x_col,
        y=y_col,
        color="Regular Loads",  # "Regular Loads",  # Optional: color points by a category column
        hover_data=hover_data,  # Show these columns on hover
        title="4 Fastest Files for 256x256x256 Matmul",
    )

    # --- Convert each figure to HTML div ---

    div1 = pio.to_html(fig1, include_plotlyjs="cdn", full_html=False)

    # --- Combine into HTML page with grid layout ---
    html = f"""
    <html>
    <head>
    <title>Plotly Express 5x1 Dashboard</title>
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
    <div class="dashboard">
    <div><h2 style="text-align:center;"> Tile 16-16-64 (dark blue point) is faster than Tile 64-16-16 (yellow point). How come?
        </h2><h2 style="text-align:center;"> Tap/Hover over the points to see factors we've considered.
        </h2></div>
        <div class="plot-box">{div1}</div>
        <h2 style="text-align:center;"> <a href="https://docs.google.com/forms/d/e/1FAIpQLSdF2JKaTMMKEZvscaYM_-02P6dRkKRDM6uwOpoSoTWbx0BYRw/viewform?usp=publish-editor"> Tell Us What You Think Here! </a></h2>
         <h3 style="text-align:center;"><a href="https://github.com/CAPS-UMU/myrtle"> Our Github Repo </a></h3>
       
        </div> 
        </body>
        </html>
        """
    return html
