import pandas as pd
import plotly.express as px
import sys
import os
import matplotlib.pyplot as plt
import plotly.io as pio
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.svm import SVC, SVR
import pickle
import recentGraphs_2_23 as rg_2_23
import addExtraMetrics as ae
import interactiveGraphs as ig
import subprocess


def main():
    # first argument is output folder name
    # each subsequent argument is a webpage to add to the index
    outputFolder = sys.argv[1]
    pages = sys.argv[2:]
    # destroy and then create the output folder
    subprocess.call(["rm", "-rf", outputFolder])
    subprocess.call(["mkdir", outputFolder])
    # copy each webpage to the output folder
    pageBases = []
    for page in pages:
        pageBase = os.path.basename(page)
        pageBases.append(pageBase)
        subprocess.call(["cp", page, f"{outputFolder}/{pageBase}"])
    # create the ToC
    toc = ""
    for page in pageBases:
        toc = toc + f'<li><a href="{page}">{page}</a></li>'
    toc = f"<ul>{toc}</ul>"
    # wrap the ToC in a customized homepage
    html = defaultHomePage(
        "128 Cube Remainder Tile Results",
        "In-progress results timing tiled matmul on snitch using remainder tiles.",
        toc,
    )
    # write everything to an index.html file in the output folder
    # --- Write to file ---
    with open(f"{outputFolder}/index.html", "w") as f:
        f.write(html)
    print(f"Generate HTML Index: placed index and subpages in {outputFolder}.")


def defaultHomePage(title, customIntro, ToC):
    style = """<style>
        body {
            font-family: Arial, sans-serif;
            margin: 30px;
            background-color: #f7f7f7;
        }

        .dashboard {
            display: flex;
            flex-direction: column;
            gap: 30px;
            /* space between charts */
        }

        .plot-box {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .attention{
            color: red;
            text-decoration: underline;
        }
    </style>"""
    theHTML = f"""<html>

<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {style}
</head>

<body>
    <h1 style="text-align:center;">{title}</h1>
    <div class="dashboard">
        <h2>Summary</h2>
        <a href="index.html">Back to Landing</a>
        <p>{customIntro}</p>
        <h2>Index of Pages</h2>
        {ToC}
       
    </div>
</body>

</html>

    """
    return theHTML


if __name__ == "__main__":
    main()
