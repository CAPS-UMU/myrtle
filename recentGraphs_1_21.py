import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def addLineOfBestFit(df, x_col, y_col, fig):
    # adding line of best fit for SSR configs vvvvvvv
    x = np.array(df[x_col])
    y = np.array(df[y_col])
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    # --- Compute coefficients ---
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(x.reshape(-1, 1), y)

    # --- Add regression line ---
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            name=f"Fit line (y = {slope:.2f}x + {intercept:.2f})",
            line=dict(color="red", width=2),
        )
    )

    # --- Add text annotation for equation and R² ---
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}<br>R² = {r2:.3f}"
    fig.add_annotation(
        x=max(x),
        y=min(y),
        text=equation_text,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
    )

    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig

def fixLegends(fig, df, color, shape):
    fig.update_layout(
        # 1. Adjust right margin to make room for both elements
        margin=dict(r=200),  # Increase right margin significantly (e.g., 200 pixels)
        # 2. Position the Color Bar (closer to the plot)
        coloraxis_colorbar=dict(
            orientation="v",  # Keep vertical
            yanchor="middle",
            y=0.5,  # Center vertically
            x=1.05,  # Position slightly outside the plot area (1.0)
            len=0.7,  # Optional: control the length
        ),
        # 3. Position the Discrete Legend (further right)
        legend=dict(
            orientation="v",  # Keep vertical
            yanchor="top",
            y=1,  # Place at the top of the figure area
            x=1.2,  # Position to the right of the color bar (adjust as needed)
            xanchor="left",
        ),
    )
    return fig


def generateInteractiveGraphs(df, title, titleOfWebpage, shapeMetric,cmFeatures, coeffs, features, df_w_prediction):
        x_col = "Regular Loads"
        y_col = "Kernel Time"
        hover_data_small = [
            "JSON Name",
            "absoluteRank",
            x_col,
            y_col,
            "costRank",
            "L3 Loads Timed",
            "L1 Usage",
            "CC L1 Footprint",
            "fmaddsPerCore",
            "SSR Configs",
        ]

        hover_data = [
            "JSON Name",
            x_col,
            y_col,
            "SSR Config Count",
            "absoluteRank",
            "costRank",
            "L3 Loads",
            "B SSR Loads",
            "Total SSR Loads",
            "A SSR Reuse Loads",
            "A Not Reused SSR Loads",
            "Hardware Loops",
            "L3 Loads Timed",
            "L1 Usage",
            "CC L1 Footprint",
            "tileA",
            "tileB",
            "tileC",
            "tileA_cc",
            "tileC_cc",
            "k",
            "cost",
            "regPerStream",
            "sumSSRsRegs",
            "k/n",
            "fmaddsPerCore",
        ]
      
        # let's merge kernel time and predicted kernel time into a single y column to graph more easily
        x_col = "Regular Loads"
        y_col = "Y_Value"
        symbol = "Y_Value_Type"
        y1 = "Kernel Time"
        y2 = "Predicted Kernel Time"

        df_long = pd.melt(
            df_w_prediction,
            # These columns will remain as identifying columns
            id_vars=[
                x_col,
                "JSON Name",
                "SSR Config Count",
                "absoluteRank",
                "predictedRank",
                "L3 Loads",
            ],
            # These are the columns you want to "melt" into a new Y-Value column
            value_vars=[y1, y2],
            # New column for the label (i.e., 'Col_A' or 'Col_B')
            var_name=symbol,
            # New column for the value (i.e., the actual number)
            value_name=y_col,
        )
        t1 = f"<br>SVR trained with features {str(features)}"
        t2 = f"<br>SVR's weights are {str(coeffs)}</p>"
        fig10 = px.scatter(
            df_long,
            x="absoluteRank",  # "SSR Config Count",#"L3 Loads",#"absoluteRank",                # The common x-axis data
            y=y_col,  # The combined y-axis values
            color=symbol,  # Colors the dots based on the original color column
            symbol=symbol,  # Gives a different shape/symbol to Col_A vs Col_B points
            hover_data=[
                "JSON Name",
                x_col,
                "SSR Config Count",
                "absoluteRank",
                "predictedRank",
                "L3 Loads",
            ],  # Show these columns on hover
            title=f"Comparison of {y1} and {y2}{t1}{t2}",
        )
        
        x_col = "absoluteRank"
        y_col = "predictedRank"
        fig20 = px.scatter(
            df_w_prediction,
            x=x_col,
            y=y_col,
            color="Kernel Time",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}"+"<br>we want this graph to look as close to y=x as possible.",
        )

        figs = []
        for feat in features:
             x_col = feat
             y_col = "Kernel Time"
             fig= px.scatter(
                  df,
                  x=x_col,
                  y=y_col,
                  color="regPerStream",  # Optional: color points by a category column
                  hover_data=hover_data,  # Show these columns on hover
                  title=f"{title} {x_col} vs {y_col}",)
             html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
             div=f"""<div class="plot-box">{html}</div>"""
             figs.append(div)
        #print('\n'.join(figs))
        feature_graphs='\n'.join(figs)

        # more experiments
        x_col = "SSR Config Count"#"sumSSRsRegs"
        y_col = "Kernel Time"
        fig8 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="regPerStream",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )
        x_col = "SSR Config Count"#"sumSSRsRegs"
        y_col = "Kernel Time"
        fig9 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="cost",#"k/n",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}, where c = regPerStream = REG_LOADS / SSR_LOADS per core = n*m/128*k",
        )
        #title=f"{title} {x_col} vs {y_col}, where c = SSR_CONFIGS - (k/n)*(NUM_HW_LOOPS) - (k/n) - L1_USAGE",
        #title=f"{title} {x_col} vs {y_col}, where c = SSR_CONFIGS / c1 + k/n * c2/c1"
        #title=f"{title} {x_col} vs {y_col}, where c = SSR_CONFIGS - (k/n)*(NUM_HW_LOOPS) - (k/n) - L1_USAGE",
        #df["cost"] = df["SSR Config Count"] - df["k/n"]*df["Hardware Loops"] - df["k/n"] - df["L1 Usage"]

        df["MNK/m"]=df["M"] * df["N"] * df["K"] / df ["m"]
        x_col = "MNK/m"
        y_col = "Kernel Time"
        fig6 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="SSR Config Count",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        df["MNK/n"]=df["M"] * df["N"] * df["K"] / df ["n"]
        x_col = "L1 Usage"
        y_col = "Kernel Time"
        fig7 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="k/n",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "L1 Usage"
        y_col = "Kernel Time"
        fig1 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Regular Loads",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "SSR Config Count"
        y_col = "Kernel Time"
        fig3 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="fmaddsPerCore",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "Regular Loads"
        y_col = "Kernel Time"
        fig2 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="L1 Usage",#"L3 L/S Timed",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "k"
        y_col = "Kernel Time"
        fig4 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Regular Loads",#"L3 L/S Timed",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "fmaddsPerCore"
        y_col = "Kernel Time"
        fig5 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Regular Loads",#"L3 L/S Timed",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        # --- Convert each figure to HTML div ---

        div10 = pio.to_html(fig10, include_plotlyjs="cdn", full_html=False)
        div20 = pio.to_html(fig20, include_plotlyjs="cdn", full_html=False)
        div1 = pio.to_html(fig1, include_plotlyjs="cdn", full_html=False)
        div2 = pio.to_html(fig2, include_plotlyjs="cdn", full_html=False)
        div3 = pio.to_html(fig3, include_plotlyjs="cdn", full_html=False)
        div4 = pio.to_html(fig4, include_plotlyjs="cdn", full_html=False)
        div5 = pio.to_html(fig5, include_plotlyjs="cdn", full_html=False)
        div6 = pio.to_html(fig6, include_plotlyjs="cdn", full_html=False)
        div7 = pio.to_html(fig7, include_plotlyjs="cdn", full_html=False)
        div8 = pio.to_html(fig8, include_plotlyjs="cdn", full_html=False)
        div9 = pio.to_html(fig9, include_plotlyjs="cdn", full_html=False)
        
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
        <div class="plot-box">{div10}</div>
        <div class="plot-box">{div20}</div>
        {"<div>Each of the features vs Kernel Time</div>"}
        {feature_graphs}
        {"<div>More Experiments</div>"}
        <div class="plot-box">{div1}</div>
        <div class="plot-box">{div2}</div>
        <div class="plot-box">{div3}</div>
        <div class="plot-box">{div4}</div>
        <div class="plot-box">{div5}</div>
        <div class="plot-box">{div6}</div>
        <div class="plot-box">{div7}</div>
        <div class="plot-box">{div8}</div>
        <div class="plot-box">{div9}</div>
        </div>
        </body>
        </html>
        """
        return html

def generateInteractiveC1C2Graph(df, titleOfWebpage):
        #print("WARNING: we remove points with 8-64-16 because it makes the scale of the graph way too large.")
        #df = df[df["fst"]!="8-64-16"]
        df["Rank Sum"] = df["P1 Rank"] + df["P2 Rank"]
        df["SSR Config Diff"] = abs(df["P1 SSR Configs"] - df["P2 SSR Configs"])
    #     df["SSR Config Diff"] = df.apply(
    #     lambda y: if df["P1 SSR Configs"] - df["P1 SSR Configs"] < 0, y[] - (y["m"] * y["n"] + y["m"] * y["k"]) / 8.0 + y["k"] * y["n"], axis=1
    # )

        x_col = "c1"
        y_col = "c2"
        hover_data = [
            "pair",
            "Time Diff",
            x_col,
            y_col,
            "P1 SSR Configs","P2 SSR Configs","Time P1","Time P2","P1 Rank","P2 Rank"
        ]

        print("HOODLE")
        print(type(df["Time P1"].iloc(0)))

        fig1 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Time P1",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 Vs C2 values colored by speed of the point P1 in pair (P1, P2).",
        )

        x_col = "c1"
        y_col = "c2"
        df_filtered = df[df["Rank Sum"] < 20]
        fig1_1 = px.scatter(
            df_filtered,
            x=x_col,
            y=y_col,
            color="Rank Sum",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 Vs C2 values colored by the sum of ranks of the two points in pair (P1, P2), only showing pairs with rank sum < 20.",
        )

        x_col = "c1"
        y_col = "c2"
        fig1_2 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Time Diff",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 Vs C2 values colored by time difference between points in pair (P1, P2)",
        )

        y_col = "Time Diff"
        x_col = "c2"
        fig7 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Rank Sum",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C2 values ordered by time difference between points in pair (P1, P2)",
        )

        y_col = "Time Diff"
        x_col = "c1"
        fig8 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Rank Sum",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 values ordered by time difference between points in pair (P1, P2)",
        )

        x_col = "Rank Sum"
        y_col = "c2"
        df_filtered = df[df["Rank Sum"] < 10]
        fig6 = px.scatter(
            df_filtered,
            x=x_col,
            y=y_col,
            color="pair",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 values for pairs of points with sum of their ranks < 10",
        )

        x_col = "c1"
        y_col = "c2"
        df_filtered = df[df["P1 SSR Configs"] == df["P2 SSR Configs"]]
        fig9 = px.scatter(
            df_filtered,
            x=x_col,
            y=y_col,
            color="P1 SSR Configs",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 vs C2 values for pairs of points with same number of SSR Configs",
        )

        x_col = "P1 SSR Configs"
        y_col = "c1"
        df_filtered = df[df["P1 SSR Configs"] == df["P2 SSR Configs"]]
        fig10 = px.scatter(
            df_filtered,
            x=x_col,
            y=y_col,
            color="P1 SSR Configs",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 values for pairs of points with same number of SSR Configs",
        )

        x_col = "P1 SSR Configs"
        y_col = "c2"
        df_filtered = df[df["P1 SSR Configs"] == df["P2 SSR Configs"]]
        fig11 = px.scatter(
            df_filtered,
            x=x_col,
            y=y_col,
            color="Rank Sum",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C2 values for pairs of points with same number of SSR Configs",
        )

        x_col = "P1 Rank"
        y_col = "P2 Rank"
        fig4 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="c1",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="Speed of P1 vs Speed of P2 colored by C1 value for pair (P1,P2)",
        )
        x_col = "P1 Rank"
        y_col = "P2 Rank"
        fig5 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="c2",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="Speed of P1 vs Speed of P2 colored by C2 value for pair (P1,P2)",
        )

        x_col = "c1"
        y_col = "c2"
        fig1_3 = px.scatter(
            df[df["fst"]=="16-16-64"],
            x=x_col,
            y=y_col,
            color="P2 Rank",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 vs C2 values when P1 = 16-16-64 (fastest)",
        )

        x_col = "c1"
        y_col = "c2"
        fig2 = px.scatter(
            df[df["fst"]=="32-32-32"],
            x=x_col,
            y=y_col,
            color="P2 Rank",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1 vs C2 values when P1 = 32-32-32 (biggest)",
        )

        # x_col = "P1 SSR Configs"
        # y_col = "c2"
        # fig12 = px.scatter(
        #     df,
        #     x=x_col,
        #     y=y_col,
        #     color="P2 SSR Configs",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
        #     hover_data=hover_data,  # Show these columns on hover
        #     title="C2 values for pairs of points as SSR Configs changes",
        # )

        # x_col = "SSR Config Diff"
        # y_col = "c1"
        # fig13 = px.scatter(
        #     df,
        #     x=x_col,
        #     y=y_col,
        #     color="SSR Config Diff",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
        #     hover_data=hover_data,  # Show these columns on hover
        #     title="As difference between SSR Configs increases, how do pairs' c1 values change?",
        # )

        df_pruned = df[df["P1 SSR Configs"] <= 8192]
        df_pruned = df_pruned[df["P2 SSR Configs"] <= 8192]

        x_col = "c1"
        y_col = "c2"
        fig12 = px.scatter(
            df_pruned,
            x=x_col,
            y=y_col,
            color="P1 SSR Configs",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="C1,C2 values for pairs of points with SSR Configs <= 8192",
        )

        x_col = "SSR Config Diff"
        y_col = "c1"
        fig13 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="SSR Config Diff",#"Hardware Loops",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title="As difference between SSR Configs increases, how do pairs' c1 values change?",
        )

        # --- Convert each figure to HTML div ---

        div1 = pio.to_html(fig1, include_plotlyjs="cdn", full_html=False)
        div1_1 = pio.to_html(fig1_1, include_plotlyjs="cdn", full_html=False)
        div1_3 = pio.to_html(fig1_3, include_plotlyjs="cdn", full_html=False)
        div1_2 = pio.to_html(fig1_2, include_plotlyjs="cdn", full_html=False)
        div2 = pio.to_html(fig2, include_plotlyjs="cdn", full_html=False)
        div4 = pio.to_html(fig4, include_plotlyjs="cdn", full_html=False)
        div5 = pio.to_html(fig5, include_plotlyjs="cdn", full_html=False)
        div6 = pio.to_html(fig6, include_plotlyjs="cdn", full_html=False)
        div7 = pio.to_html(fig7, include_plotlyjs="cdn", full_html=False)
        div8 = pio.to_html(fig8, include_plotlyjs="cdn", full_html=False)
        div9 = pio.to_html(fig9, include_plotlyjs="cdn", full_html=False)
        div10 = pio.to_html(fig10, include_plotlyjs="cdn", full_html=False)
        div11 = pio.to_html(fig11, include_plotlyjs="cdn", full_html=False)
        div12 = pio.to_html(fig12, include_plotlyjs="cdn", full_html=False)
        div13 = pio.to_html(fig13, include_plotlyjs="cdn", full_html=False)
        
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
        <div class="plot-box">{div1}</div>
        <div class="plot-box">{div1_1}</div>
        <div class="plot-box">{div7}</div>
        <div class="plot-box">{div8}</div>
        <div class="plot-box">{div1_2}</div>
        <div class="plot-box">{div1_3}</div>
        <div class="plot-box">{div6}</div>
        <div class="plot-box">{div2}</div>
        <div class="plot-box">{div4}</div>
        <div class="plot-box">{div5}</div>
        <div class="plot-box">{div9}</div>
        <div class="plot-box">{div10}</div>
        <div class="plot-box">{div11}</div>
        <div class="plot-box">{div12}</div>
        <div class="plot-box">{div13}</div>
        </div>
        </body>
        </html>
        """
        return html
