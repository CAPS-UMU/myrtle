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
            "tileA_cc",
            "tileB",
            "tileC",
            "cost",
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
                  color="Kernel Time",  # Optional: color points by a category column
                  hover_data=hover_data,  # Show these columns on hover
                  title=f"{title} {x_col} vs {y_col}",)
             html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
             div=f"""<div class="plot-box">{html}</div>"""
             figs.append(div)
        #print('\n'.join(figs))
        feature_graphs='\n'.join(figs)


        x_col = "L1 Usage"
        y_col = "Kernel Time"
        fig1 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="fmaddsPerCore",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "fmaddsPerCore"
        y_col = "Kernel Time"
        fig2 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Regular Loads",#"L3 L/S Timed",  # "Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        x_col = "SSR Config Count"
        y_col = "Kernel Time"
        fig3 = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Regular Loads",  # Optional: color points by a category column
            hover_data=hover_data,  # Show these columns on hover
            title=f"{title} {x_col} vs {y_col}",
        )

        # --- Convert each figure to HTML div ---

        div10 = pio.to_html(fig10, include_plotlyjs="cdn", full_html=False)
        div20 = pio.to_html(fig20, include_plotlyjs="cdn", full_html=False)
        div1 = pio.to_html(fig1, include_plotlyjs="cdn", full_html=False)
        div2 = pio.to_html(fig2, include_plotlyjs="cdn", full_html=False)
        div3 = pio.to_html(fig3, include_plotlyjs="cdn", full_html=False)
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
        </div>
        </body>
        </html>
        """
        return html
