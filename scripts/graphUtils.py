import plotly.express as px
import plotly.io as pio
theColorBar="haline"
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

def stack_dfs_to_html(df_list, titles, columns_subset, main_title=None, include_index=False):
    """
    Concatenates multiple DataFrames into a single HTML string with section titles.
    Rows are conditionally styled based on 'timed', 'Avg n'_sz / k_size', and 'diff':
    - 'timed' == False -> Gray text (no highlight)
    - 'diff' == 0.0 -> Red background
    - 'diff' <= 5.0 -> Blue background
    - 'Avg n'_sz / k_size' >= 1 -> Entire row is bolded (preserving colors/highlights)
    """
    if len(df_list) != len(titles):
        raise ValueError("The number of DataFrames must match the number of titles.")
        
    html_sections = []
    
    # 1. Add the main title at the very top
    if main_title:
        main_title_style = "style='font-family: Arial, sans-serif; margin-bottom: 30px; color: #111; border-bottom: 2px solid #333; padding-bottom: 10px;'"
        html_sections.append(f"<h1 {main_title_style}>{main_title}</h1>")
    
    title_style = "style='font-family: Arial, sans-serif; margin-top: 25px; margin-bottom: 10px; color: #333;'"
    table_style = "style='border-collapse: collapse; width: 75%; font-family: Arial, sans-serif; margin-bottom: 20px;'"

    # Internal helper function to apply the row styles
    def style_rows(row):
        # Default styles (no styling)
        styles = [''] * len(row)
        
        # 1. Check the 'timed' condition first
        if 'timed' in row and row['timed'] is False:
            # Gray text with no highlight background
            styles = ['color: #718096;'] * len(row)
        
        # 2. Fall back to 'diff' conditions if 'timed' is True (or missing)
        elif 'diff' in row:
            diff_val = row['diff']
            if diff_val == 0.0:
                styles = ['background-color: #ffcccc; color: #990000;'] * len(row)
            elif diff_val <= 5.0:
                styles = ['background-color: #d9ecff; color: #004085;'] * len(row)
                
        # 3. Independent condition: Bold the ENTIRE row while preserving other styles
        avg_col = "Avg n'_sz / k_size"
        if avg_col in row and row[avg_col] >= 1:
            # Append bolding to every single column's existing style string
            styles = [style + ' font-weight: bold;' for style in styles]
            
        return styles

    for df, title in zip(df_list, titles):
        html_sections.append(f"<h3 {title_style}>{title}</h3>")
        
        try:
            # 2. Filter to the requested subset of columns
            filtered_df = df[columns_subset]
            
            # 3. Use pandas Styler to apply conditional row coloring/styling
            # axis=1 applies the function row-by-row
            styled_html = (filtered_df.style
                           .apply(style_rows, axis=1)
                           .hide(axis='index' if not include_index else None)
                           .to_html())
            
            # Inject our custom width and border styling into the Styler-generated table
            styled_html = styled_html.replace('<table', f'<table border="1" {table_style}')
            
            html_sections.append(styled_html)
            
        except KeyError as e:
            html_sections.append(f"<p style='color: red;'>Error rendering table: Missing column {e}</p>")
            
    return "\n".join(html_sections)

def saveFigsInHTML(special_figs, more_figs, titleOfWebpage,table=""):
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
<span>{table}</span>
        {"<div>Pruning Step by Step</div>"}
        {moreDivsAsHTML}        
        </div>
        <span id="bottom"><a href="#top" >Back to Top</a></span>
        </body>
        </html>
        """
    return html

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

