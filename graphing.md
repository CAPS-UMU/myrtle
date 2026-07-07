# graphing and exporting to PDFs

- clone myrtle repo

- use `graph-paper-data-q.sh`, which calls `graph-quidditch.py`, which calls `visualizer.py`, which uses a function from `graphUtils.py` called **`genResultGraphQPDF`**

Example run:

```
cd scripts
bash graph-paper-data-q.sh dims-csv-name-line-by-line-paper-q.input
```

where `dims-csv-name-line-by-line-paper-q.input` is in the scripts directory and looks like:
```
1x600x400 /home/emily/myrtle/sensitivity-analysis/holistic-data/1x600x400wm-n-k-graphing-logistics.csv
1x600x600 /home/emily/myrtle/sensitivity-analysis/holistic-data/1x600x600wm-n-k-graphing-logistics.csv
1x1200x400 /home/emily/myrtle/sensitivity-analysis/holistic-data/1x1200x400wm-n-k-graphing-logistics.csv
1x400x161 /home/emily/myrtle/sensitivity-analysis/holistic-data/1x400x161wm-n-k-graphing-logistics.csv
```

^^ replace "home/emily/myrtle" with path to your clone of myrtle repo



# About **genResultGraphQPDF**

It takes in a csv file, and then calls a library func that basically returns a **plotly express graph**. If you can produce a plotly graph of your data, this function adds styling to the plotly graph to make it look like latex, and then it outputs the plotly graph to a PDF with `fig.write_image(f"out/{title}.pdf", width=1200, height=800, scale=3)`.

Most of the formatting stuff you need is inside here:

```
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
    


    fig.write_image(f"out/{title}.pdf", width=1200, height=800, scale=3)
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
```

