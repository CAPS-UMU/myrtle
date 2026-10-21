import json
import pathlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Directory configurations
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "myrtle" / "tile_selection" / "out"
OUT_DIR = SCRIPT_DIR / "out"

EXPERIMENTS = [
    ("128x128x128", "Bert Tiny"),
    ("256x256x256", "Bert Mini"),
    ("384x384x384", "Mini LM"),
    ("192x384x384", "BGE Small"),
    ("128x768x768", "Roberta"),
    ("512x512x512", "Bert"),
]

# Two-line experiment labels used for synchronized grouping and x-axis ordering
EXPERIMENT_LABELS = [f"{exp_id}<br>{model_name}" for exp_id, model_name in EXPERIMENTS]


def load_experiment_data(experiments, data_dir):
    """Loads selection method comparison rows and methodology summary stats."""
    sel_method_records = []
    stats_records = []

    for exp_id, model_name in experiments:
        sel_method_file = data_dir / f"{exp_id}w-sel-method-comparison.csv"
        stats_file = data_dir / f"{exp_id}w-methodology-stats.csv"

        if not sel_method_file.exists():
            print(f"Warning: Missing file {sel_method_file}")
            continue
        if not stats_file.exists():
            print(f"Warning: Missing file {stats_file}")
            continue

        df_stats = pd.read_csv(stats_file)
        first_stats_row = df_stats.iloc[0]

        # Use MxNxK as the first line and model name as the second line
        mxnxk_val = (
            str(first_stats_row["MxNxK"])
            if "MxNxK" in first_stats_row
            else exp_id
        )
        exp_label = f"{mxnxk_val}<br>{model_name}"

        # 1. Selection method comparison records
        df_sel = pd.read_csv(sel_method_file)
        for _, row in df_sel.iterrows():
            diff_val = float(row["diff"])
            max_diff_val = (
                float(row["max_diff"]) if pd.notna(row.get("max_diff")) else diff_val
            )
            min_diff_val = (
                float(row["min_diff"]) if pd.notna(row.get("min_diff")) else diff_val
            )

            diff_text = (
                f"{diff_val:.2f}%" if 0 < diff_val < 10 else f"{diff_val:.1f}%"
            )

            sel_method_records.append(
                {
                    "Experiment": exp_label,
                    "sel_method": str(row["sel_method"]),
                    "exp_name": str(row.get("exp_name", exp_label)),
                    "max": row.get("max", None),
                    "min": row.get("min", None),
                    "avg": row.get("avg", None),
                    "max_ts": str(row["max_ts"]) if pd.notna(row.get("max_ts")) else "",
                    "min_ts": str(row["min_ts"]) if pd.notna(row.get("min_ts")) else "",
                    "avg_ts": str(row["avg_ts"]) if pd.notna(row.get("avg_ts")) else "",
                    "diff": diff_val,
                    "max_diff": max_diff_val,
                    "min_diff": min_diff_val,
                    "diff_text": diff_text,
                }
            )

        # 2. Stats records (4 vertical bars per experiment)
        timed_pct = float(first_stats_row["percentage-of-full-timed"])
        pruned_pct = float(
            first_stats_row["percentage-of-full-survived-pruning"]
        )
        ss_exhaustive = str(first_stats_row["SS-exhaustive"])

        suggested_pct = float(first_stats_row["percentage-of-full-suggested"])
        ss_suggested = str(first_stats_row["SS-suggested"])

        stats_records.extend(
            [
                {
                    "Experiment": exp_label,
                    "Stage": "timed",
                    "Percentage": timed_pct,
                    "BarText": (
                        f"{timed_pct:.2f}%"
                        if timed_pct < 10
                        else f"{timed_pct:.1f}%"
                    ),
                },
                {
                    "Experiment": exp_label,
                    "Stage": "pruned SS",
                    "Percentage": pruned_pct,
                    "BarText": (
                        f"{pruned_pct:.2f}%"
                        if pruned_pct < 10
                        else f"{pruned_pct:.1f}%"
                    ),
                },
                {
                    "Experiment": exp_label,
                    "Stage": "full SS",
                    "Percentage": 100.0,
                    "BarText": ss_exhaustive,
                },
                {
                    "Experiment": exp_label,
                    "Stage": "suggested",
                    "Percentage": suggested_pct,
                    "BarText": ss_suggested,
                },
            ]
        )

    return pd.DataFrame(sel_method_records), pd.DataFrame(stats_records)


def create_diff_graph(df_sel_method):
    """Creates the first interactive bar graph with error bars and tile size toggle."""
    unique_methods = df_sel_method["sel_method"].unique().tolist()

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set2
    color_map = {
        method: palette[i % len(palette)] for i, method in enumerate(unique_methods)
    }

    fig1 = go.Figure()

    text_pct_only = []
    text_with_names = []

    for method in unique_methods:
        df_sub = df_sel_method[df_sel_method["sel_method"] == method].copy()

        # Align with EXPERIMENT_LABELS to keep grouped bar slots synchronized
        df_sub = (
            df_sub.set_index("Experiment")
            .reindex(EXPERIMENT_LABELS)
            .reset_index()
        )

        pct_only = [
            f"{row['diff_text']}" if pd.notna(row["diff"]) else ""
            for _, row in df_sub.iterrows()
        ]
        # Only display min_ts below the diff text, without prefix
        with_names = [
            (
                f"{row['diff_text']}<br>"
                f"<span style='font-size:10px;'>{row['min_ts']}</span>"
            )
            if pd.notna(row["diff"])
            else ""
            for _, row in df_sub.iterrows()
        ]

        text_pct_only.append(pct_only)
        text_with_names.append(with_names)

        # Asymmetric error bars:
        # upper extension = max_diff - diff
        # lower extension = diff - min_diff
        error_plus = [
            max(0.0, row["max_diff"] - row["diff"]) if pd.notna(row["diff"]) else 0.0
            for _, row in df_sub.iterrows()
        ]
        error_minus = [
            max(0.0, row["diff"] - row["min_diff"]) if pd.notna(row["diff"]) else 0.0
            for _, row in df_sub.iterrows()
        ]

        fig1.add_trace(
            go.Bar(
                name=method,
                x=df_sub["Experiment"],
                y=df_sub["diff"],
                text=pct_only,  # Default: diff percentage only
                textposition="outside",
                cliponaxis=False,
                marker_color=color_map[method],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    thickness=1.2,
                    width=4,
                    color="#222222",
                ),
                customdata=df_sub[
                    ["min_ts", "max_ts", "diff_text", "min_diff", "max_diff"]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Method: " + method + "<br>"
                    "Difference: %{customdata[2]}<br>"
                    "Min Diff: %{customdata[3]:.2f}%<br>"
                    "Max Diff: %{customdata[4]:.2f}%<br>"
                    "Min TS: %{customdata[0]}<br>"
                    "Max TS: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig1.update_layout(
        barmode="group",
        bargap=0.10,
        bargroupgap=0.02,
        xaxis=dict(
            title="Experiment (MxNxK)",
            type="category",
            categoryorder="array",
            categoryarray=EXPERIMENT_LABELS,
        ),
        yaxis=dict(
            title="% Difference from Fastest Observed",
            automargin=True,
        ),
        legend=dict(
            title_text="Selection Method",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        template="plotly_white",
        margin=dict(t=60, b=60, l=60, r=180),
        title="Cycle Count % Difference Across Selection Methods",
    )

    return fig1, text_pct_only, text_with_names


def create_search_space_graph(df_stats):
    """Creates the second interactive bar graph: 6 groups of 4 vertical bars."""
    stages = ["timed", "pruned SS", "full SS", "suggested"]

    color_discrete_map = {
        "timed": px.colors.qualitative.Plotly[0],
        "pruned SS": px.colors.qualitative.Plotly[1],
        "full SS": px.colors.qualitative.Plotly[2],
        "suggested": "gray",
    }

    fig2 = px.bar(
        df_stats,
        x="Experiment",
        y="Percentage",
        color="Stage",
        text="BarText",
        barmode="group",
        orientation="v",
        category_orders={"Stage": stages},
        color_discrete_map=color_discrete_map,
        labels={
            "Percentage": "Percentage Timed / Surviving (%)",
            "Stage": "Stage",
            "Experiment": "Experiment (MxNxK)",
        },
        title="Search Space Pruning and Timing Coverage",
    )

    fig2.for_each_trace(
        lambda trace: trace.update(
            textposition=(
                "inside"
                if trace.name in ["full SS", "suggested"]
                else "outside"
            ),
            cliponaxis=False,
        )
    )

    fig2.update_layout(
        xaxis=dict(
            title="Experiment (MxNxK)",
            type="category",
            categoryorder="array",
            categoryarray=EXPERIMENT_LABELS,
        ),
        yaxis=dict(title="Percentage (%)", range=[0, 115]),
        legend_title_text="Stage",
        bargap=0.15,
        template="plotly_white",
        margin=dict(t=50, b=60, l=50, r=40),
    )
    return fig2


def export_dashboard(
    fig1, fig2, text_pct_only, text_with_names, out_path, webpage_title
):
    """Embeds both figures into an HTML file with the single 'Show tiles' toggle button."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_1 = fig1.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="diff_graph"
    )
    html_2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    pct_only_json = json.dumps(text_pct_only)
    with_names_json = json.dumps(text_with_names)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{webpage_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background-color: #f7f9fb;
            margin: 0;
            padding: 24px;
            color: #222;
        }}
        h1 {{
            font-size: 24px;
            border-bottom: 2px solid #e1e4e8;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .graph-card {{
            position: relative;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 16px;
            margin-bottom: 30px;
        }}
        /* Toggle button placed to the right below the color legend */
        #toggleTileSizesBtn {{
            position: absolute;
            right: 28px;
            top: 260px;
            z-index: 100;
            background-color: #ffffff;
            border: 1px solid #c5cbd3;
            border-radius: 4px;
            padding: 6px 14px;
            font-size: 12px;
            font-weight: 500;
            color: #333333;
            cursor: pointer;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
            transition: all 0.15s ease-in-out;
        }}
        #toggleTileSizesBtn:hover {{
            background-color: #f3f4f6;
            border-color: #9aa2ac;
        }}
        #toggleTileSizesBtn.active {{
            background-color: #2b6cb0;
            border-color: #2b6cb0;
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <h1>{webpage_title}</h1>

    <div class="graph-card">
        <button id="toggleTileSizesBtn" onclick="toggleTileSizes()">Show tiles</button>
        {html_1}
    </div>

    <div class="graph-card">
        {html_2}
    </div>

    <script>
        const pctOnlyLabels = {pct_only_json};
        const withNamesLabels = {with_names_json};
        let tilesVisible = false;

        function toggleTileSizes() {{
            const btn = document.getElementById("toggleTileSizesBtn");
            const graphDiv = document.getElementById("diff_graph");

            tilesVisible = !tilesVisible;

            if (tilesVisible) {{
                Plotly.restyle(graphDiv, {{ text: withNamesLabels }});
                btn.classList.add("active");
            }} else {{
                Plotly.restyle(graphDiv, {{ text: pctOnlyLabels }});
                btn.classList.remove("active");
            }}
        }}
    </script>
</body>
</html>
"""

    out_path.write_text(full_html, encoding="utf-8")
    print(f"Generated dashboard saved to: {out_path}")


def main():
    df_sel_method, df_stats = load_experiment_data(EXPERIMENTS, DATA_DIR)

    if df_sel_method.empty or df_stats.empty:
        print("Error: No data loaded. Check directory paths and CSV files.")
        return

    fig1, text_pct_only, text_with_names = create_diff_graph(df_sel_method)
    fig2 = create_search_space_graph(df_stats)

    out_file = OUT_DIR / "comparison-plot.html"
    export_dashboard(
        fig1,
        fig2,
        text_pct_only,
        text_with_names,
        out_file,
        webpage_title="Myrtle Selection Method Comparisons and Search Space Coverage",
    )


if __name__ == "__main__":
    main()