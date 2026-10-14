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
    "128x128x128",
    "256x256x256",
    "384x384x384",
    "512x512x512",
    "128x768x768",
    "192x384x384"
]


def load_experiment_data(experiments, data_dir):
    """Loads baseline comparison rows and methodology summary stats."""
    baseline_records = []
    stats_records = []

    for exp_id in experiments:
        baseline_file = data_dir / f"{exp_id}-baseline-best-myrtle.csv"
        stats_file = data_dir / f"{exp_id}-methodology-stats.csv"

        if not baseline_file.exists():
            print(f"Warning: Missing file {baseline_file}")
            continue
        if not stats_file.exists():
            print(f"Warning: Missing file {stats_file}")
            continue

        df_stats = pd.read_csv(stats_file)
        first_stats_row = df_stats.iloc[0]

        # Use the "MxNxK" column value as the experiment identifier
        exp_label = (
            str(first_stats_row["MxNxK"])
            if "MxNxK" in first_stats_row
            else exp_id
        )

        # 1. Baseline records (4 vertical bars per experiment)
        df_baseline = pd.read_csv(baseline_file)
        for _, row in df_baseline.iterrows():
            diff_val = float(row["diff"])
            json_name = str(row["JSON Name"])
            diff_text = (
                f"{diff_val:.2f}%" if 0 < diff_val < 10 else f"{diff_val:.1f}%"
            )

            baseline_records.append(
                {
                    "Experiment": exp_label,
                    "JSON Name": json_name,
                    "diff": diff_val,
                    "diff_text": diff_text,
                    "point_label": str(row["point_label"]),
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

    return pd.DataFrame(baseline_records), pd.DataFrame(stats_records)


def create_diff_graph(df_baseline):
    """Creates the first interactive bar graph.

    Returns the figure and the two label sets (percentage-only and
    percentage+JSON Name) for the interactive single-button toggle.
    """
    palette = px.colors.qualitative.Plotly
    unique_labels = df_baseline["point_label"].unique().tolist()
    color_map = {
        lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)
    }

    fig1 = go.Figure()

    text_pct_only = []
    text_with_names = []

    for label in unique_labels:
        df_sub = df_baseline[df_baseline["point_label"] == label].copy()

        pct_only = [f"{row['diff_text']}" for _, row in df_sub.iterrows()]
        with_names = [
            f"{row['diff_text']}<br><span style='font-size:10px;'>{row['JSON Name']}</span>"
            for _, row in df_sub.iterrows()
        ]

        text_pct_only.append(pct_only)
        text_with_names.append(with_names)

        fig1.add_trace(
            go.Bar(
                name=label,
                x=df_sub["Experiment"],
                y=df_sub["diff"],
                text=pct_only,  # Default: hidden JSON names
                textposition="outside",
                cliponaxis=False,
                marker_color=color_map[label],
                customdata=df_sub[["JSON Name", "diff_text"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Experiment: %{x}<br>"
                    "Difference: %{customdata[1]}<br>"
                    "Label: " + label + "<extra></extra>"
                ),
            )
        )

    fig1.update_layout(
        barmode="group",
        bargap=0.12,
        bargroupgap=0.03,
        xaxis=dict(
            title="Experiment (MxNxK)",
            type="category",
            categoryorder="array",
            categoryarray=EXPERIMENTS,
        ),
        yaxis=dict(
            title="% Difference from Fastest Observed",
            automargin=True,
        ),
        legend=dict(
            title_text="Point Label",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        template="plotly_white",
        margin=dict(t=60, b=40, l=60, r=160),
        title="Cycle Count % Difference vs. Baseline and Best Myrtle",
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
            categoryarray=EXPERIMENTS,
        ),
        yaxis=dict(title="Percentage (%)", range=[0, 115]),
        legend_title_text="Stage",
        bargap=0.15,
        template="plotly_white",
        margin=dict(t=50, b=40, l=50, r=40),
    )
    return fig2


def export_dashboard(
    fig1, fig2, text_pct_only, text_with_names, out_path, webpage_title
):
    """Embeds both figures into an HTML file with a single toggle button."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use explicit element ID for fig1 so the toggle button script can target it
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
        /* Single toggle button positioned on the right, under the legend */
        #toggleTileSizesBtn {{
            position: absolute;
            right: 28px;
            top: 235px;
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
    df_baseline, df_stats = load_experiment_data(EXPERIMENTS, DATA_DIR)

    if df_baseline.empty or df_stats.empty:
        print("Error: No data loaded. Check directory paths and CSV files.")
        return

    fig1, text_pct_only, text_with_names = create_diff_graph(df_baseline)
    fig2 = create_search_space_graph(df_stats)

    out_file = OUT_DIR / "baseline-best-myrtle.html"
    export_dashboard(
        fig1,
        fig2,
        text_pct_only,
        text_with_names,
        out_file,
        webpage_title="Myrtle Experiments Compared with Naive Approaches and Best Observed",
    )


if __name__ == "__main__":
    main()