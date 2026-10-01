import pathlib
import pandas as pd


def vizSPM(df: pd.DataFrame) -> pathlib.Path:
    """Generates an HTML page visualizing the SPM layout, bank distribution,

    focused buffer view of A[0]/B[0], and 'Last k Iter' figure.

    Buffer B[0] is partitioned into groups of n cells and subgroups of 8 cells.
    Each cell displays 'X.Y' where X is the subgroup number and Y is the group
    number.
    The first cell in every n-cell group uses black text.
    """
    row = df.iloc[0]
    scheme_name = str(row["FakeNN JSON Name"])

    m = int(row["m"])
    n = int(row["n"])
    k = int(row["k"])

    tileA = m * k
    tileB = n * k
    tileC = m * n

    # SPM dimensions (128 KiB @ 8B/cell, 32 banks)
    num_rows = 512
    num_cols = 32
    total_cells = num_rows * num_cols  # 16,384

    EMPTY_COLOR = "#f0f0f0"

    color_palette = {
        "Thread Stacks": "#808080",  # Gray
        "A[0]": "#2ca02c",  # Green
        "A[1]": "#98df8a",  # Light Green
        "B[0]": "#1f77b4",  # Blue
        "B[1]": "#aec7e8",  # Light Blue
        "C[0]": "#ff7f0e",  # Orange
        "C[1]": "#ffbb78",  # Light Orange
        "Empty / Unallocated": EMPTY_COLOR,
    }

    # The 7 buffer allocation types in order
    buffer_names = [
        "Thread Stacks",
        "A[0]",
        "B[0]",
        "C[0]",
        "A[1]",
        "B[1]",
        "C[1]",
    ]

    allocations = [
        ("Thread Stacks", 2048),
        ("A[0]", tileA),
        ("B[0]", tileB),
        ("C[0]", tileC),
        ("A[1]", tileA),
        ("B[1]", tileB),
        ("C[1]", tileC),
    ]

    name_to_summary_row = {name: idx for idx, name in enumerate(buffer_names)}
    summary_counts = [[0] * num_cols for _ in range(len(buffer_names))]

    cell_labels = [None] * total_cells

    current_idx = 0
    for name, size in allocations:
        start_idx = current_idx
        end_idx = min(current_idx + size, total_cells)
        r_type = name_to_summary_row[name]

        for i in range(start_idx, end_idx):
            cell_labels[i] = name
            col = i % num_cols
            summary_counts[r_type][col] += 1

        current_idx += size
        if current_idx >= total_cells:
            break

    # Group end definitions for A[0] (8 groups of (m / 8) * k cells)
    a0_start = 2048
    a0_end = a0_start + tileA
    group_size_A = int((m / 8) * k)
    a0_group_end_indices = set()
    if group_size_A > 0:
        for g in range(8):
            last_cell = a0_start + ((g + 1) * group_size_A) - 1
            if last_cell < min(a0_end, total_cells):
                a0_group_end_indices.add(last_cell)

    # Boundaries and parameters for B[0]
    b0_start = a0_start + tileA
    b0_end = b0_start + tileB

    # Helper function to generate cell label and styling for B[0]
    def get_b0_info(cell_index: int):
        rel_idx = cell_index - b0_start
        group_Y = rel_idx // n if n > 0 else 0
        rem_in_group = rel_idx % n if n > 0 else 0
        subgroup_X = rem_in_group // 8
        label_text = f"{subgroup_X}.{group_Y}"
        # The first cell in every group of n cells has black text
        is_first_in_group = rem_in_group == 0
        text_color = "#000000" if is_first_in_group else "#ffffff"
        return label_text, text_color, is_first_in_group, subgroup_X, group_Y

    # Group start definitions for C[0] (8 groups of (m / 8) * n cells)
    c0_start = b0_start + tileB
    c0_end = c0_start + tileC
    group_size_C = int((m / 8) * n)
    c0_group_start_indices = set()
    if group_size_C > 0:
        for g in range(8):
            s_cell = c0_start + (g * group_size_C)
            if s_cell < min(c0_end, total_cells):
                c0_group_start_indices.add(s_cell)

    # Build Column Header Indices for 32 banks
    spm_header_cells = [
        f'<div class="spm-header-cell">{c}</div>' for c in range(num_cols)
    ]
    spm_header_markup = "\n            ".join(spm_header_cells)

    # Build Left Grid HTML (Figure 1: Full SPM Cells: 512 x 32)
    grid_cells_html = []
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            label = cell_labels[idx]
            cell_text = ""
            text_color = "#ffffff"

            if label is not None:
                color = color_palette[label]
                if label == "A[0]":
                    rel_idx = idx - a0_start
                    group_num = (
                        min(7, rel_idx // group_size_A)
                        if group_size_A > 0
                        else 0
                    )
                    cell_text = str(group_num)

                    if idx in a0_group_end_indices:
                        text_color = "#000000"
                        title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): A[0] [Group {group_num} END]"
                    else:
                        text_color = "#ffffff"
                        title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): A[0] [Group {group_num}]"
                elif label == "B[0]":
                    cell_text, text_color, is_first, sg_x, g_y = get_b0_info(
                        idx
                    )
                    tag = " [GROUP START]" if is_first else ""
                    title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): B[0] [Subgroup {sg_x}, Group {g_y} ({cell_text})]{tag}"
                elif label == "C[0]":
                    rel_idx = idx - c0_start
                    group_num = (
                        min(7, rel_idx // group_size_C)
                        if group_size_C > 0
                        else 0
                    )
                    cell_text = str(group_num)

                    if idx in c0_group_start_indices:
                        text_color = "#000000"
                        title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): C[0] [Group {group_num} START]"
                    else:
                        text_color = "#ffffff"
                        title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): C[0] [Group {group_num}]"
                else:
                    title_tooltip = (
                        f"Cell {idx} (Row {r}, Bank/Col {c}): {label}"
                    )
            else:
                color = color_palette["Empty / Unallocated"]
                title_tooltip = (
                    f"Cell {idx} (Row {r}, Bank/Col {c}): Unallocated"
                )

            grid_cells_html.append(
                f'<div class="cell" style="background-color: {color}; color: {text_color};" title="{title_tooltip}">{cell_text}</div>'
            )
    cells_markup = "\n        ".join(grid_cells_html)

    # Build Figure 2: Summary Distribution Grid (7 rows x 32 cols)
    summary_cells_html = []
    for r_idx, buf_name in enumerate(buffer_names):
        buf_color = color_palette[buf_name]
        for c in range(num_cols):
            cnt = summary_counts[r_idx][c]
            tooltip = f"{buf_name} in Bank/Col {c}: {cnt} cells"
            summary_cells_html.append(
                f'<div class="summary-cell" style="background-color: {buf_color};" title="{tooltip}">{cnt}</div>'
            )
    summary_cells_markup = "\n        ".join(summary_cells_html)

    # Build Figure 3: Focused A[0] and B[0] Subset
    start_row_ab = a0_start // num_cols
    end_row_ab = (
        (b0_end - 1) // num_cols if b0_end > a0_start else start_row_ab
    )
    start_row_ab = min(start_row_ab, num_rows - 1)
    end_row_ab = min(end_row_ab, num_rows - 1)
    subset_num_rows = max(1, end_row_ab - start_row_ab + 1)

    subset_cells_html = []
    for r in range(start_row_ab, end_row_ab + 1):
        for c in range(num_cols):
            idx = r * num_cols + c
            label = cell_labels[idx]
            cell_text = ""
            text_color = "#ffffff"

            if label is not None:
                color = color_palette[label]
                if label == "A[0]":
                    rel_idx = idx - a0_start
                    group_num = (
                        min(7, rel_idx // group_size_A)
                        if group_size_A > 0
                        else 0
                    )
                    cell_text = str(group_num)

                    if idx in a0_group_end_indices:
                        text_color = "#000000"
                        title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): A[0] [Group {group_num} END]"
                    else:
                        text_color = "#ffffff"
                        title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): A[0] [Group {group_num}]"
                elif label == "B[0]":
                    cell_text, text_color, is_first, sg_x, g_y = get_b0_info(
                        idx
                    )
                    tag = " [GROUP START]" if is_first else ""
                    title_tooltip = f"Cell {idx} (Row {r}, Bank/Col {c}): B[0] [Subgroup {sg_x}, Group {g_y} ({cell_text})]{tag}"
                else:
                    title_tooltip = (
                        f"Cell {idx} (Row {r}, Bank/Col {c}): {label}"
                    )
            else:
                color = color_palette["Empty / Unallocated"]
                title_tooltip = (
                    f"Cell {idx} (Row {r}, Bank/Col {c}): Unallocated"
                )

            subset_cells_html.append(
                f'<div class="subset-cell" style="background-color: {color}; color: {text_color};" title="{title_tooltip}">{cell_text}</div>'
            )
    subset_cells_markup = "\n            ".join(subset_cells_html)

    # Build Figure 4: Last k Iter (Concatenation of rows in A[0] and C[0] containing black text)
    a0_first_row = a0_start // num_cols
    a0_last_row = min((a0_end - 1) // num_cols, num_rows - 1)
    a0_black_rows = []
    for r in range(a0_first_row, a0_last_row + 1):
        row_indices = range(r * num_cols, (r + 1) * num_cols)
        if any(idx in a0_group_end_indices for idx in row_indices):
            a0_black_rows.append(r)

    c0_first_row = c0_start // num_cols
    c0_last_row = min((c0_end - 1) // num_cols, num_rows - 1)
    c0_black_rows = []
    for r in range(c0_first_row, c0_last_row + 1):
        row_indices = range(r * num_cols, (r + 1) * num_cols)
        if any(idx in c0_group_start_indices for idx in row_indices):
            c0_black_rows.append(r)

    fig4_rows = a0_black_rows + c0_black_rows
    fig4_row_count = max(1, len(fig4_rows))

    fig4_cells_html = []
    for r in fig4_rows:
        for c in range(num_cols):
            idx = r * num_cols + c
            label = cell_labels[idx]
            cell_text = ""
            text_color = "#ffffff"

            if label is not None:
                color = color_palette[label]
                if label == "A[0]":
                    rel_idx = idx - a0_start
                    group_num = (
                        min(7, rel_idx // group_size_A)
                        if group_size_A > 0
                        else 0
                    )
                    cell_text = str(group_num)
                    if idx in a0_group_end_indices:
                        text_color = "#000000"
                        title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: A[0] [Group {group_num} END]"
                    else:
                        title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: A[0] [Group {group_num}]"
                elif label == "B[0]":
                    cell_text, text_color, is_first, sg_x, g_y = get_b0_info(
                        idx
                    )
                    tag = " [GROUP START]" if is_first else ""
                    title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: B[0] [Subgroup {sg_x}, Group {g_y} ({cell_text})]{tag}"
                elif label == "C[0]":
                    rel_idx = idx - c0_start
                    group_num = (
                        min(7, rel_idx // group_size_C)
                        if group_size_C > 0
                        else 0
                    )
                    cell_text = str(group_num)
                    if idx in c0_group_start_indices:
                        text_color = "#000000"
                        title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: C[0] [Group {group_num} START]"
                    else:
                        title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: C[0] [Group {group_num}]"
                else:
                    title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: {label}"
            else:
                color = color_palette["Empty / Unallocated"]
                title_tooltip = f"Fig 4 | Orig Row {r}, Col {c}: Unallocated"

            fig4_cells_html.append(
                f'<div class="subset-cell" style="background-color: {color}; color: {text_color};" title="{title_tooltip}">{cell_text}</div>'
            )
    fig4_cells_markup = "\n            ".join(fig4_cells_html)

    # Build Legend HTML
    legend_items_html = []
    for label, color in color_palette.items():
        legend_items_html.append(
            f"""
        <div class="legend-item">
            <span class="legend-color-box" style="background-color: {color};"></span>
            <span>{label}</span>
        </div>"""
        )
    legend_markup = "\n".join(legend_items_html)

    # Full HTML Document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{scheme_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 24px;
            background-color: #fafafa;
            color: #333;
        }}
        h1 {{
            margin-bottom: 8px;
            font-size: 24px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 24px;
            padding: 10px 14px;
            background: #ffffff;
            border: 1px solid #ddd;
            border-radius: 6px;
            width: fit-content;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 13px;
            gap: 6px;
        }}
        .legend-color-box {{
            width: 14px;
            height: 14px;
            border: 1px solid rgba(0, 0, 0, 0.2);
            border-radius: 2px;
            display: inline-block;
        }}
        .main-container {{
            display: flex;
            gap: 36px;
            align-items: flex-start;
        }}
        .panel {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .right-column {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        .bottom-sub-row {{
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }}
        .panel-title {{
            font-size: 14px;
            font-weight: 600;
            color: #444;
        }}
        /* Column header styling for banks 0-31 */
        .spm-header-grid {{
            display: grid;
            grid-template-columns: repeat({num_cols}, 16px);
            gap: 1px;
            padding: 1px;
            width: fit-content;
        }}
        .spm-header-cell {{
            width: 16px;
            font-size: 8px;
            text-align: center;
            color: #666;
            user-select: none;
        }}
        /* Left Grid: Cells widened to 16px and heightened to 8px */
        .spm-grid {{
            display: grid;
            grid-template-columns: repeat({num_cols}, 16px);
            grid-template-rows: repeat({num_rows}, 8px);
            gap: 1px;
            background-color: #e0e0e0;
            padding: 1px;
            width: fit-content;
            border: 1px solid #aaa;
            max-height: 85vh;
            overflow-y: auto;
        }}
        .cell {{
            width: 16px;
            height: 8px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 5px;
            font-weight: bold;
            line-height: 1;
            user-select: none;
            overflow: hidden;
            letter-spacing: -0.5px;
        }}
        .cell:hover {{
            outline: 1px solid #000;
            z-index: 10;
        }}
        /* Right Top Grid: 7 rows x 32 cols */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat({num_cols}, 26px);
            grid-template-rows: repeat(7, 26px);
            gap: 1px;
            background-color: #bbb;
            padding: 1px;
            border: 1px solid #888;
            width: fit-content;
        }}
        .summary-cell {{
            width: 26px;
            height: 26px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 500;
            color: #111;
            box-sizing: border-box;
        }}
        .summary-cell:hover {{
            outline: 1px solid #000;
            filter: brightness(0.95);
            z-index: 10;
        }}
        .row-labels {{
            display: flex;
            flex-direction: column;
            gap: 1px;
            margin-right: 6px;
        }}
        .row-label {{
            height: 26px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            font-size: 11px;
            font-weight: bold;
            color: #555;
            padding-right: 4px;
        }}
        /* Subset Grids (Figures 3 and 4) */
        .subset-grid-container {{
            max-height: 52vh;
            overflow-y: auto;
            width: fit-content;
            border: 1px solid #aaa;
            padding: 1px;
            background-color: #e0e0e0;
        }}
        .subset-grid {{
            display: grid;
            grid-template-columns: repeat({num_cols}, 16px);
            gap: 1px;
            width: fit-content;
        }}
        .subset-cell {{
            width: 16px;
            height: 14px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 6px;
            font-weight: bold;
            user-select: none;
            overflow: hidden;
            letter-spacing: -0.5px;
        }}
        .subset-cell:hover {{
            outline: 1px solid #000;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <h1>{scheme_name}</h1>
    <div class="legend">
        {legend_markup}
    </div>

    <div class="main-container">
        <!-- Figure 1: Full SPM Memory Map -->
        <div class="panel">
            <div class="panel-title">SPM Memory Layout (512 rows &times; 32 banks)</div>
            <div class="spm-header-grid">
                {spm_header_markup}
            </div>
            <div class="spm-grid">
                {cells_markup}
            </div>
        </div>

        <!-- Right Side: Figure 2 on top, Figures 3 and 4 side-by-side underneath -->
        <div class="right-column">
            <!-- Figure 2: Bank Distribution Matrix -->
            <div class="panel">
                <div class="panel-title">Bank Distribution Matrix (7 rows &times; 32 banks)</div>
                <div style="display: flex; margin-top: 18px;">
                    <div class="row-labels">
                        {''.join(f'<div class="row-label">{name}</div>' for name in buffer_names)}
                    </div>
                    <div class="summary-grid">
                        {summary_cells_markup}
                    </div>
                </div>
            </div>

            <!-- Lower Row: Figure 3 (Left) and Figure 4 (Right) -->
            <div class="bottom-sub-row">
                <!-- Figure 3: Focused A[0] & B[0] Subset -->
                <div class="panel">
                    <div class="panel-title">Focused Buffer View: A[0] &amp; B[0] (Rows {start_row_ab} to {end_row_ab})</div>
                    <div class="spm-header-grid">
                        {spm_header_markup}
                    </div>
                    <div class="subset-grid-container">
                        <div class="subset-grid" style="grid-template-rows: repeat({subset_num_rows}, 14px);">
                            {subset_cells_markup}
                        </div>
                    </div>
                </div>

                <!-- Figure 4: Last k Iter -->
                <div class="panel">
                    <div class="panel-title">Last k Iter ({len(fig4_rows)} Rows)</div>
                    <div class="spm-header-grid">
                        {spm_header_markup}
                    </div>
                    <div class="subset-grid-container">
                        <div class="subset-grid" style="grid-template-rows: repeat({fig4_row_count}, 14px);">
                            {fig4_cells_markup}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    out_dir = pathlib.Path(__file__).parent.resolve() / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{scheme_name}-viz.html"
    out_file.write_text(html_content, encoding="utf-8")

    return out_file