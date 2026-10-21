import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import math
import re
import pathlib
import sys
from scripts.graphUtils import scatterWithColorSymbol, plot_3d_scatter,scatterWithFlatColor, genLatexyGraphPDF,scatterWithColor, addScatterFlatColorMarker, stack_dfs_to_html_w_toggle,genShelfGraphPDF,jugaadTitle
import pandas as pd
from typing import List
import numpy as np
from typing import Tuple
import tile_selection.selection_methods as sm
# # 1. Resolve paths
# # Root directory: scripts
# PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# # Inner package directory containing 'tile_static_analysis': myrtle/myrtle
# INNER_MYRTLE = PROJECT_ROOT / "myrtle"

# # 2. Add both to sys.path
# for p in [PROJECT_ROOT, INNER_MYRTLE]:
#     if str(p) not in sys.path:
#         sys.path.append(str(p))

# # 3. Now the import works without breaking internal imports inside TSA_C_Remainder:
# import myrtle.tile_static_analysis.TSA_C_Remainder as TSA
pd.set_option('display.max_rows', None)



def printFinalRankingTwoShelves(title, df, colorCol,shelfMetric='FMADDsMULsPerCore'):
    """
    1. Sorts and slices unique 'FMADDsMULsPerCore' values from smallest to largest.
    2. Groups rows by 'mRem' (ascending) and 'colorCol' (descending) within each slice.
    3. Concatenates all processed slices into a single DataFrame.
    4. Filters the concatenated DataFrame to keep only rows where 'timed' is False.
    5. Returns the HTML table, the full concatenated DataFrame, and the filtered DataFrame.
    """
    df = df.sort_values("Time (cycles)", ascending=True)
    df = df.reset_index(drop=True)
    best = df["Time (cycles)"][0]
    title = f"best observed: {best} cycles {title}"
    
    fmadd_values = []
    processed_dfs = []
    my_titles = [] #Avg m'_sz / k_size
    my_columns = ["JSON Name", "mnkRem", "timed", "Avg n'_sz / k_size","m'_sz*n_sz / k_sz",colorCol,"Avg B'", "Time (cycles)", "diff"]
    my_columns = ["JSON Name", "timed", "Avg m'_sz*n_sz / k_sz", "FMADDsMULsPerCore","mRem","mnk",colorCol,"tileB", "Time (cycles)", "diff"]
    my_columns = ["JSON Name", "timed", "Avg m'_sz*n_sz / k_sz", "mnkRem","mMod8","nMod32",colorCol,"Avg C-M Ratio", "Time (cycles)", "diff"]
    
    # Get unique FMADD values, sort them from smallest to largest
    sorted_fmadd_keys = sorted(df[shelfMetric].unique(), reverse=True)
    
    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df[shelfMetric] == fmadd_value]
        
        # Sort by 'mRem' (Smallest to Largest) and then colorCol (SMALLEST TO LARGEST)
        processed_slice = fmadd_group.sort_values(
            by=['mMod8', colorCol], 
            ascending=[True, True]
        )
        processed_slice = processed_slice.reset_index(drop=True)
        # Append to our parallel output lists
        fmadd_values.append(fmadd_value)
        processed_dfs.append(processed_slice)
        myTitle = f"FMADDS: {fmadd_value} w/ len {len(processed_slice)}"
        my_titles.append(myTitle)

    three_shelves = dict(zip(fmadd_values[:3], processed_dfs[:3]))    
    # 1. Concatenate all processed slices into a single DataFrame by stacking rows
    # (ignoring index ensures a clean, continuous index for the combined df)
    concatenated_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
    
    # 2. Filter for rows where 'timed' value is False
    filtered_df = concatenated_df[concatenated_df['timed'] == False]
    
    # Generate the usual HTML table
    # don't print all the shelves, in fact print a max of 6
    finalShelf = min(20,len(processed_dfs)-1)
    table = stack_dfs_to_html_w_toggle(processed_dfs[0:finalShelf], my_titles[0:finalShelf], my_columns, title)
    #table = stack_dfs_to_html(processed_dfs, my_titles, my_columns, title)
    
    # 3. Return all four values
    return table, concatenated_df, filtered_df, three_shelves

def printMethodologyStats(full, pruned, timed, suggested, verbose=False):
    if verbose:
        print(f"full ss has size {len(full)}")
        print(f"pruned has size {len(pruned)}")
        print(f"timed has size {len(timed)}")
        print(f"what percentage of SPM used in timed points vs pruned points? {len(timed)/len(pruned)*100.0} %")    
    stats = {}
    stats["SS-exhaustive"]=[len(full)]
    stats["SS-pruned-to-third-fewest-syncs"]=[len(pruned)]
    stats["timed-points"]=[len(timed)]
    stats["SS-suggested"]=[len(suggested)]
    stats["percentage-of-full-timed"]= [len(timed)/len(full)*100.0]
    stats["percentage-of-full-suggested"]= [len(suggested)/len(full)*100.0]
    stats["percentage-of-full-survived-pruning"]= [len(pruned)/len(full)*100.0]
    stats["percentage-of-pruned-timed"]= [len(timed)/len(pruned)*100.0]
    return stats

# return the SSR config value that is the largest of the bottom frac
def ssr_prune_frac(df,frac):
        unique_ssr_configs = list(
            set(df["SSR Config Count"].values.tolist())
        )  # remove duplicates
        unique_ssr_configs.sort()  # sort least to greateset
        third = unique_ssr_configs[0:int(len(unique_ssr_configs)/frac)]
        #print(f"{unique_ssr_configs} with len {len(unique_ssr_configs)} and bottom third {third}")
        prunePoint = unique_ssr_configs[int(len(unique_ssr_configs)/frac)]  # prune to smallest frac of ssr_configs   
        return prunePoint

def illustrate_ssr_pruning(pruned, full, more_figs,minimal_hover):
     # step 0: full search space
     x_col = "L1 Usage"
     y_col = "SSR Configs"
     more_figs.append(
         scatterWithFlatColor(
            full,
            x_col,
            y_col,
            "gray",
            minimal_hover,
            "0) full search space",
            "symbolMarker",
        )
     )


    # step 1: pruned search space
     x_col = "L1 Usage"
     y_col = "SSR Configs"
     more_figs.append(
        scatterWithColor(
            pruned,
            x_col,
            y_col,
            "SSR Configs",
            minimal_hover,
            "0.1) Full search space (multicolor points are analyzed by our model)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        more_figs[-1],
        full,
        x_col,
        y_col,
        "gray",
        "circle",
        minimal_hover,
        "full search space"
    )

def catByMDimBoundaryTile(timed, ut, niceOnly=False):
     nice_timed=timed[timed["niceMRem"]].copy()
     nice_ut=ut[ut["niceMRem"]].copy()
     nice_timed["timed"]=True
     nice_ut["timed"] = False
     mean_timed=timed[timed["niceMRem"]==False]
     mean_ut=ut[ut["niceMRem"]==False]
     if niceOnly:
          return nice_timed,nice_ut
     else:
          return nice_timed,nice_ut,mean_timed,mean_ut

def illustrate_prune_nefarious_boundary_tiles(special_figs,hover_data,timed, ut):
     nice_timed,nice_ut,mean_timed,mean_ut = catByMDimBoundaryTile(timed,ut)
     x_col = "m"
     y_col = "Global Sim E2E_dma"
     special_figs.append(
        scatterWithFlatColor(
            nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "1.2) Pruned out m-dim CL boundary tiles not divisible by 8 (marked with red x)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        nice_ut,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed w/ nice m remainder"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        mean_ut,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "untimed w/ worst case m remainder"
    )
     addScatterFlatColorMarker(
        special_figs[-1],
        mean_timed,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "timed w/ worst case m remainder"
    )


def split_longest_initial_subsequence(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col: str = "FakeNN JSON Name",
    sort_col: str = "SSR Configs",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits DataFrames A and B around the longest uninterrupted prefix of B in A.

    Both DataFrames are assumed to be sorted ascending by `sort_col`. Rows are
    matched by `id_col` to tolerate minor variations in other columns.

    Returns:
        prefix_df (pd.DataFrame): The longest contiguous prefix of B present in
        A.
        b_remainder (pd.DataFrame): Remaining rows of B after prefix_df (sorted
        order preserved).
        a_remainder (pd.DataFrame): Rows of A that do not appear in prefix_df
        (sorted order preserved).
    """
    # 1. Validation checks
    shared_cols = [c for c in df_b.columns if c in df_a.columns]
    if not shared_cols:
        raise ValueError("DataFrames A and B do not share any columns.")

    for col in (id_col, sort_col):
        if col not in df_a.columns or col not in df_b.columns:
            raise ValueError(
                f"Required column '{col}' must be present in both DataFrames."
            )

    # Edge cases: empty inputs
    if df_b.empty:
        return df_b.copy(), df_b.copy(), df_a.copy()
    if df_a.empty:
        return df_b.iloc[:0].copy(), df_b.copy(), df_a.copy()

    # 2. Match membership using the unique ID column
    a_ids = set(df_a[id_col].dropna().astype(str))
    in_a = df_b[id_col].astype(str).isin(a_ids).to_numpy()

    # 3. Determine the end index of the contiguous prefix in B
    if not in_a[0]:
        # First row doesn't match -> prefix length is 0
        break_idx = 0
    elif in_a.all():
        # Every row of B is matched
        break_idx = len(df_b)
    else:
        # First mismatch index
        break_idx = int((~in_a).argmax())

    # 4. Slice the prefix and the remainder of B
    prefix_df = df_b.iloc[:break_idx].copy()
    b_remainder = df_b.iloc[break_idx:].copy()

    # 5. Extract rows in A not present in prefix_df, preserving A's sorted order
    prefix_ids = set(prefix_df[id_col].astype(str))
    a_remainder = df_a[~df_a[id_col].astype(str).isin(prefix_ids)].copy()

    return prefix_df, b_remainder, a_remainder


def maxMNK(full):
    full["mnk"] = full["m"]*full["n"]*full["k"]
    full = full.sort_values(by="mnk",ascending=False)
    full = full.reset_index(drop=True)
    best = full[["FakeNN JSON Name","mnk"]].loc[0]
    full["mDiv8"] = full["m"] % 8
    pruned = full[full["mDiv8"]==0].copy()
    pruned = pruned.sort_values(by="mnk",ascending=False)
    pruned = pruned.reset_index(drop=True)
    bestDiv8 = pruned[["FakeNN JSON Name","mnk"]].loc[0]
    return best, bestDiv8, full.copy()

def bestObserved(timed):
    timed = timed.sort_values(by="dma",ascending=True)
    timed = timed.reset_index(drop=True)
    best = timed[["FakeNN JSON Name","dma"]].loc[0]
    return best

def extractPointsForComparison(df, id_list, label_list):
    extracted_rows = []

    for item_id, label in zip(id_list, label_list):
        # Match rows where "FakeNN JSON Name" matches the target ID
        matched = df[df["FakeNN JSON Name"] == item_id].copy()

        if not matched.empty:
            matched["point_label"] = label
            extracted_rows.append(matched)

    if not extracted_rows:
        # Return an empty DataFrame containing original columns plus "label"
        empty_cols = list(df.columns)
        if "label" not in empty_cols:
            empty_cols.append("point_label")
        return pd.DataFrame(columns=empty_cols)

    # Concatenate all extracted rows and reset the index
    result_df = pd.concat(extracted_rows, ignore_index=True)
    return result_df

def getPointIDs(df):
    if df is None:
        return []
    else:
    #print(list(df["FakeNN JSON Name"]))
        return list(df["FakeNN JSON Name"])
def getExpName(df):
    m=df["M"][0]
    n=df["N"][0]
    k=df["K"][0]
    return f"{m}x{n}x{k}" 
    
# def maxMinAvg(timed,point_ids, point_labels)
#     points = timed[timed[point_ids]]
#     if len(point_ids) == 1:
# def maxMinAvg(timed, point_ids, point_labels):
#     # Map to store stats per label: label -> (min_dma, max_dma, avg_dma)
#     label_stats = {}

#     # Step 1-3: Slice timed df and compute (min, max, avg) for each point_id_list
#     for id_list, label in zip(point_ids, point_labels):
#         slice_df = timed[timed["FakeNN JSON Name"].isin(id_list)]

#         if slice_df.empty:
#             continue

#         dma_series = slice_df["dma"]
#         n_rows = len(slice_df)

#         if n_rows == 1:
#             val = dma_series.iloc[0]
#             stats = (val, val, val)
#         else:
#             stats = (dma_series.min(), dma_series.max(), dma_series.mean())

#         label_stats[label] = stats

#     # Retrieve the average dma for "fastest observed"
#     baseline_avg = None
#     if "fastest observed" in label_stats:
#         baseline_avg = label_stats["fastest observed"][2]

#     # Step 4: Zip and print results for each pair
#     for id_list, label in zip(point_ids, point_labels):
#         if label not in label_stats:
#             print(f"Point Label: {label} (No matching rows found in dataframe)")
#             continue

#         min_val, max_val, avg_val = label_stats[label]

#         # 4.1 Print label and (min, max, avg)
#         print(f"Point Label: {label}")
#         print(
#             f"  DMA - Min: {min_val:.4f}, Max: {max_val:.4f}, Avg: {avg_val:.4f}"
#         )

#         # 4.2 Print percentage difference against 'fastest observed'
#         if baseline_avg is not None:
#             pct_diff = ((avg_val - baseline_avg) / baseline_avg) * 100.0
#             print(
#                 f"  % Difference vs 'fastest observed': {pct_diff:+.2f}%\n"
#             )
#         else:
#             print(
#                 "  % Difference: 'fastest observed' baseline not found in labels\n"
#             )
import numpy as np
import pandas as pd


def maxMinAvg(expName, timed, point_ids, point_labels, verbose=False):
    # Step 1: Pre-calculate slice statistics and best/worst rows
    processed_stats = {}

    for id_list, label in zip(point_ids, point_labels):
        # Subset timed dataframe by matching "FakeNN JSON Name"
        slice_df = timed[timed["FakeNN JSON Name"].isin(id_list)]

        if slice_df.empty:
            continue

        dma_series = slice_df["dma"]
        min_val = dma_series.min()
        max_val = dma_series.max()
        avg_val = dma_series.mean()

        # Find row corresponding to the minimum DMA value
        min_row = slice_df.loc[slice_df["dma"] == min_val].iloc[0]
        min_ts = min_row["JSON Name"]

        # Find row corresponding to the maximum DMA value
        max_row = slice_df.loc[slice_df["dma"] == max_val].iloc[0]
        max_ts = max_row["JSON Name"]

        # avg_ts rules: contains max_ts when max == min, otherwise 'n/a'
        if max_val == min_val:
            avg_ts = max_ts
        else:
            avg_ts = "n/a"

        processed_stats[label] = {
            "min": min_val,
            "max": max_val,
            "avg": avg_val,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "avg_ts": avg_ts,
        }

    # Baseline statistics for 'fastest observed'
    baseline_avg = None
    baseline_max = None
    baseline_min = None
    if "fastest observed" in processed_stats:
        baseline_avg = processed_stats["fastest observed"]["avg"]
        baseline_max = processed_stats["fastest observed"]["max"]
        baseline_min = processed_stats["fastest observed"]["min"]

    # Step 2: Print expName before printing point label statistics (only if verbose)
    if verbose:
        print(f"Experiment: {expName}")

    # Step 3 & 4: Print stats per label (if verbose) and populate output rows
    result_rows = []

    for id_list, label in zip(point_ids, point_labels):
        if label not in processed_stats:
            if verbose:
                print(
                    f"Point Label: {label} (No matching rows found in dataframe)"
                )
            continue

        stats = processed_stats[label]
        min_val = stats["min"]
        max_val = stats["max"]
        avg_val = stats["avg"]
        min_ts = stats["min_ts"]
        max_ts = stats["max_ts"]
        avg_ts = stats["avg_ts"]

        # Calculate percentage differences relative to 'fastest observed'
        if baseline_avg is not None:
            pct_diff = ((avg_val - baseline_avg) / baseline_avg) * 100.0
        else:
            pct_diff = np.nan

        if baseline_max is not None:
            max_diff = ((max_val - baseline_max) / baseline_max) * 100.0
        else:
            max_diff = np.nan

        if baseline_min is not None:
            min_diff = ((min_val - baseline_min) / baseline_min) * 100.0
        else:
            min_diff = np.nan

        # Conditionally print stats
        if verbose:
            print(f"Point Label: {label}")
            if min_val == max_val:
                print(
                    f"  DMA - Min: {min_val:.4f}, Max: {max_val:.4f}, Avg: {avg_val:.4f}"
                )
                print(f"  JSON Name: {min_ts} (no ties)")
            else:
                print(
                    f"  DMA - Min: {min_val:.4f} (JSON Name: {min_ts}), "
                    f"Max: {max_val:.4f} (JSON Name: {max_ts}), "
                    f"Avg: {avg_val:.4f}"
                )

            if not np.isnan(pct_diff):
                print(
                    f"  % Difference vs 'fastest observed': {pct_diff:+.2f}%\n"
                )
            else:
                print(
                    "  % Difference: 'fastest observed' baseline not found in labels\n"
                )

        # Append row for the returned DataFrame
        result_rows.append(
            {
                "sel_method": label,
                "exp_name": expName,
                "max": max_val,
                "min": min_val,
                "avg": avg_val,
                "max_ts": max_ts,
                "min_ts": min_ts,
                "avg_ts": avg_ts,
                "diff": pct_diff,
                "max_diff": max_diff,
                "min_diff": min_diff,
            }
        )

    # DataFrame columns specification
    cols = [
        "sel_method",
        "exp_name",
        "max",
        "min",
        "avg",
        "max_ts",
        "min_ts",
        "avg_ts",
        "diff",
        "max_diff",
        "min_diff",
    ]
    return pd.DataFrame(result_rows, columns=cols)


def pruneApproach5(timed, analyzed, full):
     expName = getExpName(timed)
     mnk_ties, mnk_priority = sm.mnk(full,timed)
     mnk_mDiv8_ties, mDiv8_priority = sm.mnk_mDiv8(full,timed)
     deeploy_ties, deeploy_priority = sm.deeploy(full,timed)
     myrtle_ties, table2, myrtle_priority = sm.myrtle(analyzed,timed)
     mnk_myrtle_ties, table3, mnk_myrtle_priority = sm.mnk_myrtle(analyzed,timed)
     fastestID = bestObserved(timed)["FakeNN JSON Name"]
     point_labels=["naive mnk", "naive mnk, m | 8","fastest observed","deeploy","myrtle","red + myrtle"]
     pointIDs=[getPointIDs(mnk_ties),getPointIDs(mnk_mDiv8_ties),[fastestID],getPointIDs(deeploy_ties),getPointIDs(myrtle_ties),getPointIDs(mnk_myrtle_ties)]
     sel_stats = maxMinAvg(expName,timed,point_ids=pointIDs,point_labels=point_labels)
     nickname=f"{expName}w-sel-method-comparison"
     sel_stats.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
     
     prunePoint = ssr_prune_frac(full,3)
     # prune full search space
     pruned = full[full["SSR Config Count"] < prunePoint] .copy()
     # identify analyzed but untimed points
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
     # prune untimed search space
     ut = ut[ut["SSR Config Count"] < prunePoint].copy()
     timed["timed"] = True
     ut["timed"] = False
     # remove m-dim NOT divisible by 8
     pruned["mMod8"] = pruned["m"] % 8
     aggressivelyPruned = pruned[pruned["mMod8"] == 0].copy()
     # remove n dim divisible by 32
     aggressivelyPruned["nMod32"]=aggressivelyPruned["n"] % 32
     aggressivelyPruned = aggressivelyPruned[aggressivelyPruned["nMod32"]!=0].copy()
     expStats = printMethodologyStats(full, pruned, timed, aggressivelyPruned)
     nickname=f"{expName}w-methodology-stats"
     expStats["MxNxK"]=[expName]
     expStats_df = pd.DataFrame(expStats)
     expStats_df.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
     
     hover_data = [
        "JSON Name",
        "timeout",
        "absoluteRank",
        "dma",
        "HW Loops",
        "FMADDsMULs",
        "FMADDsMULsPerCore",
        "Time (cycles)",#"SSR Loads per HW Loop",
        "HW Loops / SSR Loads per HW Loop",
        "remainderTiles",
        "Global Sim E2E_dma",
        "Total CL Tiles",
        "Total CC Tiles",
        #"Overlap Stall Time Per Core",
        "Avg C-M Ratio",#   "1/FMADDS",
        "L1 Usage",
        "Avg CC Tile Size",
        "mnkRem",
        "Avg L3 Loads",
        "tileB",
        "Avg n'_sz / k_size",
        "timedData",
     ]     
     special_figs=[]
     more_figs = []
     y_col = "Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "Global Sim E2E_dma" #"comp/memxfer"
     special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "diff",
            hover_data,
            "Reality Check. Make sure fastest point is ranked 1.",
            "timedData",
            ["circle","circle"]
     ))     
     #special_figs,more_figs=graphingExploration(more_figs,special_figs,timed,ut,analyzed,hover_data,prunePoint,asDF,expStats,fastestID,mnkMaxID)
     return special_figs,more_figs,f"<span>{table2}</span><span></span>"

def graphingExploration(more_figs,special_figs,timed,ut,analyzed,hover_data,prunePoint,asDF,expStats,fastestID,mnkMaxID):
     minimal_hover = [
        "JSON Name",       
        "SSR Configs",
        "L1 Usage", 
        "Avg C-M Ratio",
        "mnk"             
     ]
     specialHover = [ "JSON Name",
        "diff",
        "FMADDsMULsPerCore",
        "Avg C-M Ratio",
        "Time (cycles)",#"SSR Loads per HW Loop",
        "Total CL Tiles",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
        "mnk",
        "tileC"]
     
    
      # illustrate_ssr_pruning(pruned, full, special_figs,minimal_hover)
     y_col = "SSR Configs"
     x_col = "L1 Usage"
     fig = scatterWithFlatColor(
            ut,
            x_col,
            y_col,
            "pink",
            minimal_hover,
            f"Pruned Search Space to SSR configs <= {prunePoint} (black points are timed)",
            "symbolMarker",
        )
     special_figs.append(fig)
     addScatterFlatColorMarker(
        special_figs[-1],
        timed,
        x_col,
        y_col,
        "black",
        "circle",
        minimal_hover,
        "timed"
    )

     y_col = "timed"
     x_col = "SSR Configs"
     fig = scatterWithFlatColor(
            ut,
            x_col,
            y_col,
            "pink",
            minimal_hover,
            f"Pruned Search Space to SSR configs <= {prunePoint} (black points are timed)",
            "symbolMarker",
        )
     special_figs.append(fig)
     addScatterFlatColorMarker(
        special_figs[-1],
        timed,
        x_col,
        y_col,
        "black",
        "circle",
        minimal_hover,
        "timed"
    )

     y_col = "L1 Usage"
     x_col = "L1 Usage"
     fig = scatterWithFlatColor(
            timed,
            x_col,
            y_col,
            "black",
            minimal_hover,
            f"Pruned Search Space to SSR configs <= {prunePoint} (black points are timed)",
            "symbolMarker",
        )
     special_figs.append(fig)
  
     y_col = "L1 Usage"
     x_col = "L1 Usage"
     fig = scatterWithFlatColor(
            ut,
            x_col,
            y_col,
            "pink",
            minimal_hover,
            f"Pruned Search Space to SSR configs <= {prunePoint} (pink are untimed)",
            "symbolMarker",
        )
     special_figs.append(fig)

     y_col = "FMADDsMULsPerCore"
     x_col = "mnk"
     fig = scatterWithFlatColor(
            analyzed,
            x_col,
            y_col,
            "black",
            minimal_hover,
            "how different are mnk vs FmaddsPerCore?",
            "symbolMarker",
        )
     special_figs.append(fig)

     y_col = "FMADDsMULsPerCore"
     x_col = "FMADDsMULsPerCore"
     fig = scatterWithFlatColor(
            timed,
            x_col,
            y_col,
            "black",
            minimal_hover,
            f"Pruned Search Space to SSR configs <= {prunePoint} (black points are timed)",
            "symbolMarker",
        )
     special_figs.append(fig)
     fig = scatterWithFlatColor(
            ut,
            x_col,
            y_col,
            "pink",
            minimal_hover,
            f"Pruned Search Space to SSR configs <= {prunePoint} (pink are untimed)",
            "symbolMarker",
        )
     special_figs.append(fig)
   
     y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "mnk"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "m'_sz*n_sz / k_sz",
            specialHover,
            "is maximizing by mnk (naive) better?",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     special_figs[-1].update_traces(showlegend=False)

     y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
        timed,
            x_col,
            y_col,
            "FMADDsMULsPerCore",
            specialHover,
            "what about sorting by a so-called optimal compute-memory ratio?",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     addScatterFlatColorMarker(
                  special_figs[-1],
                  ut,
                  x_col,
                  y_col,
                  "gray",
                  "x",
                  minimal_hover,
                  "untimed"
              )
     special_figs[-1].update_traces(showlegend=False)

     fewer = timed[timed["Avg C-M Ratio"]<20].copy()
     focused = fewer[fewer["Avg C-M Ratio"]>13].copy()
     fewer_ut = ut[ut["Avg C-M Ratio"]<20].copy()
     focused_ut = fewer_ut[fewer_ut["Avg C-M Ratio"]>13].copy()
     m=timed["M"][0]
     n=timed["N"][0]
     k=timed["K"][0]
    #  nickname=f"{m}x{n}x{k}-CMRatioBetween13and20-GT-20k-mnk"
    #  to_time = focused_ut[focused_ut["nMod32"]!=0].copy()
    #  to_time = to_time[to_time["mnk"]> 20000].copy()
    #  to_time.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
    #  nickname=f"{m}x{n}x{k}-CMRatioBetween13and20-GT-20k-mnk-div8"
    #  to_time = to_time[to_time["mRem"]==0].copy()
    #  to_time.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
    
    #  y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
    #  x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
    #  special_figs.append(scatterWithColorSymbol(
    #     focused,
    #         x_col,
    #         y_col,
    #         "FMADDsMULsPerCore",
    #         specialHover,
    #         "Average Compute-Memory Overlap vs Percentage Diff from Best (Focused)",
    #         "mDiv8",
    #         ["circle","triangle-up"]
    #  ))
    #  addScatterFlatColorMarker(
    #          special_figs[-1],
    #          focused_ut,
    #          x_col,
    #          y_col,
    #          "gray",
    #          "x",
    #          minimal_hover,
    #          "untimed"
    #      )
    #  special_figs[-1].update_traces(showlegend=False)

    #  fewer = timed[timed["Avg C-M Ratio"]<20].copy()
    #  focused = fewer[fewer["Avg C-M Ratio"]>13].copy()
    #  y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
    #  x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
    #  special_figs.append(scatterWithColorSymbol(
    #     focused,
    #         x_col,
    #         y_col,
    #         "mnk",
    #         specialHover,
    #         "Average Compute-Memory Overlap vs Percentage Diff from Best (Focused)",
    #         "mDiv8",
    #         ["circle","triangle-up"]
    #  ))
     
    #  addScatterFlatColorMarker(
    #               special_figs[-1],
    #               focused_ut,
    #               x_col,
    #               y_col,
    #               "gray",
    #               "x",
    #               minimal_hover,
    #               "untimed"
    #           )
     special_figs[-1].update_traces(showlegend=False)

     mnk=pd.concat([timed,ut])
     mnk=mnk[mnk["mRem"]==0].copy()
     mnk=mnk.sort_values(by="mnk",ascending=False)
     #print(mnk[["FakeNN JSON Name","mnk","Avg C-M Ratio","timed","diff",]].head(5))
     
     # step 5: identify nice m remainders
     nice_timed,nice_ut = catByMDimBoundaryTile(timed, ut, niceOnly=True)
     # # illustrate_prune_nefarious_boundary_tiles(special_figs, hover_data, timed,ut)
     combined=pd.concat([nice_timed,nice_ut])

     # # remove m remainders < 8
     # combined = pd.concat([timed, ut])
     # combined = combined[combined["mRem"]>=8].copy()
     agressivelyPruned = nice_ut[nice_ut["mDiv8"] != 0].copy()
    #  agressivelyPruned = agressivelyPruned[agressivelyPruned["FMADDsMULsPerCore"] < 12500.0].copy()
    #  agressivelyPruned = agressivelyPruned[agressivelyPruned["FMADDsMULsPerCore"] > 10000.0].copy()
     agressivelyPruned = agressivelyPruned[agressivelyPruned["FMADDsMULsPerCore"] < 9000.0].copy()
     agressivelyPruned = agressivelyPruned[agressivelyPruned["FMADDsMULsPerCore"] > 6000.0].copy()
     agressivelyPruned = agressivelyPruned.sort_values(by="FMADDsMULsPerCore",ascending=False)
     m=timed["M"][0]
     n=timed["N"][0]
     k=timed["K"][0]
    #  nickname=f"{m}x{n}x{k}-between10and12k"
     #nickname=f"{m}x{n}x{k}-between6and9k"
     #filename= f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}"
     #agressivelyPruned.to_csv(filename,index=False)
    

     # remove n dim divisible by 32
     combined = combined[combined["nMod32"]!=0].copy()
    
     redPurple=combined[combined["mRem"]==0].copy()

     combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
     table = ""
    # table2, asDF, filteredDF, three_shelves =printFinalRankingTwoShelves("(prioritizing mRem = 0, then smaller m*n / k ratio)",combined,"m'_sz*n_sz / k_sz")
     chosenID=asDF[["FakeNN JSON Name"]].loc[0]["FakeNN JSON Name"]
     
     table3, redPurpleAsDF, filteredDF_redpurple, three_shelves_redpurple =printFinalRankingTwoShelves("(prioritizing mRem = 0, then smaller m*n / k ratio)",redPurple,"m'_sz*n_sz / k_sz","mnk")
     redPurpleID=redPurpleAsDF[["FakeNN JSON Name"]].loc[0]["FakeNN JSON Name"]       
     # create little DF with each point to compare
     nickname=f"{m}x{n}x{k}-baseline-best-myrtle"
     if f"{m}x{n}x{k}" == "192x384x384": #jugaad af
              mnkMaxID="192x384x384w46-56-45"
     if f"{m}x{n}x{k}" == "512x512x512": #jugaad af
                  mnkDiv8ID="512x512x512w48-56-43"
     point_ids=[mnkMaxID,mnkDiv8ID,fastestID,chosenID,redPurpleID]
     point_labels=["naive mnk", "naive mnk, m | 8","fastest observed","myrtle","red + myrtle"]
     df = extractPointsForComparison(timed,point_ids, point_labels)
    # print(point_ids)
     df.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
    # print(df[["FakeNN JSON Name","point_label","diff","Avg C-M Ratio","dma"]])
     nickname=f"{m}x{n}x{k}-methodology-stats"
     expStats["MxNxK"]=[f"{m}x{n}x{k}"]
     expStats_df = pd.DataFrame(expStats)
     expStats_df.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
    
    #  help = genShelfGraphPDF(jugaadTitle(timed),three_shelves,hover_data,color="mRem")
     


     #     nice_timed=nice_timed.sort_values("FMADDsMULsPerCore",ascending=False)
     #      y_col = "Avg n'_sz / k_size"
     #      x_col = "Global Sim E2E_dma"
     #      more_figs.append(scatterWithColorSymbol(
     #          nice_timed,
     #             x_col,
     #             y_col,
     #             "diff<0.5",
     #             hover_data,
     #             "Is pruning by n/k = 1 a good idea? circle is n/k < 1. orange is diff < 0.5 from best/",
     #             "n/k<1",
     #             ["triangle-up","circle"]
     #      ))
     #asDF=asDF.sort_values("mDiv8")
     timed = asDF[asDF["timedData"]==True].copy()
     ut = asDF[asDF["timedData"]==False].copy()
     timed=timed.sort_values("mDiv8")
     ut=ut.sort_values("mDiv8")
     
     y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "FMADDsMULsPerCore"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "m'_sz*n_sz / k_sz",
            specialHover,
            "maximize by FMADDsMULsPerCore, minimize by avg mn/k after pruning by ssr configs, m-rem, and preferring mMod8=0 ",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     special_figs[-1].update_traces(showlegend=False)
     
    #  y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
    #  x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
    #  special_figs.append(scatterWithColorSymbol(
    #      timed,
    #         x_col,
    #         y_col,
    #         "FMADDsMULsPerCore",
    #         specialHover,
    #         "after pruning by ssr configs, m-rem, and preferring mMod8=0 ",
    #         "mDiv8",
    #         ["circle","triangle-up"]
    #  ))
    #  special_figs[-1].update_traces(showlegend=False)
    #  y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
    #  x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
    #  special_figs.append(scatterWithColorSymbol(
    #           timed,
    #              x_col,
    #              y_col,
    #              "mnk",
    #              specialHover,
    #              "after pruning by ssr configs, m-rem, and preferring mMod8=0 ",
    #              "mDiv8",
    #              ["circle","triangle-up"]
    #       ))
    #  special_figs[-1].update_traces(showlegend=False)     

     x_col = "FMADDsMULsPerCore"
     y_col="Global Sim E2E_dma"
     forLatex = timed[timed["FMADDsMULsPerCore"]>9000.0].copy()
     if nickname=="128x128x128-between6and9k":
        forLatex = timed[timed["FMADDsMULsPerCore"]>5000.0].copy()
     forLatex_ut = ut[ut["FMADDsMULsPerCore"]>9000.0].copy()
     latexyGraph =genLatexyGraphPDF(jugaadTitle(forLatex),forLatex,x_col,y_col,"m'_sz*n_sz / k_sz","mDiv8",["triangle-up","circle"],specialHover,x_label=None,y_label="Time (cycles)",color_label="m_{size} * n_{size} / k_{size}",df_ut=None)
     special_figs.append(latexyGraph)
     #latexyGraph_include_ut =genLatexyGraphPDF(jugaadTitle(forLatex),forLatex,x_col,y_col,"m'_sz*n_sz / k_sz","mDiv8",["circle","triangle-up"],specialHover,x_label=None,y_label="Time (cycles)",color_label="m_{size} · n_{size} / k_{size}",df_ut=forLatex_ut) 
     #special_figs.append(latexyGraph_include_ut)
     # addScatterFlatColorMarker(
     #         special_figs[-1],
     #         ut,
     #         x_col,
     #         y_col,
     #         "gray",
     #         "x",
     #         hover_data,
     #         "untimed"
     #     )
     # x, y, z
     # "FMADDsMULsPerCore","diff","m'_sz*n_sz / k_sz"
     return special_figs,more_figs