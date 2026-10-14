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



def printFinalRankingTwoShelves(title, df, colorCol):
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
    my_columns = ["JSON Name", "timed", "Avg m'_sz*n_sz / k_sz", "mnkRem","mMod8","nMod32",colorCol,"tileB", "Time (cycles)", "diff"]
    
    # Get unique FMADD values, sort them from smallest to largest
    sorted_fmadd_keys = sorted(df['FMADDsMULsPerCore'].unique(), reverse=True)
    
    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df['FMADDsMULsPerCore'] == fmadd_value]
        
        # Sort by 'mRem' (Smallest to Largest) and then colorCol (SMALLEST TO LARGEST)
        processed_slice = fmadd_group.sort_values(
            by=['mMod8', colorCol], 
            ascending=[True, True]
        )
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

def printMethodologyStats(full, pruned, timed, suggested):
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

def pruneApproach5(timed, analyzed, full):
     mnk,mnkDiv8,full = maxMNK(full)
     # print(full[["FakeNN JSON Name","mnk"]].head(10))
     mnkMaxID = mnk["FakeNN JSON Name"]#.values[0]
     # print(f"max mnk is ts {mnkMaxID}\n{mnk}")
     mnkDiv8ID = mnkDiv8["FakeNN JSON Name"]#.values[0]
     # print(f"max mnk divisible by 8 in m-dim is ts {mnkDiv8ID}\n{mnkDiv8}")
     #print(full[["FakeNN JSON Name","mnk"]])
     prunePoint = ssr_prune_frac(full,3)
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
     # make sure untimed points are a subset of the pruned search space
     ut = ut[ut["SSR Config Count"] < prunePoint].copy()
     pruned = full[full["SSR Config Count"] < prunePoint]     
     if(mnkMaxID in pruned["FakeNN JSON Name"].values[0]):
        print("pruned SS contains the max mnk")
     else:
        raise Exception("we have a PROBLEM! max mnk not included in pruned SS!!!")

     timed["timed"] = True
     ut["timed"] = False
     fastest = bestObserved(timed)
     fastestID = fastest["FakeNN JSON Name"]
     print(f"fastest ts is {fastestID}\n{fastest}")

     pruned = pruned.sort_values(by="SSR Configs",ascending=True)
     pruned = pruned.reset_index(drop=True)
     timed = timed.sort_values(by="SSR Configs",ascending=True)
     timed = timed.reset_index(drop=True)
     subseq_df, pruned_rest, timed_rest = split_longest_initial_subsequence(timed, pruned)
     #print(subseq_df[["FakeNN JSON Name","SSR Configs"]])
     print("timed points breakdown:")
     print(f"min ssr configs timed: {subseq_df.shape[0]}")
     print(f"other points timed: {timed_rest.shape[0]}")
     print(f"remaining points to time: {pruned_rest.shape[0]}")
     aggressivelyPruned = pruned[pruned["mDiv8"] != 0].copy()
     # remove n dim divisible by 32
     aggressivelyPruned["nMod32"]=aggressivelyPruned["n"] % 32
     aggressivelyPruned = aggressivelyPruned[aggressivelyPruned["nMod32"]!=0].copy()

     expStats = printMethodologyStats(full, pruned, timed, aggressivelyPruned)

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
     #   "1/FMADDS",
        "L1 Usage",
        "Avg CC Tile Size",
        "mnkRem",
        "Avg L3 Loads",
        "tileB",
        "Avg n'_sz / k_size",
        "timedData",
     ]
     minimal_hover = [
        "JSON Name",       
        "SSR Configs",
        "L1 Usage",       
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

     specialHover = [ "JSON Name",
        "diff",
        "FMADDsMULsPerCore",
        "Time (cycles)",#"SSR Loads per HW Loop",
        "Total CL Tiles",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
        "mnk",
        "tileC"]
   
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

     mnk=combined=pd.concat([timed,ut])
     mnk=mnk[mnk["mRem"]==0].copy()
     mnk=mnk.sort_values(by="mnk",ascending=False)
     print(mnk[["FakeNN JSON Name","mnk","timed","diff"]].head(20))
     
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
    

     combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
     table = ""
     table2, asDF, filteredDF, three_shelves =printFinalRankingTwoShelves("(prioritizing mRem = 0, then smaller m*n / k ratio)",combined,"m'_sz*n_sz / k_sz")
     chosenID=asDF[["FakeNN JSON Name"]].loc[0]["FakeNN JSON Name"]
     # create little DF with each point to compare
     nickname=f"{m}x{n}x{k}-baseline-best-myrtle"
     point_ids=[mnkMaxID,mnkDiv8ID,fastestID,chosenID]
     point_labels=["naive mnk", "naive mnk, m | 8","fastest observed","myrtle"]
     df = extractPointsForComparison(timed,point_ids, point_labels)
     df.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}.csv",index=False)
     print(df[["FakeNN JSON Name","point_label","diff","dma"]])
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
   #  special_figs.append(plot_3d_scatter(timed,("FMADDsMULsPerCore","diff","m'_sz*n_sz / k_sz"),specialHover))
     return special_figs,more_figs,f"<span>{table2}</span><span>{table}</span>"