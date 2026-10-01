import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import math
import re
import pathlib
import sys
from scripts.graphUtils import scatterWithColorSymbol, scatterWithFlatColor, scatterWithColor, addScatterFlatColorMarker, stack_dfs_to_html_w_toggle,genShelfGraphPDF,jugaadTitle
import pandas as pd
from typing import List
import numpy as np
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
    my_columns = ["JSON Name", "timed", "Avg m'_sz*n_sz / k_sz", "mnkRem","mRem","nMod32",colorCol,"tileB", "Time (cycles)", "diff"]
    
    # Get unique FMADD values, sort them from smallest to largest
    sorted_fmadd_keys = sorted(df['FMADDsMULsPerCore'].unique(), reverse=True)
    
    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df['FMADDsMULsPerCore'] == fmadd_value]
        
        # Sort by 'mRem' (Smallest to Largest) and then colorCol (SMALLEST TO LARGEST)
        processed_slice = fmadd_group.sort_values(
            by=['mRem', colorCol], 
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

def printMethodologyStats(full, pruned, timed):
    print(f"full ss has size {len(full)}")
    print(f"pruned has size {len(pruned)}")
    print(f"timed has size {len(timed)}")
    print(f"what percentage of SPM used in timed points vs pruned points? {len(timed)/len(pruned)*100.0} %")

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

def pruneApproach5(timed, analyzed, full):
     prunePoint = ssr_prune_frac(full,3)
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
     # make sure untimed points are a subset of the pruned search space
     ut = ut[ut["SSR Config Count"] < prunePoint].copy()
     pruned = full[full["SSR Config Count"] < prunePoint]
     timed["timed"] = True
     ut["timed"] = False
     printMethodologyStats(full, pruned, timed)
     hover_data = [
        "JSON Name",
        "timeout",
        "absoluteRank",
        "dma",
        "HW Loops",
        "SSR Configs",
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
        "comp/memxfer",
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
     
     
     
     # step 5: identify nice m remainders
     nice_timed,nice_ut = catByMDimBoundaryTile(timed, ut, niceOnly=True)
     # illustrate_prune_nefarious_boundary_tiles(special_figs, hover_data, timed,ut)
     combined=pd.concat([nice_timed,nice_ut])

     # remove n dim divisible by 32
     combined = combined[combined["nMod32"]!=0].copy()
    

     combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
     table = ""
     table2, asDF, filteredDF, three_shelves =printFinalRankingTwoShelves("(prioritizing mRem = 0, then smaller m*n / k ratio)",combined,"m'_sz*n_sz / k_sz")
    
     help = genShelfGraphPDF(jugaadTitle(timed),three_shelves,hover_data,color="mRem")
     specialHover = [ "JSON Name",
        "diff",
        "SSR Configs",
        "FMADDsMULsPerCore",
        "Time (cycles)",#"SSR Loads per HW Loop",
        "Total CL Tiles",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
        "tileC",
        "comp/memxfer",]

     y_col = "Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "FMADDsMULsPerCore"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         asDF.head(50),
            x_col,
            y_col,
            "tileC",
            specialHover,
            "Best 50, maximize by FMADDsMULsPerCore, minimize by avg mn/k after pruning by ssr configs, m and m-rem ",
            "timedData",
            ["circle","circle"]
     ))
     special_figs[-1].update_traces(showlegend=False) 
     return special_figs,more_figs,f"<span>{table2}</span><span>{table}</span>"
