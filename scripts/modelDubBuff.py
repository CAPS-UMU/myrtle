from graphUtils import scatterWithColorSymbol, scatterWithFlatColorSymbol, scatterWithFlatColor, scatterWithColor, addScatterFlatColorMarker, stack_dfs_to_html, genResultGraphPDF
import pandas as pd
from typing import List
import numpy as np
pd.set_option('display.max_rows', None)

def generate_latex_table(
    df: pd.DataFrame, 
    columns_to_include: List[str], 
    output_headers: List[str],
    index: bool = False
) -> str:
    # 1. Extract the first 5 rows
    top_5 = df.head(6).copy()
    
    # 2. Find the row where diff == 0.0
    diff_zero_row = df[df['diff'] == 0.0].head(1).copy()
    
    # Combine the selections, ensuring no duplicates if the 0.0 row falls in the top 5
    if not diff_zero_row.empty and diff_zero_row.index[0] not in top_5.index:
        selected_df = pd.concat([top_5, diff_zero_row])
    else:
        selected_df = top_5
    
    # 3. Filter and order data based strictly on user input data frame columns
    selected_df = selected_df[[c for c in columns_to_include if c in selected_df.columns]]
    
    # 4. Build the LaTeX string pieces
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    
    # Dynamic column alignment string (e.g., |l|r|r|...) based on df column selection
    alignments = []
    
    # Prepend alignment for the index column if requested
    if index:
        alignments.append('r')
        
    for col in selected_df.columns:
        if col == 'FakeNN JSON Name':
            alignments.append('l')
        else:
            alignments.append('r')
    col_alignment_str = f"|{'|'.join(alignments)}|"
    
    latex_lines.append(f"\\begin{{tabular}}{{{col_alignment_str}}}")
    latex_lines.append(r"\hline")
    
    # Header Row using the custom output headers list
    display_headers = output_headers if len(output_headers) == len(selected_df.columns) else selected_df.columns
    headers = display_headers#[f"\\textbf{{{str(h).replace('_', r'\\_')}}}" for h in display_headers]
    
    # Prepend "Index" header if index=True
    if index:
        headers.insert(0, r"\textbf{Index}")
        
    header_row = " & ".join(headers) + r" \\"
    latex_lines.append(header_row)
    latex_lines.append(r"\hline")
    
    # 5. Populate rows dynamically
    for idx, (original_idx, row) in enumerate(selected_df.iterrows()):
        row_elements = []
        
        # Add the original dataframe index value if requested
        if index:
            row_elements.append(str(original_idx))
            
        for col in selected_df.columns:
            val = row[col]
            
            # Safely check for Python float or NumPy float variants
            if isinstance(val, (float, np.floating)):
                val_str = f"{val:.2f}"
            # Sanitize string columns from throwing LaTeX syntax crashes
            elif isinstance(val, str):
                val_str = val.replace("_", r"\_")
            else:
                val_str = str(val)
                
            row_elements.append(val_str)
            
        row_str = " & ".join(row_elements) + r" \\"
        
        # Apply yellow background color ONLY to the second row (the first data row, idx == 0)
        if idx == 0:
            row_str = r"\rowcolor{yellow} " + row_str
            
        latex_lines.append(row_str)
        
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{Neural Network Performance Metrics}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines)

def jugaadTitle(df):
    M=int(df["M"][0])
    N=int(df["N"][0])
    K=int(df["K"][0])
    dims=f"{M}x{N}x{K}"
    t="Transformer"
    if (M == 128) and (N == 128) and (K == 128):
        t="BertTiny" 
    if (M == 128) and (N == 768) and (K == 768):
        t="Roberta" 
    if (M == 192) and (N == 384) and (K == 384):
        t="BGESmall" 
    if (M == 384) and (N == 384) and (K == 384):
        t="MiniLM" 
    if (M == 256) and (N == 256) and (K == 256):
        t="BertMini" 
    if (M == 512) and (N == 512) and (K == 512):
        t="Bert" 
    return f"{t} Matmul {dims}"
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
def printFinalRanking(title,df,colorCol):
    print("\tFinal ranking:")
    df=df.sort_values("Time (cycles)",ascending=True)
    df = df.reset_index(drop=True)
    best=df["Time (cycles)"][0]
    title=f"best observed: {best} cycles {title}"
    print(title)
    df=df.sort_values("FMADDsMULsPerCore",ascending=False)
    df["diff"] = df["Time (cycles)"].apply(lambda x: (x - best)/best * 100)
 #   print(df[["JSON Name","FMADDsMULsPerCore","timeout","Time (cycles)","diff"]][0:9])
    print("--------------------")
    #latexList=[subFigPro(title)]
    pointsPrinted = 0
    my_columns = ["JSON Name","mRem","timed","Avg n'_sz / k_size",colorCol,"Time (cycles)","diff",]
    my_dfs = []
    my_titles = []
    containsFast128Tile = False #"64-24-64"
    for fmadds, group_df in df.groupby("FMADDsMULsPerCore",sort=False):
        if pointsPrinted < 5 or not containsFast128Tile:
               containsFast128Tile = (group_df ['JSON Name'] == '64-24-64').any()
               subtitle = f"FMADDS: {fmadds} w/ len {len(group_df)}"
               print(subtitle)
               my_titles.append(subtitle)
               pointsPrinted = pointsPrinted + len(group_df)
               sorted = group_df.sort_values(colorCol,ascending=False)
               my_dfs.append(sorted)
               print(sorted[my_columns])
    # Generate the an HTML version of ranking table
    table = stack_dfs_to_html(my_dfs, my_titles, my_columns,title)
    return table


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
    #my_columns = ["JSON Name", "mnkRem", "timed", "Avg n'_sz / k_size","m'_sz*n_sz / k_sz",colorCol,"Avg B'", "Time (cycles)", "diff"]
    my_columns = ["JSON Name", "mnkRem", "timed", "Avg n'_sz / k_size","Avg m'_sz / k_size",colorCol,"Avg B'", "Time (cycles)", "diff"]
    
    # Get unique FMADD values, sort them from smallest to largest
    sorted_fmadd_keys = sorted(df['FMADDsMULsPerCore'].unique(), reverse=True)
    
    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df['FMADDsMULsPerCore'] == fmadd_value]
        
        # Sort by 'mRem' (Smallest to Largest) and then colorCol (Largest to Smallest)
        processed_slice = fmadd_group.sort_values(
            by=['mRem', colorCol], 
            ascending=[True, False]
        )
        # Append to our parallel output lists
        fmadd_values.append(fmadd_value)
        processed_dfs.append(processed_slice)
        myTitle = f"FMADDS: {fmadd_value} w/ len {len(processed_slice)}"
        my_titles.append(myTitle)
        
    # 1. Concatenate all processed slices into a single DataFrame by stacking rows
    # (ignoring index ensures a clean, continuous index for the combined df)
    concatenated_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
    
    # 2. Filter for rows where 'timed' value is False
    filtered_df = concatenated_df[concatenated_df['timed'] == False]
    
    # Generate the usual HTML table
    # don't print all the shelves, in fact print a max of 6
    finalShelf = min(10,len(processed_dfs)-1)
    table = stack_dfs_to_html(processed_dfs[0:finalShelf], my_titles[0:finalShelf], my_columns, title)
    #table = stack_dfs_to_html(processed_dfs, my_titles, my_columns, title)
    
    # 3. Return all three values
    return table, concatenated_df, filtered_df

def printMethodologyStats(full, pruned, timed):
    print(f"full ss has size {len(full)}")
    print(f"pruned has size {len(pruned)}")
    print(f"timed has size {len(timed)}")
    print(f"what percentage of SPM used in timed points vs pruned points? {len(timed)/len(pruned)*100.0} %")

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

def pruneApproach3(timed, analyzed, full):
     prunePoint = ssr_prune_frac(full,3)
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
     # make sure untimed points are a subset of the pruned search space
     ut = ut[ut["SSR Config Count"] < prunePoint].copy()
     pruned = full[full["SSR Config Count"] < prunePoint]
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
        "1/FMADDS",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
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


     #reality check
     x_col = "Global Sim E2E_dma"#"Avg n'_sz / k_size"
     y_col = "Global Sim E2E_dma"#"Global Sim E2E_dma"
     more_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "Reality Check. Make sure fastest point is ranked 1.",
            "timedData",
            ["circle","circle"]
     ))

    # illustrate_ssr_pruning(pruned, full, more_figs,minimal_hover)
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
     more_figs.append(fig)
     addScatterFlatColorMarker(
        more_figs[-1],
        timed,
        x_col,
        y_col,
        "black",
        "circle",
        minimal_hover,
        "timed"
    )
     
     
     
     # step 5: identify nice m remainders
 

     nice_timed=timed[timed["niceMRem"]].copy()
     nice_ut=ut[ut["niceMRem"]].copy()
     nice_timed["timed"]=True
     nice_ut["timed"] = False
     mean_timed=timed[timed["niceMRem"]==False]
     mean_ut=ut[ut["niceMRem"]==False]
     x_col = "mRem"
     y_col = "Global Sim E2E_dma"
     more_figs.append(
        scatterWithFlatColor(
            nice_timed,
            x_col,
            y_col,
            "black",
            hover_data,
            "1.2) Pruned out worst case CL boundary tiles (marked with red x)",
            "symbolMarker",
        )
    )
     addScatterFlatColorMarker(
        more_figs[-1],
        nice_ut,
        x_col,
        y_col,
        "gray",
        "circle",
        hover_data,
        "untimed w/ nice m remainder"
    )
     addScatterFlatColorMarker(
        more_figs[-1],
        mean_ut,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "untimed w/ worst case m remainder"
    )
     addScatterFlatColorMarker(
        more_figs[-1],
        mean_timed,
        x_col,
        y_col,
        "red",
        "x",
        hover_data,
        "timed w/ worst case m remainder"
    )
     
     #result graph
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
     #resultGraph = genResultGraphPDF(jugaadTitle(timed),nice_timed_reduced_lt1,nice_ut_reduced_lt1,nice_timed_reduced_gte1,hover_data,"mRem")
  
     combined=pd.concat([nice_timed,nice_ut])
   #  print(combined[["JSON Name","Avg n'_sz / k_size","Avg m'_sz / k_size"]])
     #print("after pruning:")
     combined=combined[combined["Avg n'_sz / k_size"]<=1.0]

     combined = combined.sort_values(by="timed",ascending=False)
     more_figs.append(scatterWithColorSymbol(
         combined,
            x_col,
            y_col,
            "tileB",
            hover_data,
            "Pruned to n/k <= 1: Maximize by Fmadds, tie break with smaller mRem first, then larger B tile",
            "timeout",
            ["circle","cross"]
     ))
     more_figs[-1].update_traces(showlegend=False)

     combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
     table = ""
     table2, asDF, filteredDF =printFinalRankingTwoShelves("(prioritizing mRem = 0, then larger nxk = tileB)",combined,"tileB")
    
     specialHover = [ "JSON Name",
        "diff",
        "SSR Configs",
        "FMADDsMULsPerCore",
        "Time (cycles)",#"SSR Loads per HW Loop",
        "Total CL Tiles",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
        "tileB",
        "comp/memxfer",]

     y_col = "Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         asDF.head(50),
            x_col,
            y_col,
            "mRem",
            specialHover,
            "Best 50, sorting by avg mn/k after pruning by ssr configs, m-rem, and n/k<=1 ",
            "timedData",
            ["circle","circle"]
     ))
     special_figs[-1].update_traces(showlegend=False)
     
      
     return special_figs,more_figs,f"<span>{table2}</span><span>{table}</span>"