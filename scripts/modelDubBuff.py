from graphUtils import scatterWithColorSymbol, scatterWithFlatColorSymbol, scatterWithFlatColor, scatterWithColor, addScatterFlatColorMarker, stack_dfs_to_html, genResultGraphPDF
import pandas as pd
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

# def printFinalRankingTwoShelves(title,df,colorCol):
#     print("\tFinal ranking:")
#     df=df.sort_values("Time (cycles)",ascending=True)
#     df = df.reset_index(drop=True)
#     best=df["Time (cycles)"][0]
#     title=f"best observed: {best} cycles {title}"
#     print(title)
#     df=df.sort_values("FMADDsMULsPerCore",ascending=False)
#     df["diff"] = df["Time (cycles)"].apply(lambda x: (x - best)/best * 100)
#  #   print(df[["JSON Name","FMADDsMULsPerCore","timeout","Time (cycles)","diff"]][0:9])
#     print("--------------------")
#     #latexList=[subFigPro(title)]
#     pointsPrinted = 0
#     my_columns = ["JSON Name","mRem","timed","Avg n'_sz / k_size",colorCol,"Time (cycles)","diff",]
#     my_dfs = []
#     my_titles = []
#     containsFast128Tile = False #"64-24-64"
#     for fmadds, group_df in df.groupby("FMADDsMULsPerCore",sort=False):
#             subtitle = f"FMADDS: {fmadds} w/ len {len(group_df)}"
#             pointsPrinted = pointsPrinted + len(group_df)
#             containsFast128Tile = (group_df ['JSON Name'] == '64-24-64').any()
#             if pointsPrinted < 5 or not containsFast128Tile:
#                 my_titles.append(subtitle)
#                 print(subtitle)
#                 toConcat = []
#                 for mRem, shelf2 in group_df.groupby("mRem",sort=False):
                    
#                         mRemCat=shelf2.sort_values(colorCol,ascending=False)
#                         toConcat.append(mRemCat)
                        
#                 concatted = pd.concat(toConcat)     
#                 print(concatted) 
#                 my_dfs.append(concatted)
#     # Generate the an HTML version of ranking table
#     table = stack_dfs_to_html(my_dfs, my_titles, my_columns,title)
#     return table

def printFinalRankingTwoShelves(title,df,colorCol):
    """
    1. Sorts and slices unique 'FMADDsMULsPerCore' values from smallest to largest.
    2. Groups rows by 'mRem' (ascending) and 'tileB' (descending) within each slice.
    3. Returns two parallel lists: 
       - fmadd_values: The sorted 'FMADDsMULsPerCore' identifiers.
       - processed_dfs: The corresponding sorted DataFrames.
    """
    df=df.sort_values("Time (cycles)",ascending=True)
    df = df.reset_index(drop=True)
    best=df["Time (cycles)"][0]
    title=f"best observed: {best} cycles {title}"
    fmadd_values = []
    processed_dfs = []
    my_titles = []
    my_columns = ["JSON Name","mnkRem","timed","Avg n'_sz / k_size",colorCol,"Time (cycles)","diff",]
    # Get unique FMADD values, sort them from smallest to largest
    sorted_fmadd_keys = sorted(df['FMADDsMULsPerCore'].unique(),reverse=True)
    
    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df['FMADDsMULsPerCore'] == fmadd_value]
        
        # Sort by 'mRem' (Smallest to Largest) and then 'tileB' (Largest to Smallest)
        processed_slice = fmadd_group.sort_values(
            by=['mRem', colorCol], 
            ascending=[True, False]
        )
        # Append to our parallel output lists
        fmadd_values.append(fmadd_value)
        processed_dfs.append(processed_slice)
        myTitle=f"FMADDS: {fmadd_value} w/ len {len(processed_slice)}"
        my_titles.append(myTitle)
    table = stack_dfs_to_html(processed_dfs, my_titles, my_columns,title)
    return table

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
    # we assume untimed points are a subset of the pruned search space
     ut = analyzed[~analyzed["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
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
     ]
     minimal_hover = [
        "JSON Name",       
        "SSR Configs",
        "L1 Usage",       
     ]
     special_figs=[]
     more_figs = []
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
    #  y_col = "SSR Configs"
    #  x_col = "L1 Usage"
    #  fig = scatterWithFlatColor(
    #         pruned,
    #         x_col,
    #         y_col,
    #         "pink",
    #         minimal_hover,
    #         "1.2) Pruned Search Space (black points are timed); identify worst case CL boundary tiles",
    #         "symbolMarker",
    #     )
    #  more_figs.append(fig)
    #  addScatterFlatColorMarker(
    #     more_figs[-1],
    #     timed,
    #     x_col,
    #     y_col,
    #     "black",
    #     "circle",
    #     minimal_hover,
    #     "timed"
    # )
     
     
     
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
            "1.2) Pruned Search Space (black points are timed); identify worst case CL boundary tiles (marked with red x)",
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
     
       
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
    # print(f"before appending: len of more_figs is {len(more_figs)}")
     more_figs.append(scatterWithColorSymbol(
         nice_timed,
            x_col,
            y_col,
            "mRem",
            hover_data,
            "Final Cost Model selection: max. by Fmadds, tie break with smaller mRem",
            "timeout",
            ["circle","cross"]
     ))
     more_figs[-1].update_traces(showlegend=False)

     
     #result graph
     x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
     y_col = "Time (cycles)"#"Global Sim E2E_dma"
     #resultGraph = genResultGraphPDF(jugaadTitle(timed),nice_timed_reduced_lt1,nice_ut_reduced_lt1,nice_timed_reduced_gte1,hover_data,"mRem")
  
     combined=pd.concat([nice_timed,nice_ut])
     table=printFinalRanking("",combined,"tileB")
     table2=printFinalRankingTwoShelves("(prioritizing mRem = 0, then larger nxk = tileB)",combined,"tileB")
     
     
     #special_figs.append(resultGraph) 
      
     return special_figs,more_figs,f"<span>{table2}</span><span>{table}</span>"