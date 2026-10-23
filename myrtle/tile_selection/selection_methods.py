import pandas as pd
import pathlib
from scripts.graphUtils import stack_dfs_to_html_w_toggle

# return the SSR config value that is the largest of the bottom frac
def ssr_prune_frac(df,frac):
        unique_ssr_configs = list(
            set(df["SSR Config Count"].values.tolist())
        )  # remove duplicates
        unique_ssr_configs.sort()  # sort least to greateset
        prunePoint = unique_ssr_configs[int(len(unique_ssr_configs)/frac)]  # prune to smallest frac of ssr_configs   
        return prunePoint

def rankWithShelves(title, df, colorCol, shelfMetric="FMADDsMULsPerCore",htmlTable=True):
    """
    1. Sorts and slices unique shelfMetric values from smallest to largest.
    2. Groups rows by 'mRem' (ascending) and then 'colorCol' (descending) within each slice.
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
    my_titles = []  # Avg m'_sz / k_size
    # my_columns = [
    #     "JSON Name",
    #     "mnkRem",
    #     "timed",
    #     "Avg n'_sz / k_size",
    #     "m'_sz*n_sz / k_sz",
    #     colorCol,
    #     "Avg B'",
    #     "Time (cycles)",
    #     "diff",
    # ]
    # my_columns = [
    #     "JSON Name",
    #     "timed",
    #     "Avg m'_sz*n_sz / k_sz",
    #     "FMADDsMULsPerCore",
    #     "mRem",
    #     "mnk",
    #     colorCol,
    #     "tileB",
    #     "Time (cycles)",
    #     "diff",
    # ]
    my_columns = [
        "JSON Name",
        "timed",
        "mnk",
        "FMADDsMULsPerCore",
        "mnkRem",
        "mMod8",
        "nMod32",
        "m > k",
        colorCol,
        "Avg C-M Ratio",
        "Time (cycles)",
        "diff",
    ]

    # Get unique FMADD values, sort them from largest to smallest
    sorted_fmadd_keys = sorted(df[shelfMetric].unique(), reverse=True)

    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df[shelfMetric] == fmadd_value]

        # Sort by 'mRem' (Smallest to Largest) and then colorCol (SMALLEST TO LARGEST)
        processed_slice = fmadd_group.sort_values(
            by=["mMod8", colorCol], ascending=[True, True]
        )
        # Append to our parallel output lists
        fmadd_values.append(fmadd_value)
        processed_dfs.append(processed_slice)
        myTitle = f"{shelfMetric}: {fmadd_value} w/ len {len(processed_slice)}"
        my_titles.append(myTitle)

    # 1. Concatenate all processed slices into a single DataFrame by stacking rows
    # (ignoring index ensures a clean, continuous index for the combined df)
    concatenated_df = (
        pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
    )

    # 2. Generate the usual HTML table
    # don't print all the shelves, in fact print a max of 20
    finalShelf = min(20, len(processed_dfs) - 1)
    if htmlTable:
        table = stack_dfs_to_html_w_toggle(
            processed_dfs[0:finalShelf], my_titles[0:finalShelf], my_columns, title
        )
    else:
        table="empty table"
    
    # 3. Remove low priority mMod8!= 0 rows wherever possible
    high_priority_only = []
    fallback_to_m_dim_boundary_tiles = []
    for shelf in processed_dfs:   
        #print(shelf[["FakeNN JSON Name","mMod8",colorCol,shelfMetric]])     
        shelf_high_priority = shelf[shelf["mMod8"]==0.0].copy()        
        if shelf_high_priority.empty:
            #print("high priority shelf is empty")
            #print(shelf_high_priority[["FakeNN JSON Name",colorCol,shelfMetric]])
            shelf_low_priority = shelf[shelf["mMod8"]!=0.0].copy()
            shelf_low_priority = shelf_low_priority.sort_values(colorCol,ascending=True)
            fallback_to_m_dim_boundary_tiles.append(shelf_low_priority)
        else:
            shelf_high_priority = shelf_high_priority.sort_values(colorCol,ascending=True)
            high_priority_only.append(shelf_high_priority)
        if len(high_priority_only) >= 1:
            break
    if len(high_priority_only)==0:
        high_priority_only = fallback_to_m_dim_boundary_tiles
    high_priority_df = (
            pd.concat(high_priority_only, ignore_index=True) if high_priority_only else pd.DataFrame()
        )
    highest_shelf = high_priority_only[0]
    # print("top shelf:")
    # print(highest_shelf[["FakeNN JSON Name",colorCol,shelfMetric]])
    highest_shelf = highest_shelf.reset_index(drop=True)
    bestColorColVal = highest_shelf[colorCol].loc[0]
    ties = highest_shelf[highest_shelf[colorCol]==bestColorColVal]
    # if ties.shape[0]==1:
    #     print("ties: none")
    # else:
    #     print("ties:")
    #     print(ties[["FakeNN JSON Name",colorCol,shelfMetric]])
    # 4. Return the shelves concatenated, visual table representation and shelves w/ rank mMod8 != 0 removed
    return concatenated_df, table, high_priority_df, ties

def rankWithShelvesParameterized(title, df, shelfMetric="FMADDsMULsPerCore", sortShelfMetric0=("mMod8",True),sortShelfMetric1=("Avg m'_sz*n_sz / k_sz",True),htmlTable=True):
    """
    1. Sorts and slices unique 'shelfMetric' values from smallest to largest.
    2. Groups rows by 'sortShelfMetric0'[0] and then 'sortShelfMetric1'[0] within each slice.
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
    my_titles = []  # Avg m'_sz / k_size

    my_columns = [
        "JSON Name",
        "timed",
        "mnk",
        "FMADDsMULsPerCore",
        "mnkRem",
        sortShelfMetric0[0],
        "nMod32",
        "m'_sz*n_sz / k_sz",
        "mn/k",
        "Avg C-M Ratio",
        "Time (cycles)",
        "diff",
    ]

    # Get unique FMADD values, sort them from largest to smallest
    sorted_fmadd_keys = sorted(df[shelfMetric].unique(), reverse=False)

    # Process each FMADD slice in ascending order
    for fmadd_value in sorted_fmadd_keys:
        # Extract the slice for the current FMADD value
        fmadd_group = df[df[shelfMetric] == fmadd_value]

        # Sort by sortShelfMetric0 and then sortShelfMetric1
        processed_slice = fmadd_group.sort_values(
            by=[sortShelfMetric0[0], sortShelfMetric1[0]], ascending=[sortShelfMetric0[1], sortShelfMetric1[1]]
        )
        # Append to our parallel output lists
        fmadd_values.append(fmadd_value)
        processed_dfs.append(processed_slice)
        myTitle = f"{shelfMetric}: {fmadd_value} w/ len {len(processed_slice)}"
        my_titles.append(myTitle)

    # 1. Concatenate all processed slices into a single DataFrame by stacking rows
    # (ignoring index ensures a clean, continuous index for the combined df)
    concatenated_df = (
        pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
    )

    # 2. Generate the usual HTML table
    # don't print all the shelves, in fact print a max of 20
    finalShelf = min(20, len(processed_dfs) - 1)
    finalShelf = len(processed_dfs) - 1
    if htmlTable:
        table = stack_dfs_to_html_w_toggle(
            processed_dfs[0:finalShelf], my_titles[0:finalShelf], my_columns, title
        )
    else:
        table="empty table"
    
    # 3. Remove low priority sortShelfMetric0[0]!= 0 rows wherever possible
    high_priority_only = []
    fallback_to_m_dim_boundary_tiles = []
    for shelf in processed_dfs:       
        shelf_high_priority = shelf[shelf[sortShelfMetric0[0]]==0.0].copy()        
        if shelf_high_priority.empty:
            shelf_low_priority = shelf[shelf[sortShelfMetric0[0]]!=0.0].copy()
            shelf_low_priority = shelf_low_priority.sort_values(sortShelfMetric1[0],ascending=sortShelfMetric1[1])
            fallback_to_m_dim_boundary_tiles.append(shelf_low_priority)
        else:
            shelf_high_priority = shelf_high_priority.sort_values(sortShelfMetric1[0],ascending=sortShelfMetric1[1])
            high_priority_only.append(shelf_high_priority)
        if len(high_priority_only) >= 1:
            break
    if len(high_priority_only)==0:
        high_priority_only = fallback_to_m_dim_boundary_tiles
    high_priority_df = (
            pd.concat(high_priority_only, ignore_index=True) if high_priority_only else pd.DataFrame()
        )
    highest_shelf = high_priority_only[0]
    highest_shelf = highest_shelf.reset_index(drop=True)
    bestColorColVal = highest_shelf[sortShelfMetric1[0]].loc[0]
    ties = highest_shelf[highest_shelf[sortShelfMetric1[0]]==bestColorColVal]

    return concatenated_df, table, high_priority_df, ties

# naive selection
# maximize by mnk product
# we assume the full search space is analyzed
def mnk(full,timed):
    m=timed["M"][0]
    n=timed["N"][0]
    k=timed["K"][0]
    nickname=f"{m}x{n}x{k}w"
    full["mnk"] = full["m"]*full["n"]*full["k"]
    full = full.sort_values(by="mnk",ascending=False)
    full = full.reset_index(drop=True)
    best = full[["mnk"]].loc[0]["mnk"]
    ties = full[full["mnk"]==best].copy()
    toTime = ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        filename = f"{nickname}-ties-mnk-toTime.csv"
        print(f"\tsaved untimed ties to file {filename}")
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{filename}",index=False)
    else:
        print(f"all ties were timed for mnk {nickname}.")
    return ties,full

# less naive solution
# maximize by mnk product
# only consider tiling schemes with m divisible by 8
def mnk_mDiv8(full,timed):
    m=timed["M"][0]
    n=timed["N"][0]
    k=timed["K"][0]
    nickname=f"{m}x{n}x{k}w"
    full["mnk"] = full["m"]*full["n"]*full["k"]
    full["mMod8"] = full["m"] % 8
    pruned = full[full["mMod8"]==0].copy()
    pruned = pruned.sort_values(by="mnk",ascending=False)
    pruned = pruned.reset_index(drop=True)
    bestDiv8 = pruned[["mnk"]].loc[0]["mnk"]
    ties = pruned[pruned["mnk"]==bestDiv8].copy()
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        filename=f"{nickname}-ties-mnkDiv8-toTime.csv"
        print(f"\tsaved untimed ties to file {filename}")
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{filename}",index=False)
    else:
        print(f"all ties were timed for mnk_mDiv8 {nickname}.")
    return ties,pruned


# deeploy solution
# maximize by mnk product
# prefer tiling schemes with m divisible by 8
# no k tiling
def deeploy(full,timed):
    m=timed["M"][0]
    n=timed["N"][0]
    k=timed["K"][0]
    nickname=f"{m}x{n}x{k}w"
    full["mnk"] = full["m"]*full["n"]*full["k"]
    full["mMod8"] = full["m"] % 8
    pruned = full[full["k"]==k].copy() # no k tiling
    if pruned.empty:
        print("warning: tiling scheme not found for deeploy method!")
        return None,None
    # split into div8  and not div 8
    div8 = pruned[pruned["mMod8"]==0].copy()
    div8 = div8.sort_values(by="mnk",ascending=False)
    div8 = div8.reset_index(drop=True)
    notDiv8 = pruned[pruned["mMod8"]!=0].copy()
    notDiv8 = notDiv8.sort_values(by="mnk",ascending=False)
    notDiv8 = notDiv8.reset_index(drop=True)
    # print("Deeploy div8 options:")
    # print(div8[["FakeNN JSON Name","mnk"]])
    # print("Deeploy NOT div8 options:")
    # print(notDiv8[["FakeNN JSON Name","mnk"]].head(10))
    if not div8.empty:
        best = div8[["mnk"]].loc[0]["mnk"]
        ties = div8[div8["mnk"]==best].copy()
    else:
        best = notDiv8[["mnk"]].loc[0]["mnk"]
        ties = notDiv8[notDiv8["mnk"]==best].copy()
    # if ties.shape[0]==1:
    #     print("\nDeeploy ties: none")
    # else:
    #     print("\nDeeploy ties:")
    #     print(ties[["FakeNN JSON Name","mnk"]])
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        filename=f"{nickname}-ties-deeploy-toTime.csv"
        print(f"\tsaved untimed ties to file {filename}")
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{filename}",index=False)
    else:
        print(f"all ties were timed for deeploy {nickname}.")
    return ties,pruned

# myrtle original
# prune by min third cluster tiles
# prune by n-dim not-divisible-by-32
# prune by m-boundary tile divisible-by-8
# maximize by per-core-mnk product
# prefer tiling schemes with m divisible by 8
# tie break with smaller mn/k ratio
def myrtle(analyzed,timed,htmlTable=True):
    def prune(df):
         df["nMod32"]=df["n"]%32
         df= df[df["nMod32"]!=0].copy()
         df= df[df["nMod32"]!=16].copy()
         if df.empty:
             raise(Exception("DF empty after removing n-dim nMod32 = 0, 16"))
         df = df[df["niceMRem"]].copy()
         return df
    m=timed["M"][0]
    n=timed["N"][0]
    k=timed["K"][0]
    nickname=f"{m}x{n}x{k}w"
    prunePoint = ssr_prune_frac(analyzed,3)
    pruned = analyzed[analyzed["SSR Config Count"] < prunePoint].copy()
    pruned = prune(pruned)
    timed = prune(timed)
    ut = pruned[~pruned["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])].copy()
    timed["timed"] = True
    ut["timed"] = False
    combined = pd.concat([timed, ut])
    combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
    concatenated_df, table, high_priority_df, ties = rankWithShelves("(prioritizing m | 8, then smaller m*n / k ratio)",combined,"Avg m'_sz*n_sz / k_sz",htmlTable=htmlTable)
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        filename=f"{nickname}-ties-myrtle-toTime.csv"
        print(f"\tsaved untimed ties to file {filename}")
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{filename}",index=False)
    else:
        print(f"all ties were timed for myrtle {nickname}.")
    return ties, table, high_priority_df

# myrtle modified
# prune by min third cluster tiles
# prune by n-dim not-divisible-by-32
# prune by m-boundary tile divisible-by-8
# maximize by mnk product
# prefer tiling schemes with m divisible by 8
# tie break with smaller mn/k ratio
def mnk_myrtle(analyzed,timed,htmlTable=True):
    def prune(df):
         df["nMod32"]=df["n"]%32
         df= df[df["nMod32"]!=0].copy()
         df= df[df["nMod32"]!=16].copy()
         if df.empty:
                      raise(Exception("DF empty after removing n-dim nMod32 = 0, 16"))
         df = df[df["mDiv8"]==True].copy()
         return df
    m=timed["M"][0]
    n=timed["N"][0]
    k=timed["K"][0]
    nickname=f"{m}x{n}x{k}w"
    prunePoint = ssr_prune_frac(analyzed,3)
    pruned = analyzed[analyzed["SSR Config Count"] < prunePoint].copy()
    pruned = prune(pruned)
    timed = prune(timed)
    ut = pruned[~pruned["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])].copy()
    timed["timed"] = True
    ut["timed"] = False
    combined = pd.concat([timed, ut])
    shelvingMetric="mnk"
    combined = combined.sort_values(by=shelvingMetric,ascending=False)
    concatenated_df, table, high_priority_df, ties = rankWithShelves("(prioritizing m | 8, then smaller m*n / k ratio)",combined,"Avg m'_sz*n_sz / k_sz",shelfMetric=shelvingMetric,htmlTable=htmlTable)
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        filename=f"{nickname}-ties-mnk-myrtle-toTime.csv"
        print(f"\tsaved untimed ties to file {filename}")
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{filename}",index=False)
    else:
        print(f"all ties were timed for mnk-myrtle {nickname}.")
    return ties, table, high_priority_df

# new myrtle
# rankWithShelvesParameterized(title, df, 
# shelfMetric="FMADDsMULsPerCore", 
# sortShelfMetric0=("mMod8",True),
# sortShelfMetric1=("Avg m'_sz*n_sz / k_sz",True),htmlTable=True):
# prune by min third cluster tiles
# prune by n-dim not-divisible-by-32
# prune by m-boundary tile divisible-by-8
# maximize by per-core-mnk product
# prefer tiling schemes with m divisible by 8
# tie break with smaller mn/k ratio
def new_myrtle(analyzed,timed,htmlTable=True):
    def prune(df):
         df["mn/k"]=df["m"]*df["n"]/df["k"]
         df["nMod32"]=df["n"]%32
         df= df[df["nMod32"]!=16].copy()
         if df.empty:
             raise(Exception("DF empty after removing n-dim nMod32 = 0, 16"))
         df = df[df["niceMRem"]].copy() # prune by m-boundary tile divisible-by-8
         return df
    m=timed["M"][0]
    n=timed["N"][0]
    k=timed["K"][0]
    nickname=f"{m}x{n}x{k}w"
    prunePoint = ssr_prune_frac(analyzed,3)
    pruned = analyzed[analyzed["SSR Config Count"] < prunePoint].copy()
    pruned = prune(pruned)
    timed = prune(timed)
    ut = pruned[~pruned["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])].copy()
    timed["timed"] = True
    ut["timed"] = False
    combined = pd.concat([timed, ut])
    shelf="Avg m'_sz / k_size"
    sortShelf0=("mMod8",True)# we set ascending to True
    sortShelf1=("FMADDsMULsPerCore",False)
    comments = "(prioritizing m | 8)"
    combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
    concatenated_df, table, high_priority_df, ties = rankWithShelvesParameterized(comments,combined,shelfMetric=shelf,sortShelfMetric0=sortShelf0,sortShelfMetric1=sortShelf1,htmlTable=htmlTable)
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        filename=f"{nickname}-ties-myrtle-toTime.csv"
        print(f"\tsaved untimed ties to file {filename}")
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{filename}",index=False)
    else:
        print(f"all ties were timed for myrtle {nickname}.")
    return ties, table, high_priority_df