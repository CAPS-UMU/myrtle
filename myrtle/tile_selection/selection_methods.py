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

def rankWithShelves(title, df, colorCol, shelfMetric="FMADDsMULsPerCore"):
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
    my_columns = [
        "JSON Name",
        "mnkRem",
        "timed",
        "Avg n'_sz / k_size",
        "m'_sz*n_sz / k_sz",
        colorCol,
        "Avg B'",
        "Time (cycles)",
        "diff",
    ]
    my_columns = [
        "JSON Name",
        "timed",
        "Avg m'_sz*n_sz / k_sz",
        "FMADDsMULsPerCore",
        "mRem",
        "mnk",
        colorCol,
        "tileB",
        "Time (cycles)",
        "diff",
    ]
    my_columns = [
        "JSON Name",
        "timed",
        "Avg m'_sz*n_sz / k_sz",
        "mnk",
        "FMADDsMULsPerCore",
        "mnkRem",
        "mMod8",
        "nMod32",
        colorCol,
        "Avg C-M Ratio",
        "Time (cycles)",
        "diff",
    ]

    # Get unique FMADD values, sort them from smallest to largest
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
    table = stack_dfs_to_html_w_toggle(
        processed_dfs[0:finalShelf], my_titles[0:finalShelf], my_columns, title
    )
    
    # 3. Remove low priority mMod8!= 0 rows wherever possible
    high_priority_only = []
    for shelf in processed_dfs:
        shelf_high_priority = shelf[shelf["mMod8"]==0].copy()
        shelf_high_priority = shelf_high_priority.sort_values(colorCol,ascending=True)
        if shelf_high_priority.empty:
            high_priority_only.append(shelf)
        else:
            high_priority_only.append(shelf_high_priority)
    high_priority_df = (
            pd.concat(high_priority_only, ignore_index=True) if high_priority_only else pd.DataFrame()
        )
    ties = high_priority_only[0]
    # 4. Return the shelves concatenated, visual table representation and shelves w/ rank mMod8 != 0 removed
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
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}-ties-mnk-untimed.csv",index=False)
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
    full["mDiv8"] = full["m"] % 8
    pruned = full[full["mDiv8"]==0].copy()
    pruned = pruned.sort_values(by="mnk",ascending=False)
    pruned = pruned.reset_index(drop=True)
    bestDiv8 = pruned[["mnk"]].loc[0]["mnk"]
    ties = pruned[pruned["mnk"]==bestDiv8].copy()
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}-ties-mnkDiv8-untimed.csv",index=False)
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
    full["mDiv8"] = full["m"] % 8
    pruned = full[full["k"]==k].copy() # no k tiling
    if pruned.empty:
        raise(Warning("tiling scheme not found for this method!"))
    return None,None
    #else:
        # print("all options when k cannot be tiled:")
        # print(pruned[["FakeNN JSON Name","mnk"]])
    pruned = pruned.sort_values(by="mnk",ascending=False)
    pruned = pruned.reset_index(drop=True)
    best = pruned[["mnk"]].loc[0]["mnk"]
    # split into div8  and not div 8
    div8 = pruned[pruned["mDiv8"]==0].copy()
    notDiv8 = pruned[pruned["mDiv8"]!=0].copy()
    if not div8.empty:
        div8 = div8.sort_values(by="mnk",ascending=False)
        div8 = div8.reset_index(drop=True)
        bestDiv8 = div8[["mnk"]].loc[0]["mnk"]
    else:
        bestDiv8 = best - 1
    if not notDiv8.empty:
        notDiv8 = notDiv8.sort_values(by="mnk",ascending=False)
        notDiv8 = notDiv8.reset_index(drop=True)
        bestNotDiv8 = notDiv8[["mnk"]].loc[0]["mnk"]
    else:
        bestNotDiv8 = best - 1
    # if the best div8 option is equal to or better than the notDiv8 option, 
    # discard all notDiv8 options, and we know that the best is divisible by 8.
    if bestDiv8 >= bestNotDiv8:
        pruned = div8
    ties = pruned[pruned["mnk"]==best].copy()
    toTime=ties[~ties["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
    if not toTime.empty:
        toTime.to_csv(f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}-ties-deeploy-timed.csv",index=False)
    return ties,pruned

# myrtle original
# prune by min third cluster tiles
# prune by n-dim not-divisible-by-32
# prune by m-boundary tile divisible-by-8
# maximize by per-core-mnk product
# prefer tiling schemes with m divisible by 8
# tie break with smaller mn/k ratio
def myrtle(analyzed,timed):
    def prune(df):
         df["nMod32"]=df["n"]%32
         df= df[df["nMod32"]!=0].copy()
         if df.empty:
             raise(Exception("DF empty after removing n-dim not div 32"))
         df = df[df["niceMRem"]].copy()
         return df
    prunePoint = ssr_prune_frac(analyzed,3)
    pruned = analyzed[analyzed["SSR Config Count"] < prunePoint].copy()
    pruned = prune(pruned)
    timed = prune(timed)
    ut = pruned[~pruned["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])].copy()
    timed["timed"] = True
    ut["timed"] = False
    combined = pd.concat([timed, ut])
    combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
    concatenated_df, table, high_priority_df, ties = rankWithShelves("(prioritizing m | 8, then smaller m*n / k ratio)",combined,"m'_sz*n_sz / k_sz")
    return ties, table, high_priority_df

# myrtle modified
# prune by min third cluster tiles
# prune by n-dim not-divisible-by-32
# prune by m-boundary tile divisible-by-8
# maximize by mnk product
# prefer tiling schemes with m divisible by 8
# tie break with smaller mn/k ratio
def mnk_myrtle(analyzed,timed):
    def prune(df):
         df["nMod32"]=df["n"]%32
         df= df[df["nMod32"]!=0].copy()
         if df.empty:
             raise(Exception("DF empty after removing n-dim not div 32"))
         df = df[df["niceMRem"]].copy()
         return df
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
    concatenated_df, table, high_priority_df, ties = rankWithShelves("(prioritizing m | 8, then smaller m*n / k ratio)",combined,"m'_sz*n_sz / k_sz",shelfMetric=shelvingMetric)
    return ties, table, high_priority_df
