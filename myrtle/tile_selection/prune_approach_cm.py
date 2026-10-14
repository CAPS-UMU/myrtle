def pruneApproachCM(timed, analyzed, full):
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
        "C-M Ratio Variance",
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
        "Avg C-M Ratio",
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

     y_col = "diff"#"Avg n'_sz / k_size"
     x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     timed=timed.sort_values("mDiv8")
     special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "C-M Ratio Variance",#"Avg n'_sz / k_size",
            hover_data,
            "Average Compute-Memory Overlap vs FMADDs ",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     special_figs[-1].update_traces(showlegend=False)


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
     nickname=f"{m}x{n}x{k}-between6and9k"
     filename= f"{pathlib.Path(__file__).parent.resolve()}/out/{nickname}"
     agressivelyPruned.to_csv(filename,index=False)

     # remove n dim divisible by 32
     combined = combined[combined["nMod32"]!=0].copy()
    

     combined = combined.sort_values(by="FMADDsMULsPerCore",ascending=False)
     table = ""
     table2, asDF, filteredDF, three_shelves =printFinalRankingTwoShelves("(prioritizing mRem = 0, then smaller m*n / k ratio)",combined,"m'_sz*n_sz / k_sz")
    
    #  help = genShelfGraphPDF(jugaadTitle(timed),three_shelves,hover_data,color="mRem")
     specialHover = [ "JSON Name",
        "diff",
        "C-M Ratio Variance",
        "FMADDsMULsPerCore",
        "Time (cycles)",#"SSR Loads per HW Loop",
        "Total CL Tiles",
        "L1 Usage",
        "Avg CC Tile Size",
        "mRem",
        "tileC",
        "Avg C-M Ratio",]


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

     y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         timed,
            x_col,
            y_col,
            "C-M Ratio Variance",
            specialHover,
            "Average Compute-Memory Overlap vs Percentage Diff from Best",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     special_figs[-1].update_traces(showlegend=False)
     
     fewer = timed[timed["Avg C-M Ratio"]<20].copy()
     focused = fewer[fewer["Avg C-M Ratio"]>13].copy()
     y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         focused,
            x_col,
            y_col,
            "C-M Ratio Variance",
            specialHover,
            "Average Compute-Memory Overlap vs Percentage Diff from Best (Focused)",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     special_figs[-1].update_traces(showlegend=False)

     fewer = timed[timed["Avg C-M Ratio"]<20].copy()
     focused = fewer[fewer["Avg C-M Ratio"]>13].copy()
     y_col = "diff"#"Global Sim E2E_dma"#"Avg n'_sz / k_size"
     x_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
     special_figs.append(scatterWithColorSymbol(
         focused,
            x_col,
            y_col,
            "FMADDsMULsPerCore",
            specialHover,
            "Average Compute-Memory Overlap vs Percentage Diff from Best (Focused)",
            "mDiv8",
            ["circle","triangle-up"]
     ))
     special_figs[-1].update_traces(showlegend=False)

     #FMADDsMULsPerCore

    #  x_col = "FMADDsMULsPerCore"#"Avg n'_sz / k_size"
    #  y_col = "Avg C-M Ratio"#"m'_sz*n_sz / k_sz"#"Global Sim E2E_dma"
    #  special_figs.append(scatterWithColorSymbol(
    #      timed,
    #         x_col,
    #         y_col,
    #         "Global Sim E2E_dma",
    #         specialHover,
    #         "Average Compute-Memory Overlap vs E2E Execution Time",
    #         "mDiv8",
    #         ["circle"]
    #  ))
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
   #  special_figs.append(plot_3d_scatter(timed,("FMADDsMULsPerCore","diff","m'_sz*n_sz / k_sz"),specialHover))
     return special_figs,more_figs,f"<span>{table2}</span><span>{table}</span>"
