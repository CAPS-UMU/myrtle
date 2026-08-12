import sys
import json
from tile_size_generation.TSG_Quidditch import TSG_Quidditch
from tile_size_generation.TSG_C import TSG_C, NoDivisorTiles, NoDivisorTilesInAnyDim
from tile_static_analysis.TSA_Quidditch import TSA_Quidditch
from tile_static_analysis.TSA_C_Remainder import TSA_C_Remainder
from tile_size_generation.TSG_C_Remainder import TSG_C_Remainder
from tile_size_generation.TSG_C_Div_Rem import TSG_C_Div_Rem
import tile_sel.tile_selection as tss
import re
import pandas as pd
import pathlib
import os.path


def tileSelection(analyzedSearchSpaceCSVName, dispatchName, mode, outputFile):
    m, n, k, dualBuffer = tss.tileSelection(analyzedSearchSpaceCSVName, sys.argv[2])
    if mode == "sflt":
        print("myrtle: ", end="")
        print("We used simple filtering to select tiles.")
    if mode == "scyc":
        print("myrtle: ", end="")
        print("We used a simple cycle estimation to select tiles.")
    if mode == "svrcyc":
        print("myrtle: ", end="")
        print("We used an SVR to select tiles.")
    # default values
    data = {}
    node = {}
    node["loop-order"] = [[2, 0], [0, 0], [1, 0]]
    # set node values and export result to JSON
    node["tile-sizes"] = [[m], [n], [k]]
    node["dual-buffer"] = dualBuffer
    data[dispatchName] = node
    f = open(outputFile, "w")
    f.write(f"{json.dumps(data)}")
    f.close()


# arg 1 is dispatchName as a string (Quidditch Backend) or matmul dimensions (Manual C code Backend)
# arg 2 is tile selection mode
# arg 3 is file to write tile scheme to
# arg 4 is file to import tiling scheme candidates (skip tile gen)
# for Example,
# python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_1x400x200_f64" sflt "test_output-disp-0.json"
def main():
    dispatchName = sys.argv[1]
    quidditch = True
    if sys.argv[1][:6] == "matmul":
        print("\tTSG: we will prune for Manual C Backend")
        dispatchRegex = re.compile(r"matmul_(\d+)x(\d+)x(\d+)_f64")
        M, N, K = dispatchRegex.search(dispatchName).groups()
        dispatchNickName = f"{M}x{N}x{K}wm-n-k"
        quidditch = False
    else:
        print("\tTSG: we will prune for Quidditch Backend")
        dispatchRegex = re.compile(
            r"main\$async_dispatch_\d+_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64"
        )
        M, N, K = dispatchRegex.search(dispatchName).groups()
        dispatchNickName = f"{M}x{N}x{K}wm-n-k"
    prune = False
    skipTSG = False
    if len(sys.argv) >= 5:
        if sys.argv[4] == "prune":
            prune = True
        else:
            searchSpaceCSVName = sys.argv[4]
            print("myrtle: ", end="")
            print("Using search space passed in from command line.")
            if not os.path.exists(searchSpaceCSVName):
                print("myrtle: ", end="")
                print(
                    "Can't find search space file. Continuing with automatic search space generation..."
                )
            else:
                options_as_df = pd.read_csv(searchSpaceCSVName)
                skipTSG = True
        spm_opt = sys.argv[5] == "optSPM"

    # Quidditch Backend
    if quidditch:
        # generate options
        if not skipTSG:
            jen = TSG_Quidditch(
                int(M), int(N), int(K), dispatchName, l1MemoryBytes=100000
            )
            options = jen.validOptions(debug=False)
            options_as_df = jen.convertOptionsToDF(dispatchNickName, options)
            searchSpaceCSVName = jen.exportOptionsToCSV(dispatchNickName, options_as_df)
        # analyze tiling options
        ann = TSA_Quidditch()
        analyzed = ann.analyze_options(options_as_df)
        analyzedSearchSpaceCSVName = ann.exportAnalysisToCSV(dispatchNickName, analyzed)
        # select best tiling scheme using mode
        tileSelection(
            analyzedSearchSpaceCSVName, dispatchName, sys.argv[2], sys.argv[3]
        )
        return

    # Manual C Backend
    # initialize the tile size analyzer
    ann = TSA_C_Remainder(8, 8)
    # initialize the tile size generator
    jen = TSG_C_Div_Rem(
        int(M),
        int(N),
        int(K),
        dispatchName=dispatchName,
        l1MemoryBytes=112 * 1024,
        bank_size=1024,
        dualBuff=True,
        optSPM=spm_opt,
    )
    options = jen.validOptions()
    options_as_df = jen.convertOptionsToDF(dispatchNickName, options)
    annotated_options = ann.annotate_w_ssr_configs(options_as_df)
    sorted_options = annotated_options.sort_values("SSR Config Count", ascending=True)
    searchSpaceCSVName = jen.exportOptionsToCSV(dispatchNickName, sorted_options)
    # print out entire search space before pruning
    filename = f"{pathlib.Path(__file__).parent.resolve()}/out/{dispatchNickName}_ss_c_rem_div.csv"
    sorted_options.to_csv(
        filename,
        index=False,
    )
    if prune:
        suffix = "_ss_c_rem_div_pruned"
        prunePoint = TSG_C_Div_Rem.ssr_prune_frac(sorted_options, 3)
        # now that we know the prune point, go ahead and prune
        pruned_options = annotated_options[
            annotated_options["SSR Config Count"] <= prunePoint
        ]
        # pruned_options = TSG_C_Div_Rem.ssr_prune_bestX(sorted_options,45)
        if pruned_options.shape[0] < 20:
            pruned_options = TSG_C_Div_Rem.ssr_prune_bestX(annotated_options, 20)
        sorted_pruned = pruned_options.sort_values("SSR Config Count", ascending=True)
        filename = f"{pathlib.Path(__file__).parent.resolve()}/out/{dispatchNickName}{suffix}.csv"
        sorted_pruned.to_csv(
            filename,
            index=False,
        )
        # only analyze points that survive pruning
        suffix = "_rem_div_ana_pruned"
        analyzed = ann.analyze_options(sorted_pruned)
        analyzed = analyzed.sort_values(
            "SSR Config Count", ascending=True, ignore_index=True
        )
    else:
        suffix = "_ss_c_rem_div_ana"
        # analyze all the points
        analyzed = ann.analyze_options(sorted_options)
        analyzed = analyzed.sort_values(
            "SSR Config Count", ascending=True, ignore_index=True
        )

    filename = (
        f"{pathlib.Path(__file__).parent.resolve()}/out/{dispatchNickName}{suffix}.csv"
    )
    analyzed.to_csv(
        filename,
        index=False,
    )
    analyzedSearchSpaceCSVName = filename
    # select best tiling scheme using mode
    tileSelection(analyzedSearchSpaceCSVName, dispatchName, sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
