from itertools import product
import pandas as pd
import sys
import quidditch_load_counting as qlc
from utils import roundUpToNearestMultipleOf, InputMatrix, TileSizes

# inputVectorEltCount = 400  #
# outputVectorEltCount = 1200  # N
# dispatchName = "dispatch_1"

# inputVectorEltCount = 600
# outputVectorEltCount = 600
# dispatchName = "dispatch_8"

# inputVectorEltCount = 400
# outputVectorEltCount = 600
# dispatchName = "dispatch_7"

# inputVectorEltCount = 600
# outputVectorEltCount = 161
# dispatchName = "dispatch_9"


def weightMatTileSize(row_dim, reduction_dim):
    return row_dim * reduction_dim


def spaceForTiles(row_dim, reduction_dim):
    # space in element count
    inputVectorTile = 1 * reduction_dim
    weightMatTiles = 2 * weightMatTileSize(row_dim, reduction_dim)
    space = inputVectorTile + weightMatTiles
    # space in  bytes
    spaceInBytes = space * 8
    # print(f'0-{48}-{100}:\ninputVectorTile elts: {inputVectorTile} * 8 = {inputVectorTile*8}. \nweightMatTiles elts: {weightMatTiles} * 8 = {weightMatTiles*8}.\ntotal in bytes: {spaceInBytes}.')
    return spaceInBytes


def spaceRemaining(row_dim, reduction_dim):
    l1MemoryBytes = 100000
    outputMatVec = roundUpToNearestMultipleOf(outputVectorEltCount, row_dim) * 8
    outputFusedAdd = outputVectorEltCount * 8
    inputFusedAdd = outputVectorEltCount * 8
    remaining = (
        l1MemoryBytes
        - outputMatVec
        - outputFusedAdd
        - inputFusedAdd
        - spaceForTiles(row_dim, reduction_dim)
    )
    return remaining


def smallEnough(row_dim, red_dim):
    return spaceRemaining(row_dim, red_dim) > 0


def dividesIntoOutputVectorEltCount(num):
    return outputVectorEltCount % num == 0


def dividesIntoInputVectorEltCount(num):
    return inputVectorEltCount % num == 0


def rowDimOptions(c="n"):
    hardware_loop_body_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    byEight = list(map(lambda x: 8 * x, hardware_loop_body_options))
    max = outputVectorEltCount
    min = byEight[0]
    exhaustive = list(range(min, max, 8))
    if c == "p":
        print("rowDimOptions")
        print(f"\tMinimum: 3*8 = {min} \n\tMaximum: N = {max}\n", end="")
        print(f"\t{len(byEight)} sensible options: ")
        print(f"\t{byEight}")
        print(f"\t{len(exhaustive)} total options: ")
        print(f"\t{exhaustive}")
        print("\tReturning Exhaustive.")
    return exhaustive


def reductionDimOptions(c="n"):
    max = inputVectorEltCount
    min = 8
    sensible = list(range(min, (inputVectorEltCount // 2) + 1))
    exhaustive = list(range(min, inputVectorEltCount + 1))
    if (inputVectorEltCount % 2) != 0:
        print(f"WARNING: M = {inputVectorEltCount} is NOT divisible by 2!")
    if c == "p":
        print("reductionDimOptions")
        print(f"\tMinimum: {min} \n\tMaximum: M = {max}\n", end="")
        print(f"\t{len(sensible)} sensible options: ")
        print(f"\t{sensible}")
        print(f"\t{len(exhaustive)} total options: ")
        print(f"\t{exhaustive}")
        print("\tReturning Exhaustive.")
    return exhaustive


def case_1_exhaustive_options(c="n"):
    row_dim_no_padding = list(
        filter(lambda x: dividesIntoOutputVectorEltCount(x), rowDimOptions())
    )
    reduction_dim_no_padding = list(
        filter(lambda x: dividesIntoInputVectorEltCount(x), reductionDimOptions())
    )
    exhaustive = list(product(row_dim_no_padding, reduction_dim_no_padding))
    if c == "p":
        print(f"case_1_exhaustive_options")
        print(
            f"\tlen {len(row_dim_no_padding)}; row_dim_no_padding: {row_dim_no_padding}"
        )
        print(
            f"\tlen {len(reduction_dim_no_padding)}; reduction_dim_no_padding: {reduction_dim_no_padding}"
        )
    return exhaustive


def case_1_pruned_options(c="n"):
    row_dim_no_padding = list(
        filter(lambda x: dividesIntoOutputVectorEltCount(x), rowDimOptions())
    )
    reduction_dim_no_padding = list(
        filter(lambda x: dividesIntoInputVectorEltCount(x), reductionDimOptions())
    )
    if (inputVectorEltCount % 2) != 0:
        print(f"WARNING: M = {inputVectorEltCount} is NOT divisible by 2!")
    reduction_dim_halve_options_for_double_buffering = list(
        filter(lambda x: x <= (inputVectorEltCount // 2) + 1, reduction_dim_no_padding)
    )
    greaterThan20 = list(
        filter(lambda x: x > 20, reduction_dim_halve_options_for_double_buffering)
    )
    pruned = list(product(row_dim_no_padding, greaterThan20))
    if c == "p":
        print("case_1_pruned_options")
        print(
            f"\tlen {len(row_dim_no_padding)}; row dim unchanged: {row_dim_no_padding}"
        )
        print(
            f"\tlen {len(reduction_dim_halve_options_for_double_buffering)}; reduction dims pruned for double buffering: {reduction_dim_halve_options_for_double_buffering}"
        )
        print(
            f"\tlen {len(greaterThan20)}; pruned away reduction dims < 20: {greaterThan20}"
        )
    return pruned


def case_2_exhaustive_options(c="n"):
    row_dim_no_padding = list(
        filter(lambda x: dividesIntoOutputVectorEltCount(x), rowDimOptions())
    )
    # STRICTLY CASE 2
    reduction_dim_no_padding = list(
        filter(lambda x: dividesIntoInputVectorEltCount(x), reductionDimOptions())
    )
    reduction_dim_w_o_case_1 = list(
        set(reductionDimOptions()).difference(set(reduction_dim_no_padding))
    )
    exhaustive = list(product(row_dim_no_padding, reduction_dim_w_o_case_1))
    if c == "p":
        print(f"case_2_exhaustive_options")
        print(
            f"\tlen {len(row_dim_no_padding)}; row_dim_no_padding: {row_dim_no_padding}"
        )
        print(
            f"\tlen {len(reduction_dim_w_o_case_1)}; reduction_dim_w_o_case_1: {reduction_dim_w_o_case_1}"
        )
    return exhaustive


def case_2_pruned_options(c="n"):
    row_dim_no_padding = list(
        filter(lambda x: dividesIntoOutputVectorEltCount(x), rowDimOptions())
    )
    # STRICTLY CASE 2
    reduction_dim_no_padding = list(
        filter(lambda x: dividesIntoInputVectorEltCount(x), reductionDimOptions())
    )
    reduction_dim_w_o_case_1 = list(
        set(reductionDimOptions()).difference(set(reduction_dim_no_padding))
    )
    if (inputVectorEltCount % 2) != 0:
        print(f"WARNING: M = {inputVectorEltCount} is NOT divisible by 2!")
    reduction_dim_halve_options_for_double_buffering = list(
        filter(lambda x: x <= (inputVectorEltCount // 2) + 1, reduction_dim_w_o_case_1)
    )
    reduction_dim_halve_options_for_double_buffering.sort()  # sort from least to greatest before taking every 10th
    reduction_dim_every_tenth = []
    for i in range(0, len(reduction_dim_halve_options_for_double_buffering)):
        if (i % 10) == 0:
            reduction_dim_every_tenth.append(
                reduction_dim_halve_options_for_double_buffering[i]
            )
    greaterThan20 = list(filter(lambda x: x > 20, reduction_dim_every_tenth))
    exhaustive = list(product(row_dim_no_padding, greaterThan20))
    if c == "p":
        print(f"case_2_pruned_options")
        print(
            f"\tlen {len(row_dim_no_padding)}; row dim left unchanged: {row_dim_no_padding}"
        )
        print(
            f"\tlen {len(reduction_dim_w_o_case_1)}; reduction dim with case 1 pruned away: {reduction_dim_w_o_case_1}"
        )
        print(
            f"\tlen {len(reduction_dim_halve_options_for_double_buffering)}; reduction dims pruned for double buffering: {reduction_dim_halve_options_for_double_buffering}"
        )
        print(
            f"\tlen {len(reduction_dim_every_tenth)}; For dispatch 1, kept every 10th reduction dim: {reduction_dim_every_tenth}"
        )
        print(
            f"\tlen {len(reduction_dim_every_tenth)}; For dispatch 1, pruned away reduction dims < 20: {greaterThan20}"
        )

    return exhaustive


def case_3_exhaustive_options(c="n"):
    row_dim_no_padding = list(
        filter(lambda x: dividesIntoOutputVectorEltCount(x), rowDimOptions())
    )
    # STRICTLY CASE 3
    row_dim_w_o_case_1 = list(set(rowDimOptions()).difference(set(row_dim_no_padding)))
    reduction_dim_no_padding = list(
        filter(lambda x: dividesIntoInputVectorEltCount(x), reductionDimOptions())
    )
    exhaustive = list(product(row_dim_w_o_case_1, reduction_dim_no_padding))
    if c == "p":
        print(f"case_3_exhaustive_options")
        print(
            f"\tlen {len(row_dim_w_o_case_1)}; row_dim_w_o_case_1: {row_dim_w_o_case_1}"
        )
        print(
            f"\tlen {len(reduction_dim_no_padding)}; reduction_dim_no_padding: {reduction_dim_no_padding}"
        )

    return exhaustive


def case_3_pruned_options(c="n"):
    row_dim_no_padding = list(
        filter(lambda x: dividesIntoOutputVectorEltCount(x), rowDimOptions())
    )
    row_dim_less_than_or_equal_to_112 = list(
        filter(lambda x: x >= 112, rowDimOptions())
    )
    w_o_case_1 = list(
        set(row_dim_less_than_or_equal_to_112).difference(set(row_dim_no_padding))
    )
    # w_o_case_1 = list(set(rowDimOptions()).difference(set(row_dim_no_padding)))
    # w_o_case_1.sort()
    # everyOther = []
    # for i in range(0,len(w_o_case_1)):
    #   if (i%2) == 0:
    #     everyOther.append(w_o_case_1[i])
    reduction_dim_no_padding = list(
        filter(lambda x: dividesIntoInputVectorEltCount(x), reductionDimOptions())
    )
    if (inputVectorEltCount % 2) != 0:
        print(f"WARNING: M = {inputVectorEltCount} is NOT divisible by 2!")
    reduction_dim_halve_options_for_double_buffering = list(
        filter(lambda x: x <= (inputVectorEltCount // 2) + 1, reduction_dim_no_padding)
    )
    greaterThan20 = list(
        filter(lambda x: x > 20, reduction_dim_halve_options_for_double_buffering)
    )
    pruned = list(product(w_o_case_1, greaterThan20))
    if c == "p":
        print("case_3_pruned_options")
        # print(f'\tlen {len(row_dim_less_than_or_equal_to_112)}; pruned away row_dims > 112: {row_dim_less_than_or_equal_to_112}')
        print(
            f"\tlen {len(w_o_case_1)}; pruned away row dims from case 1: {w_o_case_1}"
        )
        # print(f'\tlen {len(everyOther)}; For dispatch 8, pruned away every other row dim: {everyOther}')
        print(
            f"\tlen {len(reduction_dim_halve_options_for_double_buffering)}; reduction dims pruned for double buffering: {reduction_dim_halve_options_for_double_buffering}"
        )
        print(
            f"\tlen {len(greaterThan20)}; For dispatch 1, 9, pruned away reduction dims <= 20: {greaterThan20}"
        )
    return pruned


# annotate a (row_dim, reduction_dim) pair with
# total L1 space used for tiles
# weight matrix tile size
# total spaced used in L1
# space remaining, etc.
def annotateOption(tup):
    return (
        tup,
        spaceForTiles(tup[0], tup[1]),
        weightMatTileSize(tup[0], tup[1]),
        spaceRemaining(tup[0], tup[1]),
    )

# annotate a (row_dim, reduction_dim) pair with
# quidditch load counting information
# flatten tuple
# matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)
#         
#         outputVectorEltCount = N (AKA "row_dim")
#         inputVectorEltCount = K (AKA "reduction dim")
#         
# ex. python3 tile_size_gen.py "dispatch_1" 1200 400
def flattenThenAnnotateMore(ann, caseNo:int):
    input = InputMatrix(n=outputVectorEltCount,k=inputVectorEltCount)
    tiles = TileSizes(n=ann[0][0], k=ann[0][1])
    flat = convertAnnotationToFlatTuple(ann)
    loadInfo = (caseNo,) + qlc.getLoadCountingAnn(input,tiles) 
    concatted = flat + loadInfo
    return concatted

# helper for converting to CSV
def convertAnnotationToFlatTuple(elt):
    return (
        f"{0}-{elt[0][0]}-{elt[0][1]}",
        elt[0][0],
        elt[0][1],
        elt[1],
        elt[2],
        elt[3],
    )


# helper for converting to CSV
def annotationColumnNames():
    columns = [
        "JSON Name",
        "Row Dim",
        "Reduction Dim",
        "Space Needed in L1",
        "Weight Matrix Tile Size",
        "Space Remaining",
    ]
    return columns


# export annotated options to CSV
def exportOptionsToCSV(caseNo, options):
    flat = list(map(lambda tup: flattenThenAnnotateMore(tup,caseNo), options))
    cols = (annotationColumnNames()+["Case"])+qlc.LoadCountingAnnColumnNames()
    # print(f"flat: {flat}")
    # print(f"cols: {cols}")
    df = pd.DataFrame(flat, columns=cols)
    # df["Case"] = caseNo
    df.to_csv(
        f"myrtle_tile_size_gen_out/{dispatchName}_case{caseNo}_searchSpace.csv",
        index=False,
    )
    print(
        f"Saved CSV to ./myrtle_tile_size_gen_out/{dispatchName}_case{caseNo}_searchSpace.csv"
    )


def case1():
    print(
        "\nCase 1: No padding in either dimension. --------------------------------|",
        end="",
    )
    print("\n                                                                        |")
    print("Exhaustive:")
    all_options = case_1_exhaustive_options("p")
    annotated_options = list(map(lambda tup: annotateOption(tup), all_options))
    valid_options = list(
        filter(lambda tup: smallEnough(tup[0][0], tup[0][1]), annotated_options)
    )
    # print(f'{len(valid_options)} options: {valid_options}')
    print("Pruned:")
    pruned_options = case_1_pruned_options("p")
    annotated_options = list(map(lambda tup: annotateOption(tup), pruned_options))
    valid_options = list(
        filter(lambda tup: smallEnough(tup[0][0], tup[0][1]), annotated_options)
    )
    print(f"{len(valid_options)} valid options")
    exportOptionsToCSV(1, valid_options)
    print("                                                                        |")
    print("________________________________________________________________________|")


def case2():
    print(
        "\nCase 2: Padding in column dimension ONLY. ------------------------------|",
        end="",
    )
    print("\n                                                                        |")
    print("Exhaustive:")
    all_options = case_2_exhaustive_options("p")
    annotated_options = list(map(lambda tup: annotateOption(tup), all_options))
    valid_options = list(
        filter(lambda tup: smallEnough(tup[0][0], tup[0][1]), annotated_options)
    )
    print(f"{len(valid_options)} valid options")
    print("Pruned:")
    pruned_options = case_2_pruned_options("p")
    annotated_options = list(map(lambda tup: annotateOption(tup), pruned_options))
    valid_options = list(
        filter(lambda tup: smallEnough(tup[0][0], tup[0][1]), annotated_options)
    )
    print(f"{len(valid_options)} valid options")
    exportOptionsToCSV(2, valid_options)
    print("                                                                          |")
    print("__________________________________________________________________________|")


def case3():
    print(
        "\nCase 3: Padding in row dimension ONLY. -----------------------------------|",
        end="",
    )
    print(
        "\n                                                                          |"
    )
    print("Exhaustive:")
    all_options = case_3_exhaustive_options("p")
    annotated_options = list(map(lambda tup: annotateOption(tup), all_options))
    valid_options = list(
        filter(lambda tup: smallEnough(tup[0][0], tup[0][1]), annotated_options)
    )
    print(f"{len(valid_options)} valid options")
    print("Pruned:")
    pruned_options = case_3_pruned_options("p")
    annotated_options = list(map(lambda tup: annotateOption(tup), pruned_options))
    valid_options = list(
        filter(lambda tup: smallEnough(tup[0][0], tup[0][1]), annotated_options)
    )
    print(f"{len(valid_options)} valid options")
    exportOptionsToCSV(3, valid_options)
    print("                                                                          |")
    print("__________________________________________________________________________|")



# matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)
#         
#         outputVectorEltCount = N (AKA "row_dim")
#         inputVectorEltCount = K (AKA "reduction dim")
#         
# ex. python3 tile_size_gen.py "dispatch_1" 1200 400
def main():
    global dispatchName 
    global outputVectorEltCount 
    global inputVectorEltCount 
    args = sys.argv[1:]
    if len(args) != 3:
        print("USAGE: tile_size_gen.py <kernel_inst_name> <N> <K>")
        print("\t kernel_inst_name = unique name of kernel to tile")
        print("\t N = # of output elements")
        print("\t K = # of input elements")
        print(
            "^^ Representing a  matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`"
        )
        exit()
    else:
        dispatchName = args[0]
        outputVectorEltCount = int(args[1]) # (AKA N or row_dim)
        inputVectorEltCount = int(args[2]) # (AKA K or reduction_dim)

    print(
        "\n|----------- CREATING TILE SIZE/SHAPE SEARCH SPACE FOR KERNEL ----------|",
        end="",
    )
    print(
        "\n|                                                                       |",
        end="",
    )
    print(f"\n\t\t{dispatchName}")
    print(
        "\n|_______________________________________________________________________|\n",
        end="",
    )
    case1()
    case2()
    # case3()
    qlc.yodel()


if __name__ == "__main__":
    main()
