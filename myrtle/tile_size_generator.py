from dataclasses import dataclass, field
import pandas as pd
import sys
from itertools import product
import myrtle.quidditch_load_counting as qlc
from myrtle.utils import InputMatrix, TileSizes, roundUpToNearestMultipleOf
# @dataclass
# class InputMatrix:
#     """Class for keeping track of matrix dimensions in"""
#     """matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)"""
#     n: int = 1200
#     k: int = 400


class TileSizeGenerator:
    def __init__(self, outputVectorEltCount, inputVectorEltCount, dispatchName=""):
        self.me = InputMatrix(n=outputVectorEltCount, k=inputVectorEltCount)
        self.dispatchName = dispatchName

    def dividesIntoN(self, num):
        return self.me.n % num == 0

    def dividesIntoK(self, num):
        return self.me.k % num == 0

    def myfunc(self):
        print(f"I am {self.me}")
        self.validOptions()

    def rowDimOptions(self):
        hardware_loop_body_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        byEight = list(map(lambda x: 8 * x, hardware_loop_body_options))
        max = self.me.n
        min = byEight[0]
        exhaustive = list(range(min, max+1, 8))
        return exhaustive

    def reductionDimOptions(self):
        max = self.me.k
        min = 8
        exhaustive = list(range(min, max + 1))
        if (self.me.k % 2) != 0:
            print(f"WARNING: K = {self.me.k} is NOT divisible by 2!")
        return exhaustive

    def paddedNDimOptions(self):
        hardware_loop_body_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        byEight = list(map(lambda x: 8 * x, hardware_loop_body_options))
        max = roundUpToNearestMultipleOf(self.me.n, 8) # "pad" to nearest multiple of 8
        min = byEight[0]
        exhaustive = list(range(min, max+1, 8))  
        return exhaustive

    def paddedKDimOptions(self):
        step = 1
        if self.me.k < 40:
            return [self.me.k]
        else: # we want about 40 tile sizes to pick from
            step = self.me.k // 40
        max = self.me.k
        min = 8
        exhaustive = list(range(min, max + 1,step))
        if (self.me.k % 2) != 0:
            print(f"WARNING: K = {self.me.k} is NOT divisible by 2!")
      
        return exhaustive

    def validOptions(self):
        # all possible values for n and k
        little_n_options = self.rowDimOptions()
        little_k_options = self.reductionDimOptions()
        print(f'n options are {list(little_n_options)}')
        print(f'k options are {list(little_k_options)}')
        # filter for n's and k's that divide evenly into N and K
        little_n_no_pad = list(filter(lambda x: self.dividesIntoN(x), little_n_options))
        if len(little_n_no_pad) <= 1: # prime N dimension, or not divisible by 8
            n_options = self.paddedNDimOptions()
        else:
            n_options = little_n_no_pad
        little_k_no_pad = list(filter(lambda x: self.dividesIntoK(x), little_k_options))
        if len(little_k_no_pad) == 1: # prime K dimension
            k_options = self.paddedKDimOptions()
        else:
            k_options = little_k_no_pad
        # have k dim options for double buffering
        k_options = list(
        filter(lambda x: x <= (self.me.k // 2) + 1, k_options))
        print(f'now n options are {list(n_options)}')
        print(f'now k options are {list(k_options)}')
        options_as_pairs = list(product(n_options, k_options))
        annotated_options = list(map(lambda tup: self.annotateOption(tup), options_as_pairs))
        # filter by size
        valid_options = list(
            filter(lambda tup: self.smallEnough(tup[0][0], tup[0][1]), annotated_options)
        )
        print(valid_options)
        return valid_options
        # filter by size
        # sizeInfo = list(map(lambda tup: jen.annotateOption(tup), valid_options))
        # # add load counting information
        # sizeAndLoadInfo = jen.exportOptionsToCSV(dispatchName, caseNo, sizeInfo)
        # extras = jen.addMoreColsForConvenience(sizeAndLoadInfo)
        # return(extras)
       # return list(product(little_n_no_pad, little_k_no_pad))
        #return list(product(little_n_no_pad, k_dim_halve_options_for_double_buffering))

        # print("litlle n no padding: [", end="")
        # for i in little_n_no_pad:
        #     print(i, end=", ")
        # print("]")
        # print("little k no padding:[", end="")
        # for i in little_k_no_pad:
        #     print(i, end=", ")
        # print("]")
        # # return (little_n_no_pad,little_k_no_pad)
        # return list(product(little_n_no_pad, little_k_no_pad))

    def weightMatTileSize(self, row_dim, reduction_dim):
        return row_dim * reduction_dim

    def spaceForTiles(self, row_dim, reduction_dim):
        # space in element count
        inputVectorTile = 1 * reduction_dim
        weightMatTiles = 2 * self.weightMatTileSize(row_dim, reduction_dim)
        space = inputVectorTile + weightMatTiles
        # space in  bytes
        spaceInBytes = space * 8  # number of elements * 8 bytes per element
        # print(f'0-{48}-{100}:\ninputVectorTile elts: {inputVectorTile} * 8 = {inputVectorTile*8}. \nweightMatTiles elts: {weightMatTiles} * 8 = {weightMatTiles*8}.\ntotal in bytes: {spaceInBytes}.')
        return spaceInBytes

    def spaceRemaining(self, row_dim, reduction_dim):
        l1MemoryBytes = 100000
        outputMatVec = roundUpToNearestMultipleOf(self.me.n, row_dim) * 8
        # if we padded the row dimension, 
        # we allocate an extra (unused) buffer of size n
        remainder = self.me.n % row_dim
        if (remainder != 0):
            outputMatVec = outputMatVec + self.me.n*8
        outputFusedAdd = self.me.n * 8
        inputFusedAdd = self.me.n * 8
        remaining = (
            l1MemoryBytes
            - outputMatVec
            - outputFusedAdd
            - inputFusedAdd
            - self.spaceForTiles(row_dim, reduction_dim)
        )
        return remaining

    def smallEnough(self,row_dim, red_dim):
        return self.spaceRemaining(row_dim, red_dim) > 0

    # annotate a (row_dim, reduction_dim) pair with
    # total L1 space used for tiles
    # weight matrix tile size
    # total spaced used in L1
    # space remaining, etc.
    def annotateOption(self, tup):
        return (
            tup,
            self.spaceForTiles(tup[0], tup[1]),
            self.weightMatTileSize(tup[0], tup[1]),
            self.spaceRemaining(tup[0], tup[1]),
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
    def flattenThenAnnotateMore(self, ann, caseNo: int):
        input = InputMatrix(n=self.me.n, k=self.me.k)
        tiles = TileSizes(n=ann[0][0], k=ann[0][1])
        flat = self.convertAnnotationToFlatTuple(ann)
        loadInfo = (caseNo,) + qlc.getLoadCountingAnn(input, tiles)
        concatted = flat + loadInfo
        return concatted

    # helper for converting to CSV
    def convertAnnotationToFlatTuple(self, elt):
        return (
            f"{0}-{elt[0][0]}-{elt[0][1]}",
            elt[0][0],
            elt[0][1],
            elt[1],
            elt[2],
            elt[3],
        )

    # helper for converting to CSV
    def annotationColumnNames(self):
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
    def exportOptionsToCSV(self, dispatchName, caseNo, options):
        flat = list(map(lambda tup: self.flattenThenAnnotateMore(tup, caseNo), options))
        cols = (
            self.annotationColumnNames() + ["Case"]
        ) + qlc.LoadCountingAnnColumnNames()
        df = pd.DataFrame(flat, columns=cols)
        df.to_csv(
            f"./{dispatchName}_case{caseNo}_searchSpace.csv",
            index=False,
        )
        print(
            f"Saved CSV to ./{dispatchName}_case{caseNo}_searchSpace.csv"
        )
        return df

    def tupleIze(a, b):
        return (a, b)

    def get_tile_shape(row_dim, col_dim):
        if row_dim > col_dim:
            shape = "tall"
        if row_dim < col_dim:
            shape = "wide"
        if row_dim == col_dim:
            shape = "square"
        return shape

    def addMoreColsForConvenience(df):
        df["Total Loads"] = df["Regular Loads"] + df["Total Streaming Loads"]
        df["Tile Shape"] = df.apply(
            lambda x: get_tile_shape(x["Row Dim"], x["Reduction Dim"]), axis=1
        )
        return df

    def main():
        args = sys.argv[1:]


     #sizeInfo = list(map(lambda tup: jen.annotateOption(tup), valid_options))
        # add load counting information
        # sizeAndLoadInfo = jen.exportOptionsToCSV(dispatchName, caseNo, sizeInfo)
        # extras = jen.addMoreColsForConvenience(sizeAndLoadInfo)
        # return(extras)


    # if len(args) != 6:
    #   print("USAGE: python3 tile_size_generator.py <N> <K> <tilesRequested.csv> <dispatchName> <CaseNo> <oracle-output-dir>")
    #   print(
    #         "\twhere N, K, refer to dimensions in a\n\tmatrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`"
    #   )
    #   print(
    #         "\tNote: when <tilesRequested.csv> is set to ORACLE_MODE, \n\t      the tile generator will automatically generate a set of valid tile sizes, \n\t      placing them in <oracle-output-dir>."
    #   )
    #   exit(1)
    # if args[2] == "ORACLE_MODE":
    #     print("we're in oracle mode!!")
    #     dispatchName = args[3]
    #     caseNo = int(args[4])
    #     outputDir = args[5]
    #     jen = TileSizeGenerator(int(args[0]),int(args[1]))
    #     # sizeInfo = list(map(lambda tup: jen.annotateOption(tup), tups))
    #     # sizeAndLoadInfo = jen.exportOptionsToCSV(dispatchName, caseNo, sizeInfo)
    #     # extras = addMoreColsForConvenience(sizeAndLoadInfo)
    #     jen.myfunc()
    #     print(f'wrote outputs to directory {outputDir}')
    #     exit(0)
    # jen = TileSizeGenerator(int(args[0]),int(args[1]))
    # firstCSV=pd.read_csv(args[2])
    # dispatchName = args[3]
    # caseNo = int(args[4])
    # justRuntimeInfo=firstCSV[["JSON Name","Kernel Name","Kernel Time","Total Time"]]
    # firstCSV["tuples"]=firstCSV.apply(lambda x: tupleIze(x["Row Dim"], x["Reduction Dim"]), axis=1)
    # tups = firstCSV["tuples"].tolist()
    # sizeInfo = list(map(lambda tup: jen.annotateOption(tup), tups))
    # sizeAndLoadInfo = jen.exportOptionsToCSV(dispatchName, caseNo, sizeInfo)
    # extras = addMoreColsForConvenience(sizeAndLoadInfo)
    # merged = pd.merge(justRuntimeInfo,extras,on="JSON Name",how="outer")
    # print(merged)
    # merged.to_csv(f"search_spaces_out/{dispatchName}_case{caseNo}_everything.csv",index=False)

    qlc.yodel()


if __name__ == "__main__":
    main()
