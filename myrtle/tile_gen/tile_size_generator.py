from dataclasses import dataclass, field
import pandas as pd
import sys
from itertools import product
import tile_SA.quidditch_load_counting as qlc
from tile_SA.utils import InputMatrix, TileSizes, roundUpToNearestMultipleOf
# @dataclass
# class InputMatrix:
#     """Class for keeping track of matrix dimensions in"""
#     """matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)"""
#     n: int = 1200
#     k: int = 400


class TileSizeGenerator:
    def __init__(self, M_dim, outputVectorEltCount, inputVectorEltCount, dispatchName=""):
        self.me = InputMatrix(m=M_dim,n=outputVectorEltCount, k=inputVectorEltCount)
        self.dispatchName = dispatchName
    
    def dividesIntoM(self, num):
        return self.me.m % num == 0

    def dividesIntoN(self, num):
        return self.me.n % num == 0

    def dividesIntoK(self, num):
        return self.me.k % num == 0

    def myfunc(self):
        print(f"I am {self.me}")
        self.validOptions()
    
    def mDimOptions(self):
        max = self.me.m
        min = 8
        exhaustive = list(range(min, max + 1))
        if (self.me.k % 2) != 0:
            print(f"WARNING: K = {self.me.k} is NOT divisible by 2!")
        return exhaustive

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
    
    def paddedMDimOptions(self):
        step = 1
        if self.me.m < 40:
            return [self.me.m]
        else: # we want about 40 tile sizes to pick from
            step = self.me.m // 40
        max = self.me.m
        min = 8
        exhaustive = list(range(min, max + 1,step))
        if (self.me.k % 2) != 0:
            print(f"WARNING: M = {self.me.m} is NOT divisible by 2!")
        return exhaustive

    def validOptions(self):
        # all possible values for m, n, and k
        little_m_options=self.mDimOptions()
        little_n_options = self.rowDimOptions()
        little_k_options = self.reductionDimOptions()
        # filter for m's, n's and k's that divide evenly into M, N and K respectively
        little_m_no_pad = list(filter(lambda x: self.dividesIntoM(x), little_m_options))
        m_options = little_m_no_pad
        little_n_no_pad = list(filter(lambda x: self.dividesIntoN(x), little_n_options))
        if len(little_m_no_pad) <= 1: # prime N dimension, or not divisible by 8
            m_options = self.paddedMDimOptions()
        else:
            m_options = little_m_no_pad
        little_n_no_pad = list(filter(lambda x: self.dividesIntoN(x), little_n_options))
        if len(little_n_no_pad) <= 1: # prime N dimension, or not divisible by 8
            n_options = self.paddedNDimOptions()
        else:
            n_options = little_n_no_pad
        little_k_no_pad = list(filter(lambda x: self.dividesIntoK(x), little_k_options))
        # if len(little_k_no_pad) < 5: # prime K dimension, or unfriendly K dim
        #     print(f'k_options without padding: {little_k_no_pad}')
        #     no_pad = set(little_k_no_pad)
        #     pad = set(self.paddedKDimOptions())
        #     print(f'k_options without padding as SET: {no_pad}')
        #     print(f'k_options with padding as SET: {list(pad)}')
        #     k_options = list(no_pad.union(pad))
        if len(little_k_no_pad) == 1: # prime K dimension
            k_options = self.paddedKDimOptions()
        else:
            k_options = little_k_no_pad
        # halve k dim options for double buffering
        k_options = list(
        filter(lambda x: x <= (self.me.k // 2) + 1, k_options))
        options_as_triples = list(product(m_options,n_options, k_options))
        #print(f"{options_as_triples}")
        annotated_options = list(map(lambda tup: self.annotateOption(tup), options_as_triples))
        #print(annotated_options)
        # filter out tiling schemes that do not fit in L1
        valid_options = list(
            filter(lambda tup: self.smallEnough(tup[0][0], tup[0][1],tup[0][2]), annotated_options)
        )
        print(f'abaout to return valid options: {valid_options}')
        return valid_options

    def weightMatTileSize(self, row_dim, reduction_dim):
        return row_dim * reduction_dim

    def spaceForTiles(self, m_dim, row_dim, reduction_dim):
        # ignore output matrix tiles
        # space in element count
        inputMatTile = m_dim * reduction_dim
        weightMatTiles = 2 * self.weightMatTileSize(row_dim, reduction_dim)
        space = inputMatTile + weightMatTiles
        # space in  bytes
        spaceInBytes = space * 8  # number of elements * 8 bytes per element
        if((m_dim == 20) and (row_dim == 40) and (reduction_dim == 10)):
            print(f'for input matrix, allocate {m_dim}x{reduction_dim} = {m_dim * reduction_dim} elements')
            print(f'for weight matrix, allocate {row_dim}x{reduction_dim} ={self.weightMatTileSize(row_dim, reduction_dim)} elements')
            print(f'for double buffering, allocate again {self.weightMatTileSize(row_dim, reduction_dim)} elements')
            # print(f'so total is {space} elements = {spaceInBytes} bytes')
        return spaceInBytes

    def spaceRemaining(self, m_dim, row_dim, reduction_dim):
        l1MemoryBytes = 100000
        outputMatMul_m = roundUpToNearestMultipleOf(self.me.m, m_dim)
        outputMatMul_n = roundUpToNearestMultipleOf(self.me.n, row_dim)
        outputMatMul = outputMatMul_m * outputMatMul_n * 8
        # if we padded the row dimension, 
        # we allocate an extra (unused) buffer of size m*n
        remainder = self.me.n % row_dim
        if (remainder != 0):
            outputMatMul = outputMatMul + outputMatMul_m*self.me.n*8
        # what happens to allocation if we pad the M dimension???
        remainder = self.me.m % m_dim
        if (remainder != 0):
            raise Exception(f"we do not support padding m-dimension yet!")
        outputElemAdd = outputMatMul
        inputElemAdd = self.me.n * 8
        remaining = (
            l1MemoryBytes
            - outputMatMul
            - outputElemAdd
            - inputElemAdd
            - self.spaceForTiles(m_dim, row_dim, reduction_dim)
        )
        if((m_dim == 20) and (row_dim == 40) and (reduction_dim == 10)):
            # print(f'm_dim is {m_dim}')
            print(f'output of matmul has size {outputMatMul_m}x{outputMatMul_n} = {outputMatMul_m * outputMatMul_n}={outputMatMul}')
            print(f'output of matmul has size {outputMatMul_m}x{outputMatMul_n} = {outputMatMul_m * outputMatMul_n}={outputElemAdd}')
            print(f'bias has size {inputElemAdd/8}')
            print(f"so remaining is {remaining}")
            print()
        return remaining

    def smallEnough(self,m_dim, row_dim, red_dim):
        return self.spaceRemaining(m_dim, row_dim, red_dim) > 0

    # annotate a (row_dim, reduction_dim) pair with
    # total L1 space used for tiles
    # weight matrix tile size
    # total spaced used in L1
    # space remaining, etc.
    def annotateOption(self, tup):
        return (
            tup,
            self.spaceForTiles(tup[0], tup[1], tup[2]),
            self.weightMatTileSize(tup[1], tup[2]),
            self.spaceRemaining(tup[0], tup[1], tup[2]),
        )

    # annotate a (m_dim, row_dim, reduction_dim) triple with
    # quidditch load counting information
    # flatten tuple
    # matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)
    #
    #         outputVectorEltCount = N (AKA "row_dim")
    #         inputVectorEltCount = K (AKA "reduction dim")
    #
    def flattenThenAnnotateMore(self, ann, caseNo: int):
        input = InputMatrix(m=self.me.m,n=self.me.n, k=self.me.k)
        tiles = TileSizes(m=ann[0][0],n=ann[0][1], k=ann[0][2])
        flat = self.convertAnnotationToFlatTuple(ann)
        # print(f'FLAT IS {flat}')
        loadInfo = (caseNo,) + qlc.getLoadCountingAnn(input, tiles)
        concatted = flat + loadInfo
        return concatted

    # helper for converting to CSV
    def convertAnnotationToFlatTuple(self, elt):
        return (
            f"{elt[0][0]}-{elt[0][1]}-{elt[0][2]}",
            elt[0][0],
            elt[0][1],
            elt[0][2],
            elt[1],
            elt[2],
            elt[3],
        )

    # helper for converting to CSV
    def annotationColumnNames(self):
        columns = [
            "JSON Name",
            "m Dim",
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
            f"TSG: wrote search space to ./{dispatchName}_case{caseNo}_searchSpace.csv"
        )
        return df

    def addMoreColsForConvenience(df):
        df["Total Loads"] = df["Regular Loads"] + df["Total Streaming Loads"]
        return df

def main():
        args = sys.argv[1:]


if __name__ == "__main__":
    main()
