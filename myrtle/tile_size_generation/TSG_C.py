from dataclasses import dataclass, field
import pandas as pd
import sys
from itertools import product, chain
from tile_static_analysis.utils import MatmulInputs, TileSizes, roundUpToNearestMultipleOf
import pathlib
from tile_size_generation.TSG_Quidditch import TSG_Quidditch

# Manual C Code Tile Constraints
# m, n, and k must all divide evenly into corresponding M,N,K input sizes
# n must be a multiple of 8 due to fixed unroll and jam factor of 8
# m, n, and k must each be able to fit into 8 TCDM banks (optimized scratchpad layout constraint)
#   BANK_SIZE = 1 * 1024  # ask Luca about this; if TCDM is 128 KiB and 32 banks,
# shouldn't bank size be 4KiB = 2048 bytes??
#   MAX_ALLOWED_SIZE = 8 * BANK_SIZE
# 2 * (a_tile_sz + b_tile_sz + c_tile_sz) < TCDM_HEAP_SIZE
#   a_tile_sz = m * k
#   b_tile_sz = n * k
#   c_tile_sz = m * n
#   (This is the Double Buffering Constraint)
# Other Notes:
# matmul with type `<MxK>, <KxN> -> <MxN>`
# m is the parallel dimension but does NOT need to be a multiple of 8
# TCDM size in bytes: TCDM_HEAP_SIZE = 112 * 1024

# class Shape:
#     def __init__(self, shapename, **kwds):
#         self.shapename = shapename
#         super().__init__(**kwds)        

# class ColoredShape(Shape):
#     def __init__(self, color, **kwds):
#         self.color = color
#         super().__init__(**kwds)
# def __init__(self, other):
# cs = ColoredShape(color='red', shapename='circle')

class TSG_C(TSG_Quidditch):
    def hello(self):
        print("I am a tile size generator for the manual C backend")
    # def __init__(self, other):
#          
    def mDimOptions(self):
        max = self.me.m
        min = 8 if self.me.m >= 8 else 1
        exhaustive = list(range(min, max + 1))
        if (self.me.m % 2) != 0:
            print(f"WARNING: M = {self.me.m} is NOT divisible by 2!")
        return exhaustive
    
    def kDimOptions(self):
        max = self.me.k
        min = 8 if self.me.k >= 8 else 3  # min is 3 due to prologue and epilogue of HW Loop in assembly
        exhaustive = list(range(min, max + 1))
        if (self.me.n % 2) != 0:
            print(f"WARNING: K = {self.me.k} is NOT divisible by 2!")
        return exhaustive

    def nDimOptions(self):
        hardware_loop_body_options = [8]  # extend to 8,5 later
        max = self.me.n  # hides built-in max function
        # ASSUMES hardware loop body options are listed LEAST to GREATEST
        if max < hardware_loop_body_options[0]:
            raise Exception("input dimension N is smaller than smallest unroll and jam factor")
        def byUnrollAndJamFactor(uAJ, max=max):
            return list(range(uAJ, max + 1, uAJ))
        multiples = list(map(byUnrollAndJamFactor, hardware_loop_body_options))
        # print(multiples) # debugging only
        # print(list(chain.from_iterable(multiples))) # debugging only
        # first convert to set to remove duplicates, then convert to list
        exhaustive = list(set(chain.from_iterable(multiples)))
        # print(exhaustive) # debugging only
        if (self.me.n % 2) != 0:
            print(f"WARNING: N = {self.me.n} is NOT divisible by 2!")
        return exhaustive

    def validOptions(self, debug=False):
        # all possible values for m, n, and k
        little_m_options = self.mDimOptions()
        little_n_options = self.nDimOptions()
        little_k_options = self.kDimOptions()

        # filter for m's, n's and k's that divide evenly into M, N and K respectively
        little_m_no_pad = list(filter(lambda x: self.dividesIntoM(x), little_m_options))
        m_options = little_m_no_pad
        if len(little_m_no_pad) < 1:  # prime M dimension
            raise Exception(
                f"TSG: Cannot find a tile size that divides evenly into dimension M = {self.me.m}!"
            )

        little_n_no_pad = list(filter(lambda x: self.dividesIntoN(x), little_n_options))
        n_options = little_n_no_pad
        if len(little_n_no_pad) < 1:  # prime N dimension
            raise Exception(
                f"TSG: Cannot find a tile size that divides evenly into dimension N = {self.me.n}!"
            )

        little_k_no_pad = list(filter(lambda x: self.dividesIntoK(x), little_k_options))
        k_options = little_k_no_pad
        if len(little_k_no_pad) < 1:  # prime K dimension
            raise Exception(
                f"TSG: Cannot find a tile size that divides evenly into dimension K = {self.me.k}!"
            )

        options_as_triples = list(product(m_options, n_options, k_options))

        annotated_options = list(
            map(lambda tup: self.annotateOptionWL1Usage(tup), options_as_triples)
        )

        # filter out tiling schemes that do not fit in L1
        valid_options_l1 = list(
            filter(lambda d: d["Space Remaining"] >= 0, annotated_options)
        )

        # filter out tile sizes that do not fit within 8 banks
        eb = 8 * self.bankSizeBytes # eb stands for "eight banks"
        valid_options_8_banks = list(
            filter(lambda d: max(d["tileA"],d["tileB"],d["tileC"]) <= eb, valid_options_l1)
        )

        # mark each of these tiling schemes with padding info
        valid_options_8_banks_no_pad = list(
            map(lambda d: self.annnotatePaddingStatus(d), valid_options_8_banks)
        )

        if debug:
            valid_options= list(
                map(lambda tup: self.dictToTuple(tup),  valid_options_l1)
            )
            print("\tTSG: options that fit in L1 are ", end="")
            print(valid_options)
            valid_options= list(
                map(lambda tup: self.dictToTuple(tup),  valid_options_8_banks)
            )
            print("\tTSG: options that fit in L1 AND comform to 8-bank constraint are ", end="")
            print(valid_options)
        if len(valid_options_8_banks) == 0:
            raise Exception("Cannot find a valid tiling scheme!")
        return valid_options_8_banks_no_pad

    # regular matmul: A : MxK, B : KxN, C : MxN
    # compute L1 usage measured in ELEMENT COUNT
    def computeL1Usage(self, m, n, k, debug=False):
        tileSpace = 0
        weightMatTileSpace = 0
        total = 0
        tileA = m * k
        tileB = n * k
        tileC = m * n
        # m is the parallel dimension
        m_prime = m // 8
        tileA_cc = m_prime * k 
        tileB_cc = n * k 
        tileC_cc = m_prime * n

        if debug:
            print("\n")
            print(f"Regular Matmul {self.me.m}-{self.me.n}-{self.me.k}:")
            print(f"Tiling Scheme {m}-{n}-{k}:")
            print(f"Allocate A tile: {m}x{k}")
            if self.dualBuff:
                print(f"Allocate A2 tile: {m}x{k}")
            print(f"Allocate B1 tile: {k}x{n}")
            if self.dualBuff:
                print(f"Allocate B2 tile: {k}x{n}")
            print(f"Allocate C tile: {m}x{n}")
            if self.dualBuff:
                print(f"Allocate C2 tile: {m}x{n}")

        # NO PADDING EVER

        total = 2 * (tileA + tileB + tileC) if self.dualBuff else tileA + tileB + tileC

        if debug:
            if self.dualBuff:
                print(
                    f"total = {tileA} + {tileB} + {tileC} = {total} elements or {total*8} bytes"
                )
            else:
                print(
                    f"total = {2} * ({tileA} + {tileB} + {tileC}) = {total} elements or {total*8} bytes"
                )
            if total * 8 > self.l1MemoryBytes:
                print(
                    f"which does NOT fit in L1 with {(total*8)-self.l1MemoryBytes} too many bytes!",
                    end="\n\n",
                )
            else:
                print(
                    f"which fits in L1 with {self.l1MemoryBytes-(total*8)} bytes to spare",
                    end="\n\n",
                )

        weightMatTileSpace = 2 * tileB if self.dualBuff else tileB
        tileSpace = total
        totalSpace = total
        return tileSpace, weightMatTileSpace, totalSpace, tileA, tileB, tileC, tileA_cc, tileB_cc, tileC_cc

    # annotate a (m, n, k) tile size triple with
    # total spaced used in L1
    # weight matrix tile size
    # space remaining
    # measured in BYTES
    def annotateOptionWL1Usage(self, tup):
        tileSpace, weightMatTileSpace, totalUsage, tileA, tileB, tileC, tileA_cc, tileB_cc, tileC_cc = self.computeL1Usage(
            tup[0], tup[1], tup[2]
        )
        return {
            "id": tup,
            "Space Needed in L1": totalUsage * 8,
            "Weight Matrix Tile Size": weightMatTileSpace * 8,
            "Space Remaining": self.l1MemoryBytes - totalUsage * 8,
            "tileA": tileA * 8,
            "tileB": tileB * 8,
            "tileC": tileC * 8,
            "tileA_cc": tileA_cc *8,
            "tileB_cc": tileB_cc * 8,
            "tileC_cc": tileC_cc * 8
        }
    
    def annnotatePaddingStatus(self, d):
        d.update({"remainderTiles": "000", "Mpad": 0,"Npad": 0,"Kpad": 0})
        return d

    # convert dictionary to simpler, more readable, annotated triple
    def dictToTuple(self, d):
        return (
            d["id"],
            d["Space Needed in L1"],
            d["Weight Matrix Tile Size"],
            d["Space Remaining"]
        )

    # flatten dictionary + add more information
    def convertAnnotationToFlatDict(self, d):
        tup = d["id"]
        return {
            "JSON Name": f"{tup[0]}-{tup[1]}-{tup[2]}",
            "FakeNN JSON Name":f"{self.me.m}x{self.me.n}x{self.me.k}w{tup[0]}-{tup[1]}-{tup[2]}",
            "Original Name":f"{self.me.m}x{self.me.n}x{self.me.k}w{tup[0]}-{tup[1]}-{tup[2]}",
            "m Dim":tup[0],
            "Row Dim":tup[1],
            "Reduction Dim":tup[2],
            "M":self.me.m,
            "N":self.me.n,
            "K":self.me.k,
            "m":tup[0],
            "n":tup[1],
            "k":tup[2],
            "Space Needed in L1": d["Space Needed in L1"],
            "Weight Matrix Tile Size": d["Weight Matrix Tile Size"],
            "Space Remaining": d["Space Remaining"],
            "tileA": d["tileA"],
            "tileB": d["tileB"],
            "tileC": d["tileC"],
            "tileA_cc": d["tileA_cc"],
            "tileB_cc": d["tileB_cc"],
            "tileC_cc": d["tileC_cc"],
            "remainderTiles" : d["remainderTiles"],
            "Mpad":d["Mpad"],
            "Npad":d["Npad"],
            "Kpad":d["Kpad"],
        }

    # helper for converting to CSV
    # def annotationColumnNames(self):
    #     columns = [
    #         "JSON Name",
    #         "m Dim",
    #         "Row Dim",
    #         "Reduction Dim",
    #         "Space Needed in L1",
    #         "Weight Matrix Tile Size",
    #         "Space Remaining",
    #     ]
    #     return columns

    def convertOptionsToDF(self, dispatchNickName, options):
        flat = list(map(lambda ann: self.convertAnnotationToFlatDict(ann).values(), options))
        cols = self.convertAnnotationToFlatDict(options[0]).keys()
        df = pd.DataFrame(flat, columns=cols)        
        preferred_front_order = [
            "FakeNN JSON Name",
            "M",
            "N",
            "K",
            "m",
            "n",
            "k",
            "JSON Name",
        ]
        pfoSet = set(preferred_front_order)
        wofSet = set(set(df.columns).difference(pfoSet))
        preferred_order = preferred_front_order + list(wofSet)
        df = df[preferred_order]
        return df

    def exportOptionsToCSV(self, dispatchNickName, df):

        filename = f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_c_gen.csv"
        df.to_csv(
            filename,
            index=False,
        )
        print("\t", end="")
        print("TSG: wrote search space to")
        print("\t", end="")
        print(f"     {filename}")
        return filename
