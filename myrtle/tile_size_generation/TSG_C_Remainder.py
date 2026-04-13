from dataclasses import dataclass, field
import pandas as pd
import sys
from itertools import product, chain
from tile_static_analysis.utils import MatmulInputs, TileSizes, roundUpToNearestMultipleOf
import pathlib
from tile_size_generation.TSG_C import TSG_C
from tile_size_generation.TileSizeGenerator import TileSizeGenerator

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

class TSG_C_Remainder(TileSizeGenerator):
    def hello(self):
        print("I am a tile size generator for the manual C backend, and I consider remainder tiles.")
    def __init__(
        self,
        M_dim,
        N_dim,
        K_dim,
        dispatchName="",
        l1MemoryBytes=100000,
        bank_size=1024,
        dualBuff=True,
    ):
        self.M=M_dim
        self.N=N_dim
        self.K=K_dim
        self.l1MemoryBytes = l1MemoryBytes
        self.kernelName = dispatchName
        self.bankSizeBytes = bank_size
        self.dualBuff = dualBuff

    def dividesIntoM(self, num):
        return self.M % num == 0

    def dividesIntoN(self, num):
        return self.N % num == 0

    def dividesIntoK(self, num):
        return self.K % num == 0   
    
#          
    def mDimOptions(self):
        max = self.M
        min = 8 if self.M >= 8 else 1
        exhaustive = list(range(min, max + 1))
        if (self.M % 2) != 0:
            print(f"WARNING: M = {self.M} is NOT divisible by 2!")
        return exhaustive
    
    def kDimOptions(self):
        max = self.K
        min = 8 if self.K >= 8 else 3 # min is 3 due to prologue and epilogue of HW Loop in assembly
        exhaustive = list(range(min, max + 1))
        if (self.N % 2) != 0:
            print(f"WARNING: K = {self.K} is NOT divisible by 2!")
        return exhaustive
    
    def remainderKGreaterThanTwo(self, num):
        rem_k = self.K % num
        return rem_k >= 3


    def nDimOptions(self):
        hardware_loop_body_options = [8]  # extend to 8,5 later
        max = self.N  # hides built-in max function
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
        if (self.N % 2) != 0:
            print(f"WARNING: N = {self.N} is NOT divisible by 2!")
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
                f"TSG: Cannot find a tile size that divides evenly into dimension M = {self.M}!"
            )
        little_n_no_pad = list(filter(lambda x: self.dividesIntoN(x), little_n_options))
        n_options = little_n_no_pad
        if len(little_n_no_pad) < 1:  # prime N dimension
            raise Exception(
                f"TSG: Cannot find a tile size that divides evenly into dimension N = {self.N}!"
            )
        little_k_no_pad = list(filter(lambda x: self.dividesIntoK(x), little_k_options))
        k_options = little_k_no_pad
        if len(little_k_no_pad) < 1:  # prime K dimension
            raise Exception(
                f"TSG: Cannot find a tile size that divides evenly into dimension K = {self.K}!"
            )
        
        # filter for m's, n's and k's that DO NOT divide evenly into M,N,K respectively
        little_m_pad = list(filter(lambda x: not self.dividesIntoM(x), little_m_options))
        if len(little_m_pad) < 1:  
            print(
                f"TSG: Cannot find a tile size that DOESN'T divide evenly into dimension M = {self.M}!"
            )
        little_n_pad = list(filter(lambda x: not self.dividesIntoN(x), little_n_options))
        if len(little_n_pad) < 1:  
            print(
                f"TSG: Cannot find a tile size that DOESN'T divide evenly into dimension N = {self.N}!"
            )
        little_k_pad = list(filter(lambda x: not self.dividesIntoK(x), little_k_options))
        if len(little_k_pad) < 1:  
            print(
                f"TSG: Cannot find a tile size that DOESN'T divide evenly into dimension K = {self.K}!"
            )
        else:
            # since we DO have some remainder tile options, make sure the remainder tile in the k dim is >= 3
            little_k_pad = list(filter(lambda x: self.remainderKGreaterThanTwo(x), little_k_options))

        # print(f"little_m_pad is {little_m_pad}")
        # print(f"little_m_options is {little_m_options}")
        # enumerate all remainder tile possibilities
        mnk = list(product(little_m_pad, little_n_pad, little_k_pad))     # M, N, K :)
        only_m = list(product(little_m_pad, n_options, k_options))        # only M
        only_mn = list(product(little_m_pad, little_n_pad, k_options))    # only M, N
        only_mk = list(product(little_m_pad, n_options, little_k_pad))    # only M, K
        only_n = list(product(m_options, little_n_pad, k_options))        # only N
        only_nk = list(product(m_options, little_n_pad, little_k_pad))    # only N, K
        only_k = list(product(m_options, n_options, little_k_pad))        # only K
        
        # remove duplicates
        options = set(mnk)
        options.update(only_m)
        options.update(only_mn)
        options.update(only_mk)
        options.update(only_n)
        options.update(only_nk)
        options.update(only_k)
        
        options_as_triples = list(options)
        return self.pruneForSizeConstraints(options_as_triples, debug)
        

    def pruneForSizeConstraints(self, options_as_triples, debug = False):
        options_as_dicts = list(map(lambda tup: {"id":tup}, options_as_triples))

        annotated_options = list(
            map(lambda d: self.annnotateRemainderTileStatus(d), options_as_dicts)
        )

        annotated_options = list(
            map(lambda d: self.annotateOptionWL1Usage(d), annotated_options)
        )

        # filter out tiling schemes that do not fit in L1
        valid_options_l1 = list(
            filter(lambda d: d["Space Remaining"] >= 0, annotated_options)
        )

        # filter out tile sizes that do not fit within 8 banks
        eb = 8 * self.bankSizeBytes # eb stands for "eight banks"
        valid_options_8_banks = list(
            filter(lambda d: max(d["tileA"],d["tileB"],d["tileC"]) <= eb, annotated_options)
        )
        # print(f"ignoring 8 bank constraint - 8 banks BTW takes up {self.bankSizeBytes} bytes")
        # valid_options_8_banks = valid_options_l1

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
        return valid_options_8_banks

    def annnotateRemainderTileStatus(self, d):
        m = d["id"][0]
        n = d["id"][1]
        k = d["id"][2]
        mRem = self.M % m
        nRem = self.N % n
        kRem = self.K % k
        # padding in M dim?
        if mRem == 0:
            mPadType = 0
        else:
            mPadType = "M"
            mPad = m - mRem
        # padding in N dim?
        if nRem == 0:
            nPadType = "0"
        else:
            nPadType = "N"
        # padding in K dim?
        if kRem == 0:
            kPadType = "0"
        else: 
            kPadType = "K"
        paddingType = f"{mPadType}{nPadType}{kPadType}"
        d.update({"remainderTiles": paddingType, "FakeNN JSON Name": f"{self.M}x{self.N}x{self.K}w{m}-{n}-{k}"})
        return d
    
    # flatten dictionary + add more information
    def convertAnnotationToFlatDict(self, d):
        tup = d["id"]
        return {
            "JSON Name": f"{tup[0]}-{tup[1]}-{tup[2]}",
            "FakeNN JSON Name":d["FakeNN JSON Name"],
            "m Dim":tup[0],
            "Row Dim":tup[1],
            "Reduction Dim":tup[2],
            "M":self.M,
            "N":self.N,
            "K":self.K,
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
        }

    # regular matmul: A : MxK, B : KxN, C : MxN
    # compute L1 usage measured in ELEMENT COUNT
    def computeL1Usage(self, d, debug=False):
        m = d["id"][0]
        n = d["id"][1]
        k = d["id"][2]
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
            print(f"Regular Matmul {self.M}-{self.N}-{self.K}:")
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

    # augment a dictionary {"id":(m,n,k)} to include
    # total spaced used in L1
    # weight matrix tile size
    # space remaining
    # measured in BYTES
    def annotateOptionWL1Usage(self, d):
        tileSpace, weightMatTileSpace, totalUsage, tileA, tileB, tileC, tileA_cc, tileB_cc, tileC_cc = self.computeL1Usage(
            d
        )
        d["Space Needed in L1"]= totalUsage * 8
        d["Weight Matrix Tile Size"]= weightMatTileSpace * 8
        d["Space Remaining"]= self.l1MemoryBytes - totalUsage * 8
        d["tileA"]= tileA * 8
        d["tileB"]= tileB * 8
        d["tileC"]= tileC * 8
        d["tileA_cc"]= tileA_cc *8
        d["tileB_cc"]= tileB_cc * 8
        d["tileC_cc"]= tileC_cc * 8
        return d

    # convert dictionary to simpler, more readable, annotated triple
    def dictToTuple(self, d):
        return (
            d["id"],
            d["Space Needed in L1"],
            d["Weight Matrix Tile Size"],
            d["Space Remaining"]
        )

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
            "remainderTiles",          
        ]
        pfoSet = set(preferred_front_order)
        wofSet = set(set(df.columns).difference(pfoSet))
        preferred_order = preferred_front_order + list(wofSet)
        df = df[preferred_order]
        return df

    def exportOptionsToCSV(self, dispatchNickName, df):
        filename = f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_c_rem_gen.csv"
        df.to_csv(
            filename,
            index=False,
        )
        # filenameSorted = f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_c_pad_ord_L1.csv"
        # sortedByL1=df.sort_values("Space Needed in L1", ascending=False)
        # sortedByL1.to_csv(filenameSorted,index=False)
        # filenameNoK = f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_padded_c_no_K.csv"
        # noK = df[df["Kpad"] == 0 ]
        # noK.to_csv(filenameNoK,index=False)
        # filenameOnlyK = f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_padded_c_only_K.csv"
        # noK = df[df["padding"] == "00K" ]
        # noK.to_csv(filenameOnlyK,index=False)
        print("\t", end="")
        print("TSG: wrote remainder-tile search space to")
        print("\t", end="")
        print(f"     {filename}")
        return filename

