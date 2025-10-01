from dataclasses import dataclass, field
import pandas as pd
import sys
from itertools import product
from tile_sa.utils import MatmulInputs, TileSizes, roundUpToNearestMultipleOf
import pathlib

class TileSizeGenerator:
    def __init__(self, M_dim, N_dim, K_dim, dispatchName="",l1MemoryBytes = 100000):
        self.me = MatmulInputs(m=M_dim,n=N_dim, k=K_dim)
        self.l1MemoryBytes = l1MemoryBytes
        self.kernelName=dispatchName
    
    def dividesIntoM(self, num):
        return self.me.m % num == 0

    def dividesIntoN(self, num):
        return self.me.n % num == 0

    def dividesIntoK(self, num):
        return self.me.k % num == 0
    
    def mDimOptions(self):
        max = self.me.m
        min = 8 if self.me.m >= 8 else 1
        exhaustive = list(range(min, max + 1))
        if (self.me.k % 2) != 0:
            print(f"WARNING: M = {self.me.m} is NOT divisible by 2!")
        return exhaustive

    def nDimOptions(self):
        hardware_loop_body_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        byEight = list(map(lambda x: 8 * x, hardware_loop_body_options))
        max = self.me.n
        min = byEight[0]
        exhaustive = list(range(min, max + 1, 8))
        return exhaustive

    def kDimOptions(self):
        max = self.me.k
        min = 8 if self.me.k >= 8 else 1
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
        min = 8 if self.me.k >= 8 else 1
        step = 1
        max = self.me.k
        # we want about 40 tile sizes to pick from
        if self.me.k > 40:
            step = self.me.k // 40
        exhaustive = list(range(min, max + 1,step)) # not actually exhaustive...
        if (self.me.k % 2) != 0:
            print(f"WARNING: K = {self.me.k} is NOT divisible by 2!")
        return exhaustive
    
    def paddedMDimOptions(self):
        min = 8 if self.me.m >= 8 else 1
        step = 1
        max = self.me.m
        if self.me.m > 40:
            step = self.me.m // 40
        # we want about 40 tile sizes to pick from
        exhaustive = list(range(min, max + 1,step)) # not actually exhaustive...
        return exhaustive

    def validOptions(self, debug = False):
        # all possible values for m, n, and k
        little_m_options = self.mDimOptions()
        little_n_options = self.nDimOptions()
        little_k_options = self.kDimOptions()
        
        # filter for m's, n's and k's that divide evenly into M, N and K respectively
        little_m_no_pad = list(filter(lambda x: self.dividesIntoM(x), little_m_options))
        m_options = little_m_no_pad
        if len(little_m_no_pad) < 1: # prime M dimension
            m_options = self.paddedMDimOptions()
            if debug:
                print("\t TSG: ",end='')
                print(f'little m no pad is {little_m_no_pad} so we use padded options: {self.paddedMDimOptions()}',end="\n\n")
            
        little_n_no_pad = list(filter(lambda x: self.dividesIntoN(x), little_n_options))
        n_options = little_n_no_pad
        if len(little_n_no_pad) < 1: # prime N dimension, or not divisible by 8
            n_options = self.paddedNDimOptions()
            if debug:
                print("\t TSG: ",end='')
                print(f'little n no pad is {little_n_no_pad} so we use padded options: {self.paddedNDimOptions()}',end="\n\n")

        little_k_no_pad = list(filter(lambda x: self.dividesIntoK(x), little_k_options))
        k_options = little_k_no_pad
        if len(little_k_no_pad) < 1: # prime K dimension
            k_options = self.paddedKDimOptions()
            if debug:
                print("\t TSG: ",end='')
                print(f'little k no pad is {little_k_no_pad} so we use padded options: {self.paddedKDimOptions()}',end="\n\n")
        # halve k dim options for double buffering
        k_options = list(
        filter(lambda x: x <= (self.me.k // 2) + 1, k_options))
        options_as_triples = list(product(m_options, n_options, k_options))
        
        #options_as_triples =[(20,120,10),(20,40,10)]
        #options_as_triples =[(1,40,100)]
        annotated_options = list(map(lambda tup: self.annotateOptionWL1Usage(tup), options_as_triples))
        #print(annotated_options)
        # filter out tiling schemes that do not fit in L1
        # valid_options = list(
        #     filter(lambda tup: self.smallEnough(tup[0][0], tup[0][1],tup[0][2]), annotated_options)
        # )
       # print(tup)
        valid_options = list(
            filter(lambda tup: tup[3] >= 0, annotated_options)
        )
        if debug:
            print("\t TSG: options are ",end='')
            print(valid_options)
        if(len(valid_options)==0):
            raise Exception("Cannot find a valid tiling scheme!")
        return valid_options

  

    # matmul_transpose_b: A : MxK, B : NxK, C : MxN
    # elementwise addition: C : MxN, D : N = E: MxN
    def computeL1Usage(self, m, n, k, debug=False):
        tileSpace = 0
        weightMatTileSpace = 0
        total = 0
        tileA = m * k
        doubleBuff_B = self.me.n > n
        doubleBuff_A = (self.me.m > m) and (not doubleBuff_B)
        if doubleBuff_A and doubleBuff_B:
            raise Exception("Cannot double buffer both A and B operands")
        # if B operand is NOT tiled at L1 level, A will get double buffered instead.
        tileA2 = m * k if doubleBuff_A else 0
        tileB = n * k
        tileB2 = n * k if doubleBuff_B else 0
        entireC = roundUpToNearestMultipleOf(self.me.m,m) * roundUpToNearestMultipleOf(self.me.n, n)
        entireCExtra = 0
        if debug:
            print('\n')
            print(f'Linear Layer {self.me.m}-{self.me.n}-{self.me.k}:')
            print(f'Tiling Scheme {m}-{n}-{k}:')
            print(f'Allocate A tile: {m}x{k}')
            if(doubleBuff_A):
                print(f'Allocate A2 tile: {m}x{k}')
            print(f'Allocate B1 tile: {n}x{k}')
            if doubleBuff_B:
                print(f'Allocate B2 tile: {n}x{k}')
            print(f'Allocate C tile: {roundUpToNearestMultipleOf(self.me.m,m)}x{roundUpToNearestMultipleOf(self.me.n, n)}')
        # # TWO PADDING CASES:
        # 1) m  or n requires padding
        remainder_m = self.me.m % m
        if remainder_m != 0:
            entireCExtra = self.me.m * self.me.n
            if debug:
                print('Allocate Extra C tile: (m-padding)')
        remainder_n = self.me.n %n
        if remainder_n != 0:
        # then we allocate an extra unused buffer
            entireCExtra = self.me.m * self.me.n
            if debug:
                print('Allocate Extra C tile: (n-padding)')
        if debug:
            if remainder_m or remainder_n:
                print(f'Allocate Extra C tile: {self.me.m}x{self.me.n}')

        # 3) k requires padding
        # (no changes)
        # FINALLY, include elementwise addition...
        bias = roundUpToNearestMultipleOf(self.me.n, n)
        E = self.me.m * roundUpToNearestMultipleOf(self.me.n, n)
        if debug:
            if remainder_n:
                print(f'Allocate Bias Vector Tile: 1x{bias} (padded n)')
                print(f'Allocate Addition Output tile: {self.me.m}x{self.me.n} (padded n)')
            else:
                print(f'Allocate Bias Vector Tile: 1x{bias}')
                print(f'Allocate Addition Output tile: {self.me.m}x{self.me.n}')
        total = tileA + tileA2 + tileB + tileB2 + entireC + entireCExtra + bias + E
        if debug:
            print(f'total = {tileA} + {tileA2} + {tileB} + {tileB2} + {entireC} + {entireCExtra}+ {bias} + {E} = {total} elements or {total*8} bytes')
            if total*8 > self.l1MemoryBytes:
                print(f"which does NOT fit in L1 with {(total*8)-self.l1MemoryBytes} too many bytes!",end="\n\n")
            else:
                print(f"which fits in L1 with {self.l1MemoryBytes-(total*8)} bytes to spare",end="\n\n")
          
        weightMatTileSpace = tileB + tileB2
        tileSpace = tileA + tileA2 + tileB + tileB2
        totalUsage = total 
        return tileSpace, weightMatTileSpace, totalUsage
    
    
    # annotate a (row_dim, reduction_dim) pair with
    # total spaced used in L1
    # weight matrix tile size
    # space remaining, etc.
    def annotateOptionWL1Usage(self, tup):
        self.l1MemoryBytes = 100000
        tileSpace, weightMatTileSpace, totalUsage = self.computeL1Usage(tup[0],tup[1],tup[2])
        x= (
            tup,
            totalUsage*8,
            weightMatTileSpace*8,
            self.l1MemoryBytes - totalUsage*8
        )
        return x
    
    # annotate a (m_dim, row_dim, reduction_dim) triple with
    # quidditch load counting information
    # flatten tuple
    # matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)
    #
    #         outputVectorEltCount = N (AKA "row_dim")
    #         inputVectorEltCount = K (AKA "reduction dim")
    #
    def flattenThenAnnotateMore(self, ann):
        input = MatmulInputs(m=self.me.m,n=self.me.n, k=self.me.k)
        tiles = TileSizes(m=ann[0][0],n=ann[0][1], k=ann[0][2])
        flat = self.convertAnnotationToFlatTuple(ann)
        # loweringInfo = tsa.getLoweringInfoAnnotation(input, tiles)
        # print("\t",end='')
        # print(f"TSS: sa annotation is: {loweringInfo}")
        concatted = flat + (self.me.m,self.me.n,self.me.k) #+ loweringInfo
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

    def convertOptionsToDF(self, dispatchNickName, options):
        flat = list(map(lambda tup: self.flattenThenAnnotateMore(tup), options))
        cols =self.annotationColumnNames()+ ["M","N","K"]
        df = pd.DataFrame(flat, columns=cols)
        # add logistical info to data frame
        df["FakeNN JSON Name"]=df.apply(lambda y: f'{y["M"]}x{y["N"]}x{y["K"]}w{y["m Dim"]}-{y["Row Dim"]}-{y["Reduction Dim"]}' ,axis=1)
        df["m"]=df.apply(lambda y: y["m Dim"], axis=1)
        df["n"]=df.apply(lambda y: y["Row Dim"], axis=1)
        df["k"]=df.apply(lambda y: y["Reduction Dim"], axis=1)     
        preferred_front_order = ['FakeNN JSON Name','M','N','K','m','n','k','JSON Name']
        pfoSet = set(preferred_front_order)
        wofSet = set(set(df.columns).difference(pfoSet))
        preferred_order = preferred_front_order + list(wofSet)
        df = df[preferred_order]
        return df


    # export annotated options to CSV
    def exportOptionsToCSV(self, dispatchNickName, df):
        filename=f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_searchSpace.csv"
        df.to_csv(
            filename,
            index=False,
        )
        print("\t",end='')
        print(
            
            f"TSG: wrote search space to {filename}"
        )
        return filename
        

def main():
        args = sys.argv[1:]


if __name__ == "__main__":
    main()
