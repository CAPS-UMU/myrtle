from dataclasses import dataclass, field
import pandas as pd
import sys
from itertools import product
from tile_sa.utils import MatmulInputs, TileSizes, roundUpToNearestMultipleOf


class TileSizeGenerator:
    def __init__(self, M_dim, N_dim, K_dim, dispatchName="",l1MemoryBytes = 100000):
        self.me = MatmulInputs(m=M_dim,n=N_dim, k=K_dim)
        self.l1MemoryBytes = l1MemoryBytes
    
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

    def weightMatTileSize(self, row_dim, reduction_dim):
        return row_dim * reduction_dim

    def spaceForTiles(self, m_dim, row_dim, reduction_dim):
        # ignore output matrix tiles (for now, entire output always in L1)
        # space in element count
        inputMatTile = m_dim * reduction_dim
        weightMatTiles = 2 * self.weightMatTileSize(row_dim, reduction_dim)
        space = inputMatTile + weightMatTiles
        # space in  bytes
        spaceInBytes = space * 8  # number of elements * 8 bytes per element
        return spaceInBytes

    def spaceRemaining(self, m_dim, row_dim, reduction_dim):
        self.l1MemoryBytes = 100000
        outputMatMul_m = roundUpToNearestMultipleOf(self.me.m, m_dim)
        outputMatMul_n = roundUpToNearestMultipleOf(self.me.n, row_dim)
        outputMatMul = outputMatMul_m * outputMatMul_n * 8
        print(f'output of matmul will be {outputMatMul_m} by {outputMatMul_n}')
        print(f'input mat tile of {m_dim} x {reduction_dim}')
        print(f'weight mat tile of size {row_dim}x{reduction_dim}')
        print(f'weight mat tile of size {row_dim}x{reduction_dim}')
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
            self.l1MemoryBytes
            - outputMatMul
            - outputElemAdd
            - inputElemAdd
            - self.spaceForTiles(m_dim, row_dim, reduction_dim)
        )
        return remaining

    def computeLargestL1Tiles(self, debug=False):
        def exhaustiveDescending(max):
            min = 1 # (obviously)
            step = 1 # we want to be exhaustive            
            return list(reversed(list(range(min, max + 1, step))))
        m  = exhaustiveDescending(self.me.m)
        n = list(reversed(self.paddedNDimOptions()))
        k = exhaustiveDescending(self.me.k)
        print("HELP D\':")
        def L1Usage(m,n,k,debug = False):
            print()
            print(f'L1 Usage for {m}-{n}-{k}:')
            A_tile = m*k
            B_tile = n*k
            B2_tile = n*k
            C_tile = m*n
            bias_tile = n
            E_tile = m*n
            total = A_tile + B_tile + B2_tile + C_tile + bias_tile + E_tile
            print(f'Allocate A tile: {m}x{k} = {A_tile} elts')
            print(f'Allocate B1 tile: {n}x{k} = {B_tile} elts')
            print(f'Allocate B2 tile: {n}x{k} = {B2_tile} elts')
            print(f'Allocate C tile: {m}x{n} = {C_tile} elts')
            print(f'Allocate Bias Vector Tile: 1x{n} = {bias_tile} elts')
            print(f'Allocate Addition Output tile: {m}x{n} = {E_tile} elts')
            print(f'Total = {A_tile} + {B_tile} + {B2_tile} + {C_tile} + {bias_tile} + {E_tile} = {total} elts = {total*8} bytes')
            if total*8 > self.l1MemoryBytes:
                print(f"which does NOT fit in L1 with {(total*8)-self.l1MemoryBytes} too many bytes!")
            else:
                print(f"which fits in L1 with {self.l1MemoryBytes-(total*8)} bytes to spare")
            return total*8

        def fixedL1Usage(M,N,K,debug=False):
            if debug:
                print()
                print(f'FIXED L1 Usage for {M}-{N}-{K}:')
            C_tile = M*N
            E_tile = M*N
            bias_tile = N
            total = C_tile + E_tile + bias_tile
            # (M*N)+(M*N)+N < 100000
            if debug:
                print(f'Total = {C_tile} + {bias_tile} + {E_tile} = {total} elts = {total*8} bytes')
                if total*8 > self.l1MemoryBytes:
                    print(f"which does NOT fit in L1 with {(total*8)-self.l1MemoryBytes} too many bytes!")
                else:
                    print(f"which fits in L1 with {self.l1MemoryBytes-(total*8)} bytes to spare")
            return total*8
        
        M = exhaustiveDescending(self.l1MemoryBytes//8)
        N = exhaustiveDescending(self.l1MemoryBytes//8)
        K = exhaustiveDescending(self.l1MemoryBytes//8)

        options_as_triples = product(M, N, K) 
        usage = 0 
        usage_name=""
        dict = {"best":[]}
        # for triple in options_as_triples:
        #     print(triple)
        
        #     for (m , n, k) in triple:
        #         (my_usage, fits) = fixedL1Usage(m,n,k)
        #         if fits and my_usage > usage:
        #             usage = my_usage
        #             usage_name=f"{m}-{n}-{k}"
        #             #dict["best"].append((f"{m}-{n}-{k}",my_usage))
        # print(dict)

        # options_l1_usage = list(map(lambda tup: fixedL1Usage(tup), options_as_triples))
        # options_fit = list(filter(lambda tup: tup[1], options_l1_usage))
        # print(options_fit)
        fixedL1Usage(1,1,1,debug=True)
        fixedL1Usage(m[0],1,1,debug=True)
        fixedL1Usage(1,n[0],1,debug=True)
        fixedL1Usage(1,1,k[0],debug=True)
        # L1Usage(8,24,8,debug=True)
        # print(exhaustiveDescending(self.me.n),end="\n\n")
        # print(self.nDimOptions(),end="\n\n")
        # print(self.paddedNDimOptions(),end="\n\n")
        # print(n)
        # print(f"max m (no padding) is {m[0]}")
        # print(f"max n is {n[0]}")
        # print(f"max k (no padding) is {k[0]}")

    # def smallEnough(self,m_dim, row_dim, red_dim):
    #     return self.spaceRemaining(m_dim, row_dim, red_dim) > 0

    # matmul_transpose_b: A : MxK, B : NxK, C : MxN
    # elementwise addition: C : MxN, D : N = E: MxN
    def computeL1Usage(self, m, n, k, debug=False):
        tileSpace = 0
        weightMatTileSpace = 0
        total = 0
        tileA = m * k
        tileB = n * k
        tileB2 = n * k
        entireC = roundUpToNearestMultipleOf(self.me.m,m) * roundUpToNearestMultipleOf(self.me.n, n)
        entireCExtra = 0
        if debug:
            print('\n')
            print(f'Linear Layer {self.me.m}-{self.me.n}-{self.me.k}:')
            print(f'Tiling Scheme {m}-{n}-{k}:')
            print(f'Allocate A tile: {m}x{k}')
            print(f'Allocate B1 tile: {n}x{k}')
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
        total = tileA + tileB + tileB2 + entireC + entireCExtra + bias + E
        if debug:
            print(f'total = {tileA} + {tileB} + {tileB2} + {entireC} + {entireCExtra}+ {bias} + {E} = {total} elements or {total*8} bytes')
            if total*8 > self.l1MemoryBytes:
                print(f"which does NOT fit in L1 with {(total*8)-self.l1MemoryBytes} too many bytes!",end="\n\n")
            else:
                print(f"which fits in L1 with {self.l1MemoryBytes-(total*8)} bytes to spare",end="\n\n")
          
        weightMatTileSpace = tileB + tileB2
        tileSpace = tileA + tileB + tileB2
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
        filename=f"./{dispatchNickName}_searchSpace.csv"
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
