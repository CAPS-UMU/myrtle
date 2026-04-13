import sys
import re
import subprocess
# Simulate counters from snitch matmul code so I can understand it

from dataclasses import dataclass

@dataclass
class PTS:
    """Class for keeping track of padded tiling scheme"""
    M: int
    N: int
    K: int
    m: int
    n: int
    k: int
    m_tiles: int
    n_tiles: int
    k_tiles: int
    m_rem: int # tile size for "padded" tile
    n_rem: int # tile size for "padded" tile
    k_rem: int # tile size for "padded" tile

    def __init__(self, M,N,K,m,n,k,unPadM,unPadN,unPadK):
        self.m_tiles = M // m if M % m == 0 else unPadM // m 
        self.n_tiles = N // n if N % n == 0 else unPadN // n 
        self.k_tiles = K // k if K % k == 0 else unPadK // k 
        self.M = M
        self.N = N
        self.K = K
        self.m = m
        self.n = n
        self.k = k
        self.m_rem = unPadM % m if unPadM%m != 0 else m
        self.n_rem = unPadN % n if unPadN%n != 0 else n
        self.k_rem = unPadK % k if unPadK%k != 0 else k
        self.m_rem_idx = M // m - 1
        self.n_rem_idx = N // n - 1
        self.k_rem_idx = K // k - 1


@dataclass
class DMA_in:
    """Class for keeping track of DMA_in counters"""
    dma_in_i: int
    dma_in_k: int
    dma_in_mn : int
    dma_in_n : int
    dma_in_m : int
    dma_in_m_abs: int

    def __init__(self, i, ts):
        self.dma_in_i = i
        self.dma_in_k = self.dma_in_i % ts.k_tiles
        self.dma_in_mn = self.dma_in_i // ts.k_tiles
        self.dma_in_n = self.dma_in_mn % ts.n_tiles
        self.dma_in_m = self.dma_in_mn // ts.n_tiles
        self.dma_in_m_abs = self.dma_in_m # we have one cluster, so no difference in absolute and regular m index
        self.m = ts.m_rem if self.dma_in_m == ts.m_rem_idx else ts.m
        self.n = ts.n_rem if self.dma_in_n == ts.n_rem_idx else ts.n
        self.k = ts.k_rem if self.dma_in_k == ts.k_rem_idx else ts.k

@dataclass
class DMA_out:
    """Class for keeping track of DMA_out counters"""
    dma_out_i: int
    dma_out_k: int
    dma_out_mn : int
    dma_out_n : int
    dma_out_m : int
    dma_out_m_abs: int

    def __init__(self, i, ts):
        self.dma_out_i = i-2 # because we double buffer
        self.dma_out_k = self.dma_out_i % ts.k_tiles      # cluster_k_tiles
        self.dma_out_mn = self.dma_out_i // ts.k_tiles    # cluster_k_tiles
        self.dma_out_n = self.dma_out_mn % ts.n_tiles     
        self.dma_out_m = self.dma_out_mn // ts.n_tiles
        self.dma_out_m_abs = self.dma_out_m # we have one cluster, so no difference in absolute and regular m index
        self.m = ts.m_rem if self.dma_out_m == ts.m_rem_idx else ts.m
        self.n = ts.n_rem if self.dma_out_n == ts.n_rem_idx else ts.n
        self.k = ts.k_rem if self.dma_out_k == ts.k_rem_idx else ts.k
        

@dataclass
class compute:
    """Class for keeping track of compute counters"""
    comp_i: int
    comp_k: int
    comp_mn : int
    comp_n : int
    comp_m : int
    comp_m_abs : int

    def __init__(self, i, ts):
        self.comp_i = i-1 # because we double buffer
        self.comp_k = self.comp_i % ts.k_tiles      # cluster_k_tiles
        self.comp_mn = self.comp_i // ts.k_tiles    # cluster_k_tiles
        self.comp_n = self.comp_mn % ts.n_tiles
        self.comp_m = self.comp_mn // ts.n_tiles
        self.comp_m_abs = self.comp_m # we have one cluster, so no difference in absolute and regular m index
        self.m = ts.m_rem if self.comp_m == ts.m_rem_idx else ts.m
        self.n = ts.n_rem if self.comp_n == ts.n_rem_idx else ts.n
        self.k = ts.k_rem if self.comp_k == ts.k_rem_idx else ts.k
# python counter-simulation.py 24x16x24w8-8-8
# clear;python counter-simulation.py 24x16x24w8-8-8
# clear;python counter-simulation.py 30x16x40w10-8-20
# clear;python counter-simulation.py 24x16x24w10-8-20 > mod.output; diff original.output mod.output
def main():
    #print("Usage example: python topTenFromMNK.py \"fileWInputSizes.txt\" \"outputFolderName\" all")
    tilingScheme = sys.argv[1]
    expNameRegex = re.compile(
                r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
            )
    M_str, N_str, K_str, m_str, n_str, k_str = expNameRegex.search(tilingScheme).groups()
    M = int(M_str)
    N = int(N_str)
    K = int(K_str)
    m = int(m_str)
    n = int(n_str)
    k = int(k_str)
    paddedM = m * (M // m) + m if (M % m) != 0 else M
    paddedN = n * (N // n) + n if (N % n) != 0 else N
    paddedK = k * (K // k) + k if (K % k) != 0 else K
  
    # originalMatmul(M,N,K,m,n,k)
    #print(f'passing in paddedMatMul({paddedM},{paddedN},{paddedK},{m},{n},{k},{M},{N},{K})')
    paddedMatmul(paddedM, paddedN, paddedK, m,n,k, M,N,K)
    
    # To overlap memory and compute, we employ a double buffering algorithm, 
    # which at each step performs at least one of three actions:
    #  - copy in new data
    #  - compute current data
    #  - write out computed data
    # These steps are performed until TWO conditions are met:
    # 1) all data has been COMPUTED 
    # 2) all data has been WRITTEN BACK to L3


    #print(f"bash many_gemms.sh {ss} compile no no no > ./{outputFolder}/compile-{basename}.txt;",file=compileScript)

    # generate actual values for 
    # # - the load_2d tiles calls, 
    # # - the store_2d tile calls, 
    # # - the gemm calls with the gemm_arg struct initialized
    # for each step, which actions are performed (and then which function call details?)

    # once I am clear on this, I should be able to modify the algorithm to select strips of "padded" tiles and treat them differently
    # maybe because of the double buffering prologue and epilogue, the zero computations are worth it? But then we have if-statements I think...
  
# def originalMatmul(M,N,K,m,n,k):
#     ts = PTS(M,N,K,m,n,k,M,N,K)
#     num_tiles = ts.m_tiles * ts.n_tiles * ts.k_tiles
#     # x represents the number double buffer steps needed to complete a matmul
#     x = num_tiles + 2
#     for i in range (0,x):
#         print("\tDouble Buffer Iteration "+str(i))
#         dma_in = DMA_in(i,ts)
#         dma_out = DMA_out(i,ts)
#         compu = compute(i,ts)
#         if(dma_out.dma_out_i >= 0):
#             print("\tDMA OUT ",end='')
#             buff_idx = dma_out.dma_out_mn % 2 # switch C buffers
#             print("\t",end='')
#             print(f"sntr_dma_store_2d_tile(L3 C ptr, SPM C buff[{buff_idx}],",end='')
#             print(f"{dma_out.dma_out_m_abs},{dma_out.dma_out_n},",end='')
#             print("\t",f"{ts.m},{ts.n}, C flat size, prec)")
#         else:
#             print("\tDMA OUT (skipped)")
#         if (dma_in.dma_in_i < num_tiles):
#             print("\tDMA IN")
#             buff_idx = dma_in.dma_in_i % 2 # switch A, B buffers
#             c_buff_idx = dma_in.dma_in_mn % 2 # switch C buffers
#             # we always load a new A and B tile when DMA In action is enabled
#             # load A
#             print("\t\t",end='')
#             print(f"sntr_dma_load_2d_tile(SPM A buff[{buff_idx}], L3 A ptr, ",end='')
#             print(f"{dma_in.dma_in_m_abs},{dma_in.dma_in_k},",end='')
#             print("\t",f"{ts.m},{ts.k}, A flat size, prec)")
#             # load B
#             print("\t\t",end='')
#             print(f"sntr_dma_load_2d_tile(SPM B buff[{buff_idx}], L3 B ptr, ",end='')
#             print(f"{dma_in.dma_in_k},{dma_in.dma_in_n},",end='')
#             print("\t",f"{ts.k},{ts.n}, B flat size, prec)")
#             if(dma_in.dma_in_k == 0): # load C (only on first k iteration)
#                 print("\t\t",end='')
#                 print(f"sntr_dma_load_2d_tile(SPM C buff[{c_buff_idx}], L3 C ptr, ",end='')
#                 print(f"{dma_in.dma_in_m_abs},{dma_in.dma_in_n},",end='')
#                 print("\t",f"{ts.m},{ts.n}, C flat size, prec)")
#         else:
#             print("\tDMA IN (skipped)")
#         if((compu.comp_i >= 0) and (compu.comp_i < num_tiles)):
#             print("\tCOMPUTE", end="")
#             buff_idx = compu.comp_i % 2 # switch A, B buffers
#             c_buff_idx = compu.comp_mn % 2 # switch C buffers
#             print("\t",end='')
#             print("\t",f"sc_st_gemm(a=SPM A buff[{buff_idx}], lda = {ts.k}, b=SPM B buff[{buff_idx}], ldb = {ts.n}, c=SPM C buff[{c_buff_idx}], ldc = {ts.n})")
#         else:
#             print("\tCOMPUTE (skipped)")
  

#         print("")
#     print("")
#     for i in range(0,x):
#         print(DMA_in(i,ts))
#     print("")
#     for i in range(0,x):
#         dma_out = DMA_out(i,ts)
#         if(dma_out.dma_out_i >= 0):
#             print(dma_out)
#     print("")
#     for i in range(0,x):
#         compu = compute(i,ts)
#         if((compu.comp_i >= 0) and (compu.comp_i < num_tiles)):
#             print(compu)

def paddedMatmul(M,N,K,m,n,k,unPadM,unPadN,unPadK):
    ts = PTS(M,N,K,m,n,k,unPadM,unPadN,unPadK)
    num_tiles = ts.m_tiles * ts.n_tiles * ts.k_tiles
    # x represents the number of double buffer steps needed to complete a matmul
    x = num_tiles + 2
    for i in range (0,x):
        print("\tDouble Buffer Iteration "+str(i))
        dma_in = DMA_in(i,ts)
        dma_out = DMA_out(i,ts)
        compu = compute(i,ts)

        # DMA OUT
        if(dma_out.dma_out_i >= 0):
            print("\tDMA OUT ",end='')
            buff_idx = dma_out.dma_out_mn % 2 # switch C buffers
            print("\t",end='')
            print(f"sntr_dma_store_2d_tile(L3 C ptr, SPM C buff[{buff_idx}],",end='')
            print(f"{dma_out.dma_out_m_abs},{dma_out.dma_out_n},",end='')
            print("\t",f"{dma_out.m},{dma_out.n}, C flat size, prec)")
        else:
            print("\tDMA OUT (skipped)")

        # DMA IN
        if (dma_in.dma_in_i < num_tiles):
            print("\tDMA IN")
            buff_idx = dma_in.dma_in_i % 2 # switch A, B buffers
            c_buff_idx = dma_in.dma_in_mn % 2 # switch C buffers
            # we always load a new A and B tile when DMA In action is enabled
            # load A
            print("\t\t",end='')
            print(f"sntr_dma_load_2d_tile(SPM A buff[{buff_idx}], L3 A ptr, ",end='')
            print(f"{dma_in.dma_in_m_abs},{dma_in.dma_in_k},",end='')
            print("\t",f"{dma_in.m},{dma_in.k}, A flat size, prec)")
            # load B
            print("\t\t",end='')
            print(f"sntr_dma_load_2d_tile(SPM B buff[{buff_idx}], L3 B ptr, ",end='')
            print(f"{dma_in.dma_in_k},{dma_in.dma_in_n},",end='')
            print("\t",f"{dma_in.k},{dma_in.n}, B flat size, prec)")
            # when beta = 0, we never load C
            # if(dma_in.dma_in_k == 0): # load C (only on first k iteration)
            #     print("\t\t",end='')
            #     print(f"sntr_dma_load_2d_tile(SPM C buff[{c_buff_idx}], L3 C ptr, ",end='')
            #     print(f"{dma_in.dma_in_m_abs},{dma_in.dma_in_n},",end='')
            #     print("\t",f"{dma_in.m},{dma_in.n}, C flat size, prec)")
        else:
            print("\tDMA IN (skipped)")

        # COMPUTE
        if((compu.comp_i >= 0) and (compu.comp_i < num_tiles)):
            print("\tCOMPUTE", end="")
            buff_idx = compu.comp_i % 2 # switch A, B buffers
            c_buff_idx = compu.comp_mn % 2 # switch C buffers
            print("\t",end='')
            print("\t",f"sc_st_gemm(a=SPM A buff[{buff_idx}], lda = {compu.k}, b=SPM B buff[{buff_idx}], ldb = {compu.n}, c=SPM C buff[{c_buff_idx}], ldc = {compu.n})")
        else:
            print("\tCOMPUTE (skipped)")
  

        print("")
    # print("")
    # for i in range(0,x):
    #     print(DMA_in(i,ts))
    # print("")
    # for i in range(0,x):
    #     dma_out = DMA_out(i,ts)
    #     if(dma_out.dma_out_i >= 0):
    #         print(dma_out)
    # print("")
    # for i in range(0,x):
    #     compu = compute(i,ts)
    #     if((compu.comp_i >= 0) and (compu.comp_i < num_tiles)):
    #         print(compu)

if __name__ == "__main__":
    main()