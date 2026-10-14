from dataclasses import dataclass
import pprint

# Simulate counters from snitch matmul code
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
    dma_out_last_k_iter:bool

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
        self.dma_out_last_k_iter = self.dma_out_k == ts.k_rem_idx
        

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

# Keep track of memory compute overlap during double buffering
@dataclass
class Mem_Compute_Overlap:
    """Class for keeping track of memory compute overlap during double buffering"""
    compute: (int,int,int)
    load: (int,int,int)
    store : int
    iters : int    

    def incrementIters(self):
     self.iters = self.iters + 1

    def __init__(self, cmp, ld ,st):
        self.compute = cmp
        self.load = ld
        self.store = st
        self.iters = 1

# given a dictionary of Mem_compute_Overlap shapes, take their weighted average
def Avg_C_M_Overlap(d,verbose=False):
    if verbose:
        pprint.pprint(d,width=1)
    (compute,load,store),v = list(d.items())[0]
    comp = compute[0]*compute[1]*compute[2] # m * n * k
    mem = load[0]*load[2] + load[2]*load[1] + store # m*k + k*n + STORE (m*n)
    if comp != 0:
        raise Exception("First double buffer iter should NEVER compute!")
    total = 0
    weightSum = 0
    notOverlapped = 1
    for (compute,load,store),v in list(d.items())[1:]:
        comp = compute[0]*compute[1]*compute[2] # m * n * k
        mem = load[0]*load[2] + load[2]*load[1] + store # m*k + k*n + STORE (m*n)
        comp_weighted = comp *v.iters
        if mem==0:
            # if len(list(d.items()))-1 > 2:
            #     raise Exception("A double buffer iter ONLY COMPUTES but there are more than 2 iters!")
            notOverlapped = notOverlapped + 1
            continue
        cmRatio_weighted = comp_weighted/mem
        if cmRatio_weighted == 0: # only store
            if mem != store:
                Exception("A double buffer iter does NOT COMPUTE NOR STORE")
            notOverlapped = notOverlapped + 1
            continue
        total = total + cmRatio_weighted
        weightSum = weightSum + v.iters
        if verbose:
            print(f"{(compute,load,store)} with freq {v.iters}")
    if notOverlapped >= 4:
        raise Exception("Max of 3 non-overlapped double buffer phases violated!")
    return total / weightSum

# given the weighted average, compute the weighted variance
def Weighted_C_M_Overlap_Variance(weighted_avg,d,verbose=False):
    if verbose:
        pprint.pprint(d,width=1)
    (compute,load,store),v = list(d.items())[0]
    comp = compute[0]*compute[1]*compute[2] # m * n * k
    mem = load[0]*load[2] + load[2]*load[1] + store # m*k + k*n + STORE (m*n)
    if comp != 0:
        raise Exception("First double buffer iter should NEVER compute!")
    total = 0
    weightSum = 0
    notOverlapped = 1
    for (compute,load,store),v in list(d.items())[1:]:
        comp = compute[0]*compute[1]*compute[2] # m * n * k
        mem = load[0]*load[2] + load[2]*load[1] + store # m*k + k*n + STORE (m*n)
        comp_weighted = comp *v.iters
        if mem==0:
            # if len(list(d.items()))-1 > 2:
            #     raise Exception("A double buffer iter ONLY COMPUTES but there are more than 2 iters!")
            notOverlapped = notOverlapped + 1
            continue
        cmRatio_weighted = comp_weighted/mem
        if cmRatio_weighted == 0: # only store
            if mem != store:
                Exception("A double buffer iter does NOT COMPUTE NOR STORE")
            notOverlapped = notOverlapped + 1
            continue
        squaredDiff=v.iters*((cmRatio_weighted - weighted_avg)**2)
        total = total + squaredDiff
        weightSum = weightSum + v.iters
        if verbose:
            print(f"{(compute,load,store)} with freq {v.iters}")
    if notOverlapped >= 4:
        raise Exception("Max of 3 non-overlapped double buffer phases violated!")
    return total / weightSum