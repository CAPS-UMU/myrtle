from dataclasses import dataclass

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

# Keep track of memory compute overlap during double buffering
@dataclass
class Mem_Compute_Overlap:
    """Class for keeping track of memory compute overlap during double buffering"""
    compute: int
    load: int
    store : int
    iters : int    

    def incrementIters(self):
     self.iters = self.iters + 1

    def __init__(self, cmp, ld ,st):
        self.compute = cmp
        self.load = ld
        self.store = st
        self.iters = 1
