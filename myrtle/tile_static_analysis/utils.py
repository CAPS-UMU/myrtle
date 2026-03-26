from dataclasses import dataclass, field
#from xdsl.dialects.riscv_snitch import FrepOuter
from math import ceil, floor

class ClusterTile():
    def __init__(
        self,
        m_size,
        n_size,
        k_size,
        frequency,
        computeCoreTiles
    ):
        self.m_sz = m_size
        self.n_sz = n_size
        self.k_sz = k_size
        self.freq = frequency
        self.cctls = computeCoreTiles
    def __str__(self):
        return f"m_sz: {self.m_sz}, n_sz: {self.n_sz}, k_sz: {self.k_sz}, freq: {self.freq}, ccts"
    def metrics(self):
        info = {}
        area_a_prime = self.m_sz * self.k_sz
        area_b_prime = self.k_sz * self.n_sz
        area_c_prime = self.m_sz * self.n_sz
        info["L3 Loads"] = area_a_prime + area_b_prime
        info["L3 Stores"] = area_c_prime
        cc_metrics_sum = {}
        # sum over metrics from compute cores to get the compute core metric sum
        # add totals individually as different keys in the dictionary
@dataclass
class Sizes:
    m_sz: int
    n_sz: int
    k_sz: int

class ComputeCoreTile():
    def __init__(
        self,
        clusterTile,
        m_prime_size,
        frequency,
        UaJF
    ):
        self.l1Tile = Sizes(clusterTile.m_sz,clusterTile.n_sz,clusterTile.k_sz)
        self.m_prime_sz = m_prime_size
        self.freq = frequency
        self.u = UaJF
        self.n_u = clusterTile.n_sz / UaJF # we assume n_sz % u == 0
    def __str__(self):
        return f"m_prime_sz: {self.m_prime_sz}, freq: {self.freq}"

    def metrics(self):
        info = {}
        info["SSR Loads"] = 2 * self.u * self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["FMADDs"] = self.u * self.l1Tile.k_sz * self.n_u * self.m_prime_sz
        info["MULs"] = self.u * self.m_prime_sz * self.n_u
        info["HW Loops"] = self.m_prime_sz * self.n_u
        info["HW Loops / SSR Loads"] = info["HW Loops"]/info["SSR Loads"]
        # legacy values
        info["A Not Reused SSR Loads"]= self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["A SSR Reuse Loads"]=7 * self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["A SSR Start Reuse Loads"]= info["A Not Reused SSR Loads"]
        info["B SSR Loads"]=self.u * self.n_u * self.l1Tile.k_sz * self.m_prime_sz 
        return info

class TilingScheme():
    def __init__(
        self,
        M : int,
        N : int,
        K : int,
        m: int,
        n : int,
        k : int,
        u : int,
        p : int
    ):
        self.M = M
        self.N = N
        self.K = K
        self.m = m
        self.n = n
        self.k = k
        self.m_tiles = ceil(M/m)
        self.n_tiles = ceil(N/n)
        self.k_tiles = ceil(K/k)
        self.m_prime_tiles = p
        self.m_rem = M % m
        self.n_rem = N % n
        self.k_rem = K % k
        self.p = p
        self.u = u

    # returns cluster tiles used by this tiling scheme, 
    # and their associated compute core tiles wrapped in classes.
    def myClusterTiles(self):
        d = {}
        shapes = self.myClusterTileShapes()
        for (l1Shape, ccShapes) in shapes.items():
            ccTls = []
            for x in ccShapes: 
                ccTls.append(ComputeCoreTile(Sizes(x[0][0],x[0][1],x[0][2]),x[1],x[2],self.u))
            d[l1Shape] = ClusterTile(l1Shape[0],l1Shape[1],l1Shape[2],l1Shape[3],ccTls)
        return d


    # returns cluster tile shapes with a non-zero frequency
    def myClusterTileShapes(self):
        def count(D,d,d_size):
            if D % d == 0:
                d_count = D // d 
                d_rem_count = 0
            else:
                d_count = d - 1
                d_rem_count = 1
            return d_count if d == d_size else d_rem_count
        all = self.clusterTileShapes()
        mine = {}#[]
        for (m_sz,n_sz,k_sz) in all:
            m_iters = count(self.M,self.m,m_sz)
            n_iters = count(self.N,self.n,n_sz)
            k_iters = count(self.K,self.k,k_sz)
            freq = m_iters * n_iters * k_iters
            if freq != 0:
                mine[(m_sz,n_sz,k_sz,freq)]=self.myComputeCoreTileShapes((m_sz,n_sz,k_sz))
                #mine.append((m_sz,n_sz,k_sz,freq))
        return mine

    # returns compute core tile shapes with a non-zero frequency  
    def myComputeCoreTileShapes(self,cluster_tile_shape):
        m_size = cluster_tile_shape[0]
        m_prime = floor(m_size / self.p)
        m_hat = m_prime + 1
        rem = m_size % self.p
        if rem == 0:
            return [(cluster_tile_shape, m_prime, self.p)]
        else:
            return [(cluster_tile_shape, m_prime, self.p-rem),(cluster_tile_shape, m_hat,rem )]

        
        
    def clusterTileShapes(self):
        m_sizes = (self.m, self.m_rem)
        n_sizes = (self.n, self.n_rem) 
        k_sizes = (self.k, self.k_rem)
        all_imaginable = []
        for m in m_sizes:
             for n in n_sizes:
                for k in k_sizes:
                    all_imaginable.append((m,n,k))
        #assert len(all_imaginable) == 8 # debugging only
        return all_imaginable

    def computeCoreTileShapes(self,cluster_tile_shape):
        m_size = cluster_tile_shape[0]
        m_prime = floor(m_size / self.p)
        m_hat = m_prime + 1
        all_imaginable = [(cluster_tile_shape, m_prime),(cluster_tile_shape, m_hat)]
        #assert len(list(all_imaginable)) == 2 # debugging only
        return all_imaginable

def roundUpToNearestMultipleOf(num, row_dim):
  remainder = num % row_dim
  if (remainder != 0):
    return (num//row_dim) * row_dim + row_dim
  else:
    return num

@dataclass
class MatmulInputs:
    """Class for keeping track of input dimensions of a"""
    """matmul transpose with type `<MxK>, <NxK> -> <MxN>`"""
    """OR a regular matmul with type `<MxK>, <KxN> -> <MxN>`"""
    m: int = 1
    n: int = 1200
    k: int = 400

@dataclass
class TileSizes:
    """Class for keeping track of tiling in each dimension for a"""
    """matmul transpose with type `<MxK>, <NxK> -> <MxN>`"""
    """OR a regular matmul with type `<MxK>, <KxN> -> <MxN>`"""
    m: int = 1
    n : int = 40
    k : int = 100
    m_count: int = 1 # number of tiles of size m
    n_count : int = 1 # number of tiles of size n
    k_count: int = 1 # number of tiles of size k


@dataclass
class HardwareLoop:
    """Class for keeping track of hardware loop characteristics"""
    name: str = "frepOuter"#FrepOuter.name
    loop_iters: int = 1 # number of times loop executes
    body_size: int = 1  # number of instructions in body of the loop
    # right now, we assume all the instructions inside the body are the SAME
    # we assume each instruction takes 2 operands: a, b, -> c
    # we assume operands a and b using SSRs, and c does not


@dataclass
class EnclosingSCFLoop:
    """Class for keeping track of a potential loop surrounding the hardware loop"""
    name: str = "an enclosing loop"
    iters : int = 1   # number of times the enclosing loop executes; 
                      # if iters == 1, enclosing loop DNE.

@dataclass
class LoadCounts:
    """Class for keeping track of a regular vs streaming loads while processing one core tile"""
    regular_loads : int = 0
    a_operand_ssr_reuse_loads : int = 0
    a_operand_ssr_start_reuse_loads : int = 0
    b_operand_ssr_loads : int = 0
    not_resused_ssr_loads : int = 0
    total_ssr_loads : int = 0

def givenLoopsCreateLoadCount(hLoop : HardwareLoop, oLoop:EnclosingSCFLoop, ooLoop:EnclosingSCFLoop):
    initialized = LoadCounts()
    initialized.regular_loads = hLoop.body_size*oLoop.iters*ooLoop.iters
    initialized.a_operand_ssr_reuse_loads = (hLoop.body_size-1)*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    initialized.a_operand_ssr_start_reuse_loads = 1*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    initialized.b_operand_ssr_loads = hLoop.body_size*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    initialized.not_resused_ssr_loads = initialized.a_operand_ssr_start_reuse_loads + initialized.b_operand_ssr_loads
    initialized.total_ssr_loads = (hLoop.body_size*2)*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    assert initialized.total_ssr_loads == (initialized.a_operand_ssr_reuse_loads+initialized.a_operand_ssr_start_reuse_loads+initialized.b_operand_ssr_loads)
    return initialized
    
def multByInt(left, i : int):
    prod = LoadCounts()
    prod.regular_loads = i*left.regular_loads
    prod.a_operand_ssr_reuse_loads = i*left.a_operand_ssr_reuse_loads
    prod.a_operand_ssr_start_reuse_loads = i*left.a_operand_ssr_start_reuse_loads
    prod.b_operand_ssr_loads = i*left.b_operand_ssr_loads
    prod.not_resused_ssr_loads = i*left.not_resused_ssr_loads
    prod.total_ssr_loads= i*left.total_ssr_loads
    return prod

def sumLoadCounts(left, other):
    sum = LoadCounts()
    sum.regular_loads = left.regular_loads + other.regular_loads
    sum.a_operand_ssr_reuse_loads = left.a_operand_ssr_reuse_loads + other.a_operand_ssr_reuse_loads
    sum.a_operand_ssr_start_reuse_loads = left.a_operand_ssr_start_reuse_loads + other.a_operand_ssr_start_reuse_loads
    sum.b_operand_ssr_loads = left.b_operand_ssr_loads + other.b_operand_ssr_loads
    sum.not_resused_ssr_loads = left.not_resused_ssr_loads + other.not_resused_ssr_loads
    sum.total_ssr_loads= left.total_ssr_loads + other.total_ssr_loads
    return sum

@dataclass
class LoadCounter:
    a_ssr : int = 0
    a_ssr_reuse : int = 0
    b_ssr : int = 0
    c_regular : int = 0
    def mapMult(self, x:int):
        return LoadCounter(self.a_ssr*x,self.a_ssr_reuse*x,self.b_ssr*x,self.c_regular*x)

def add(l:LoadCounter, r:LoadCounter):
        return LoadCounter(l.a_ssr + r.a_ssr,l.a_ssr_reuse + r.a_ssr_reuse,l.b_ssr + r.b_ssr,l.c_regular + r.c_regular)

@dataclass
class ComputeCoreTiles:
    mPrime : TileSizes = field(default_factory=TileSizes)
    mHat : TileSizes = field(default_factory=TileSizes)