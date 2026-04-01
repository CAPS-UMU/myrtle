from dataclasses import dataclass
from math import ceil, floor

def applyFuncToDictPair(f, left, right):
    res = {}
    for k in left.keys():
        res[k] = f(left[k],right[k])
    return res

def applyFuncToDict(f, dict):
    res = {}
    for k in dict.keys():
        res[k] = f(dict[k])
    return res

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
        return f"Compute Core Tile: my l1 tile: {self.l1Tile}; m_prime_sz: {self.m_prime_sz}, freq: {self.freq}"

    def metrics(self):
        info = {}
        info["SSR Loads"] = 2 * self.u * self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        if info["SSR Loads"] == 0:
            print("HELP")
            print(self)
            print("2 * self.u * self.n_u * self.l1Tile.k_sz * self.m_prime_sz")
            print(f"2 * {self.u} * {self.n_u} * {self.l1Tile.k_sz} * {self.m_prime_sz}")
        info["FMADDs"] = self.u * self.l1Tile.k_sz * self.n_u * self.m_prime_sz
        info["MULs"] = self.u * self.m_prime_sz * self.n_u
        info["HW Loops"] = self.m_prime_sz * self.n_u
        info["myRegPerStream"] =self.l1Tile.m_sz*self.l1Tile.n_sz / (128*self.l1Tile.k_sz)
        #df["n"] * df["m"] / (128.0 * df["k"])
        info["HW Loops / SSR Loads"] = self.m_prime_sz * self.l1Tile.n_sz / (16 * self.l1Tile.k_sz)#info["HW Loops"]/info["SSR Loads"]
      
        # legacy values (corrected to not use k-2 iters)
        info["A Not Reused SSR Loads"]= self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["A SSR Reuse Loads"]=7 * self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["A SSR Start Reuse Loads"]= info["A Not Reused SSR Loads"]
        info["B SSR Loads"]= self.u * self.n_u * self.l1Tile.k_sz * self.m_prime_sz 
        info["A SSR Loads"] =  info["A Not Reused SSR Loads"] + info["A SSR Reuse Loads"]
        return info
    def emptyMetrics():
        info = {}
        info["SSR Loads"] = 0
        info["FMADDs"] = 0
        info["MULs"] = 0
        info["HW Loops"] = 0
        info["HW Loops / SSR Loads"] = 0
        info["myRegPerStream"] = 0
        #info["8 * HW Loops"]=0
        # legacy values
        info["A Not Reused SSR Loads"]= 0
        info["A SSR Reuse Loads"]=0
        info["A SSR Start Reuse Loads"]= 0
        info["B SSR Loads"]=0 
        info["A SSR Loads"] = 0
        return info

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
        str = f"Cluster Tile: m_sz: {self.m_sz}, n_sz: {self.n_sz}, k_sz: {self.k_sz}, freq: {self.freq}"
        str = str + "\n\t\t my compute core tiles are..."
        for c in self.cctls:
            str = str + "\n\t\t\t" + c.__str__()
        return str
    
    def metrics(self):
        info = {}
        area_a_prime = self.m_sz * self.k_sz
        area_b_prime = self.k_sz * self.n_sz
        area_c_prime = self.m_sz * self.n_sz
        info["L3 Loads"] = area_a_prime + area_b_prime
        info["L3 Stores"] = area_c_prime
        # each cluster tile has either 1 or 2 compute core tile shapes
        if len(self.cctls) == 2:
            left = self.cctls[0]
            right = self.cctls[1]
            # scale all metrics by number of times its tile shape is used
            scaledLeft = applyFuncToDict(lambda x: left.freq * x,left.metrics())
            scaledRight = applyFuncToDict(lambda x: right.freq * x,right.metrics())
            # scaledLeft["myRegPerStream"]=left.metrics()["myRegPerStream"]
            # scaledRight["myRegPerStream"]=right.metrics()["myRegPerStream"]
            # take the sum of the scaled metrics
            cc_metrics_sum = applyFuncToDictPair(lambda x, y: x + y,scaledLeft,scaledRight)
            # we don't want to scale the ratio
            # val = left.metrics()["myRegPerStream"]
            # print(f"myRegsPerStream is {val}")
            # we don't want to sum our regPerStream:
           # cc_metrics_sum["myRegPerStream"] = left.metrics()["myRegPerStream"] # only keep m' regPerStream val
            # val = cc_metrics_sum["myRegPerStream"]
            # print(f"NOW myRegsPerStream is {val}")
            #scaledLeft["HW Loops / SSR Loads"]=left.metrics()["HW Loops / SSR Loads"]
            #scaledRight["HW Loops / SSR Loads"]=right.metrics()["HW Loops / SSR Loads"]
        else:
            only = self.cctls[0]    
            cc_metrics_sum = applyFuncToDict(lambda x: only.freq * x,only.metrics())
            # we don't want to scale the ratio
           # cc_metrics_sum["myRegPerStream"]=only.metrics()["myRegPerStream"]
            #cc_metrics_sum["HW Loops / SSR Loads"]=only.metrics()["HW Loops / SSR Loads"]
            
        info.update(cc_metrics_sum)
     #   print(f"info right before I return from metrics is {info}")
        return info
    def emptyMetrics():
        info = {}
        info["L3 Loads"] = 0
        info["L3 Stores"] = 0
        info.update(ComputeCoreTile.emptyMetrics())
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
        p : int,
        remainderTiles : str
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
        self.remainderTiles = remainderTiles

    def __str__(self):
        str = f"Tiling Scheme: {self.M}x{self.N}x{self.K}w{self.m}-{self.n}-{self.k} and remainder tiles {self.remainderTiles}"
        str = str + "\n\tmy cluster tiles are..."
        cts = self.myClusterTiles()
        for k in cts.keys():
            str = str + "\n\t" + cts[k].__str__()
            print(cts[k])
        return str
    def remainderTiles(self):
        return not ((self.m_rem==0) and (self.n_rem==0) and (self.k_rem==0))
    
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
        mine = {}
        for (m_sz,n_sz,k_sz) in all:
            m_iters = count(self.M,self.m,m_sz)
            n_iters = count(self.N,self.n,n_sz)
            k_iters = count(self.K,self.k,k_sz)
            freq = m_iters * n_iters * k_iters
            if freq != 0:
                mine[(m_sz,n_sz,k_sz,freq)]=self.myComputeCoreTileShapes((m_sz,n_sz,k_sz))
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
            if m_prime == 0:
                return [(cluster_tile_shape, m_hat,rem )]
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