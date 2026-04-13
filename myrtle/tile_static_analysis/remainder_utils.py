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
        info["SSR Loads"] = self.m_prime_sz * self.n_u *self.u *  self.l1Tile.k_sz *2
        info["HW Loops"] = self.m_prime_sz * self.n_u
        info["FMADDs"] = self.m_prime_sz * self.l1Tile.n_sz * (self.l1Tile.k_sz * -1)
        info["MULs"] = self.u * self.m_prime_sz * self.n_u
        info["FMADDsMULs"] = info["FMADDs"] + info["MULs"]
        info["SSR Loads per HW Loop"] = self.u * self.l1Tile.k_sz * 2
        info["HW Loops / SSR Loads per HW Loop"] = self.m_prime_sz * self.l1Tile.n_sz / (128 * self.l1Tile.k_sz)#info["HW Loops"]/info["SSR Loads"]
        info["myRegPerStream"] =self.l1Tile.m_sz*self.l1Tile.n_sz / (128*self.l1Tile.k_sz) # old reg per stream metric that didn't use CC tile shape
        info["HW Loops / SSR Loads"] = self.m_prime_sz * self.l1Tile.n_sz / (16 * self.l1Tile.k_sz)# deprecated
        
        # legacy values (corrected to not use k-2 iters)
        info["A Not Reused SSR Loads"]= self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["A SSR Reuse Loads"]=7 * self.n_u * self.l1Tile.k_sz * self.m_prime_sz
        info["A SSR Start Reuse Loads"]= info["A Not Reused SSR Loads"]
        info["B SSR Loads"]= self.u * self.n_u * self.l1Tile.k_sz * self.m_prime_sz 
        info["A SSR Loads"] =  info["A Not Reused SSR Loads"] + info["A SSR Reuse Loads"]
        # TODO: redo these using m_sz, n_sz, k_sz instead of l1 tile sizes!
        info["Reused / Total SSR Loads"] = info["A SSR Reuse Loads"] / (info["A SSR Loads"] + info["B SSR Loads"])
        info["Core : A SSR Reuse Loads"] = 1/info["A SSR Reuse Loads"]
        return info
    
    # return a dictionary of compute core tile metrics, with every value set to zero
    def emptyMetrics():
        info = {}
        info["SSR Loads"] = 0
        info["HW Loops"] = 0
        info["FMADDs"] = 0
        info["MULs"] = 0
        info["FMADDsMULs"] = info["FMADDs"] + info["MULs"]
        info["SSR Loads per HW Loop"] = 0
        info["HW Loops / SSR Loads per HW Loop"] =0
        info["myRegPerStream"] =0
        info["HW Loops / SSR Loads"] = 0
        # legacy values
        info["A Not Reused SSR Loads"]= 0
        info["A SSR Reuse Loads"]=0
        info["A SSR Start Reuse Loads"]= 0
        info["B SSR Loads"]=0 
        info["A SSR Loads"] = 0
        info["Core : A SSR Reuse Loads"] = 0
        info["Reused / Total SSR Loads"] = 0
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
            # take the sum of the scaled metrics
            cc_metrics_sum = applyFuncToDictPair(lambda x, y: x + y,scaledLeft,scaledRight)
        else:
            only = self.cctls[0]    
            cc_metrics_sum = applyFuncToDict(lambda x: only.freq * x,only.metrics())
        info.update(cc_metrics_sum)
        
        return info
    
    # return a dictionary of cluster tile metrics, with every value set to zero
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
        
        self.m_rem = M % m
        self.n_rem = N % n
        self.k_rem = K % k
        self.p = p
        self.u = u
        self.remainderTiles = remainderTiles

    def __str__(self):
        str = f"Tiling Scheme: {self.M}x{self.N}x{self.K}w{self.m}-{self.n}-{self.k} and remainder tiles {self.remainderTiles}"
        str = str + "\n\tmy cluster tiles are..."
        cts = self.validClusterTiles()
        for k in cts.keys():
            str = str + "\n\t" + cts[k].__str__()
            print(cts[k])
        return str
    def remainderTiles(self):
        return not ((self.m_rem==0) and (self.n_rem==0) and (self.k_rem==0))
    
    # returns all cluster tiles used by this tiling scheme, 
    # including their associated compute core tiles, all wrapped in classes.
    # tiles are returned in the form of a key-value pair map
    # key: 3-tuple representing CL tile's m_sz, n_sz, and k_sz and frequency
    # value: A Cluster tile object containing m_sz, n_sz, k_sz, frequency, and a list of its CC tile objects
    def validClusterTiles(self):
        d = {}
        shapes = self.validClusterTileShapes()
        for (l1Shape, ccShapes) in shapes.items():
            ccTls = []
            for x in ccShapes: 
                ccTls.append(ComputeCoreTile(Sizes(x[0][0],x[0][1],x[0][2]),x[1],x[2],self.u))
            d[l1Shape] = ClusterTile(l1Shape[0],l1Shape[1],l1Shape[2],l1Shape[3],ccTls)
        return d


    # Given this tiling scheme, return its
    # cluster tile shapes with a non-zero frequency.
    # CL tile shapes are returned in the form of a key-value pair map
    # key: a cluster tile shape represented as a 4-tuple (m_sz,n_sz,k_sz,freq)
    # value: list of valid compute core tile shapes (a list of 3-tuples), ex [(cl_tile, m'_size, freq)].
    def validClusterTileShapes(self):
        def dimFreq(D,d,d_size): # how many times do we tile (cluster level) in dimension d?
            if D % d == 0:
                d_count = D // d 
                d_rem_count = 0
            else:
                d_count = D // d 
                d_rem_count = 1
            return d_count if d == d_size else d_rem_count
        potentialShapes = self.potentialClusterTileShapes()
        valid = {}
        for (m_sz,n_sz,k_sz) in potentialShapes:
            m_iters = dimFreq(self.M,self.m,m_sz)
            n_iters = dimFreq(self.N,self.n,n_sz)
            k_iters = dimFreq(self.K,self.k,k_sz)
            freq = m_iters * n_iters * k_iters
           # print(f"{m_sz} {n_sz} {k_sz} has num_tiles {m_iters} * {n_iters} * {k_iters} = {freq}")
            if freq != 0:
                valid[(m_sz,n_sz,k_sz,freq)]=self.validComputeCoreTileShapes((m_sz,n_sz,k_sz))
        return valid
    
    # Given a cluster tile shape
    # returns its compute core tile shapes with a non-zero frequency. 
    # CC tile shapes are returned as a list of 3-tuples in the form (cl_tile, m'_size, freq).
    def validComputeCoreTileShapes(self,cluster_tile_shape):
        m_size = cluster_tile_shape[0]
        m_prime = floor(m_size / self.p)
        m_hat = m_prime + 1
        rem = m_size % self.p
        if rem == 0:
            return [(cluster_tile_shape, m_prime, self.p)]
        else:
            if m_prime == 0: # this is the edge case when m_size = m_rem and m_rem < 8
                return [(cluster_tile_shape, m_hat,rem )]
            else:
                return [(cluster_tile_shape, m_prime, self.p-rem),(cluster_tile_shape, m_hat,rem )]

    # Given this tiling scheme's input dimensions and tile sizes, 
    # return all potential cluster tile shapes.
    # CL tile shapes returned as a list of tuples of the form (m_sz,n_sz,k_sz).
    def potentialClusterTileShapes(self):
        m_sizes = (self.m, self.m_rem)
        n_sizes = (self.n, self.n_rem) 
        k_sizes = (self.k, self.k_rem)
        all_imaginable = []
        for m in m_sizes:
             for n in n_sizes:
                for k in k_sizes:
                    if m != 0 and n != 0 and k != 0:
                        all_imaginable.append((m,n,k))
        #assert len(all_imaginable) == 8 # debugging only
        return all_imaginable