from tile_static_analysis.remainder_utils import ClusterTile, TilingScheme, applyFuncToDict, applyFuncToDictPair
import pandas as pd
import pathlib
from tile_static_analysis.TileSizeAnalyzer import TileSizeAnalyzer
from functools import reduce

class TSA_C_Remainder(TileSizeAnalyzer):
    def __init__(
        self,
        unrollAndJamFactor = 8,
        degreeOfParallelism = 8
    ):
        self.UaJF = unrollAndJamFactor
        self.DoP = degreeOfParallelism

    def analyze_options(self, df):
        options_as_dicts = list(df.to_dict('records'))
        analyzed = list(map(lambda d: self.analyze_option(d), options_as_dicts))
        cols = analyzed[0].keys()
        df = pd.DataFrame(analyzed, columns=cols)  
        return df

    def analyze_option(self, d):
        ts = TilingScheme(int(d["M"]),int(d["N"]),int(d["K"]),int(d["m"]),int(d["n"]),int(d["k"]),self.UaJF,self.DoP,d["remainderTiles"])
        d.update(self.tilingSchemeMetrics(ts))
        return d

    def tilingSchemeMetrics(self,ts):
        cc_tile_count = ts.m_tiles * ts.n_tiles * ts.k_tiles * ts.m_prime_tiles
        info ={
                "m_tiles":ts.m_tiles,
                "n_tiles":ts.n_tiles,
                "k_tiles":ts.k_tiles,
                "SSR Config Count":cc_tile_count,
                "Regular Loads" : 0
        }
        info.update(self.legacyMetrics(ts))    
        clusterTiles = ts.myClusterTiles()
        if len(clusterTiles.values())==1:
            oneTile = next(iter(clusterTiles.values()))
            # print(f"onetile is {oneTile} with metrics {oneTile.metrics()}")
            summedMetrics = applyFuncToDict(lambda x: oneTile.freq * x, oneTile.metrics())#'map(lambda ct : applyFuncToDict(lambda x: ct.freq * x, ct.metrics()),clusterTiles.values())
        else:
            # scale each cluster tile's metrics by its frequency
            scaledMetrics = map(lambda ct : applyFuncToDict(lambda x: ct.freq * x, ct.metrics()),clusterTiles.values())
            # sum cluster tile metrics together
            summedMetrics = reduce(lambda x, y: applyFuncToDictPair(lambda a, b: a+b,x,y), scaledMetrics, ClusterTile.emptyMetrics())
        info.update(summedMetrics)
        info["Total SSR Loads"] = info["A SSR Loads"] + info["B SSR Loads"]
        info["remainderTiles"] = ts.remainderTiles
        return info

    def unrollAndJamFactor(self, rowDim):
            return self.UaJF # fixed unroll and jam factor

    def exportAnalysisToCSV(self, dispatchNickName, df):
        if (df['remainderTiles'] == "000").all():
            suffix = "_c_ana"
        else:
            suffix = "_c_rem_ana"

        #print ((df['remainderTiles'] == df['remainderTiles'][0]).all())
        filename= f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss{suffix}.csv"
        df.to_csv(
            filename,
            index=False,
        )
        print("\t",end='')
        print(
            
            f"TSA: wrote analyzed, padded search space to {filename}"
        )
        return filename
    
    def legacyMetrics(self,ts):
        info = {}
        clusterTiles = ts.myClusterTiles()
        if (ts.M%ts.m == 0) and (ts.N % ts.n == 0) and (ts.K % ts.k == 0):
            onlyKey = next(iter(clusterTiles))
            tile = clusterTiles[onlyKey]
            info["mPrime Little VecMat Runs"]=tile.cctls[0].m_prime_sz
            info["mPrime UnrollAndJam Loop Iters"]=int(tile.n_sz / self.UaJF)
            info["mPrime HW Loop Iters"]=tile.k_sz
            info["mPrime HW Loop Body Size"]=self.UaJF
            info["mPrime"]=tile.cctls[0].m_prime_sz
            info["Little K"]=tile.k_sz
            if len(tile.cctls) == 2:
                info["oldRegPerStream"]=-1
                info["mHat Little VecMat Runs"]=tile.cctls[1].m_prime_size
                info["mHat UnrollAndJam Loop Iters"]=int(tile.n_sz / self.UaJF)
                info["mHat HW Loop Iters"]=tile.k_sz
                info["mHat HW Loop Body Size"]=self.UaJF
                info["mHat"]=tile.cctls[1].m_prime_sz
            else: # we assume the first cc tile is m', not m hat
                info["oldRegPerStream"]=tile.m_sz * tile.n_sz / (128 * tile.k_sz)
                info["mHat Little VecMat Runs"]=tile.cctls[0].m_prime_sz+1
                info["mHat UnrollAndJam Loop Iters"]=int(tile.n_sz / self.UaJF)
                info["mHat HW Loop Iters"]=tile.k_sz
                info["mHat HW Loop Body Size"]=self.UaJF
                info["mHat"]=tile.cctls[0].m_prime_sz+1
        else:
            info["oldRegPerStream"]=-1
            info["mPrime Little VecMat Runs"]=-1
            info["mPrime UnrollAndJam Loop Iters"]=-1
            info["mPrime HW Loop Iters"]=-1
            info["mPrime HW Loop Body Size"]=-1
            info["mPrime"]=-1
            info["mHat Little VecMat Runs"]=-1
            info["mHat UnrollAndJam Loop Iters"]=-1
            info["mHat HW Loop Iters"]=-1
            info["mHat HW Loop Body Size"]=-1
            info["mHat"]=-1
        return info