from tile_static_analysis.remainder_utils import ClusterTile, TilingScheme, applyFuncToDict, applyFuncToDictPair
import pandas as pd
import pathlib
from tile_static_analysis.TileSizeAnalyzer import TileSizeAnalyzer
from functools import reduce
import math
from pandarallel import pandarallel
import pprint
import re

class TSA_C_Remainder(TileSizeAnalyzer):
    def __init__(
        self,
        unrollAndJamFactor = 8,
        degreeOfParallelism = 8
    ):
        self.UaJF = unrollAndJamFactor
        self.DoP = degreeOfParallelism
    
    def annotate_w_ssr_configs(self,df):
        pandarallel.initialize(verbose=0)
        df["SSR Config Count"] = df[["M","N","K","m","n","k"]].parallel_apply(lambda r: sum([TSA_C_Remainder.computeCoreTileCount(r["M"],r["N"],r["K"],r["m"],r["n"],r["k"],i) for i in range(0,8)] ),axis=1)        
        return df

    def analyze_options(self, df):
        options_as_dicts = list(df.to_dict('records'))
        analyzed = list(map(lambda d: self.analyze_option(d), options_as_dicts))
        cols = analyzed[0].keys()
        df = pd.DataFrame(analyzed, columns=cols)  
        return df

    def analyze_option(self, d):
        ts = TilingScheme(int(d["M"]),int(d["N"]),int(d["K"]),int(d["m"]),int(d["n"]),int(d["k"]),self.UaJF,self.DoP,d["remainderTiles"])
        # info = ts.doubleBufferIters()
        # print(f"The double buffering info is")
        # pprint.pprint(info,width=1)
        d.update(self.tilingSchemeMetrics(ts))
        return d
    
    def computeCoreTileCount(M, N, K, m, n, k, idx):
        cct_per_m_cluster_tiles = int(M / m) * math.ceil(N / n) * math.ceil(K / k)
        # when cluster tile has an m dimension < 8, we don't use all of the cores
        # only cores with indices less than rem_m will execute for this cluster tile
        cct_per_m_rem_cluster_tiles = 1 * math.ceil(N / n) * math.ceil(K / k)
        rem_m = M % m
        if idx < rem_m:
            tiles = cct_per_m_cluster_tiles + cct_per_m_rem_cluster_tiles
        else:
            tiles = cct_per_m_cluster_tiles
        return tiles

    def tilingSchemeMetrics(self,ts):
        #cc_tile_count = ts.m_tiles * ts.n_tiles * ts.k_tiles * ts.m_prime_tiles
        cc_tile_count = 0
        for i in range(0,8):
            cc_tile_count = cc_tile_count + TSA_C_Remainder.computeCoreTileCount(ts.M,ts.N,ts.K,ts.m,ts.n,ts.k,i)
        info ={
                "m_tiles":ts.m_tiles,
                "n_tiles":ts.n_tiles,
                "k_tiles":ts.k_tiles,
                "SSR Config Count":cc_tile_count,
                "Regular Loads" : 0,
                "Total CL Tiles":ts.m_tiles*ts.n_tiles*ts.k_tiles,
        }
        info.update(self.legacyMetrics(ts))    
        clusterTiles = ts.validClusterTiles()
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
        info["FMADDsPerCore"] = info["FMADDs"] / cc_tile_count
        info["Avg A''"] = info["A''"] / cc_tile_count
        info["Avg B'"] = info["B'"] / cc_tile_count
        info["Avg C''"] = info["C''"] / cc_tile_count
        info["Avg CC Tile Size"] = info["CC Tile Size"] / cc_tile_count
        info["Avg (A''+ B') / C''"] = info["(A''+ B') / C''"] / cc_tile_count
        info["Avg L3 Loads"] = info["L3 Loads"] / info["Total CL Tiles"]
        info["Avg L3 Stores"] = info["L3 Stores"] / info["Total CL Tiles"]
        info["Avg A''/ B'"] =  info["A''/ B'"] / cc_tile_count
        info["Avg A'"] =  info["A'"] / info["Total CL Tiles"]
        info["Avg m'_sz / k_size"] = info["m'_sz / k_size"] / cc_tile_count
        info["Avg n'_sz / k_size"] = info["n'_sz / k_size"] / cc_tile_count
        info["Avg m'_sz*n_sz / k_sz"] = info["m'_sz*n_sz / k_sz"] / cc_tile_count
        info["Avg m_sz*n_sz / k_sz"] = info["m_sz*n_sz / k_sz"] / cc_tile_count
        info["remainderTiles"] = ts.remainderTiles
        return info

    def unrollAndJamFactor(self, rowDim):
            return self.UaJF # fixed unroll and jam factor

    def exportAnalysisToCSV(self, dispatchNickName, df, pruned=False):
        if (df['remainderTiles'] == "000").all():
            suffix = "_c_ana"
        else:
            suffix = "_c_rem_ana"
           # nextBunch = df[df["SSR Config Count"].between(0,24576)]
            # filenameSorted = f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_c_pad_ord_L1.csv"
            #nextBunch=nextBunch.sort_values("SSR Config Count", ascending=True)
        
           # print(f'Pruned analyzed ss contains: {nextBunch[["FakeNN JSON Name","SSR Config Count"]]}')
            
            #nextBunch.to_csv(filename, index=False)

        #print ((df['remainderTiles'] == df['remainderTiles'][0]).all())
        if pruned:
            filename= f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss{suffix}_pruned.csv"
        else:
            filename= f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss{suffix}.csv"
        df.to_csv(
            filename,
            index=False,
        )
        print("\t",end='')
        print(
            
            f"TSA: wrote analyzed, remainder search space to {filename}"
        )
        return filename
    
    def legacyMetrics(self,ts):
        info = {}
        clusterTiles = ts.validClusterTiles()
        if (ts.M%ts.m == 0) and (ts.N % ts.n == 0) and (ts.K % ts.k == 0):
            onlyKey = next(iter(clusterTiles))
            tile = clusterTiles[onlyKey]
            info["mPrime Little VecMat Runs"]=tile.cctls[0].m_prime_sz
            info["mPrime UnrollAndJam Loop Iters"]=int(tile.n_sz / self.UaJF)
            info["mPrime HW Loop Iters"]=tile.k_sz
            info["mPrime HW Loop Body Size"]=self.UaJF
            info["mPrime"]=tile.cctls[0].m_prime_sz
           # info["Little K"]=tile.k_sz
            if len(tile.cctls) == 2:
                info["oldRegPerStream"]=-1
                info["mHat Little VecMat Runs"]=tile.cctls[1].m_prime_sz
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
    # Annotate dataframe with features derived from columns already present 
    @classmethod
    def annotate_w_derived_features(cls,df,untimed=False,df_timed = None):
        def handle_timeouts(df):
            df=df [df["FakeNN JSON Name"] != "timeout"].copy()
            return df
        def addFakeTime(df_ut, df_t):
            avgTime = sum(df_t["Kernel Time"].values) / len(df_t["Kernel Time"].values)
            df_ut["Kernel Time"] = avgTime
            maxTime = df_t["Global Sim E2E_dma"].max()*1.5
            df_ut["Global Sim E2E_dma"] = maxTime
            df_ut["dma"] = maxTime
            df_ut["absoluteRank"] = -1
            df_ut["Overlap Stall Time Total"] = -1
            df_ut["Raw Compute Time Total"] = -1
            df_ut["Overlap Stall Time Per Core"] = -1
            return df_ut
        def parseDimM(nm):
            expNameRegex = re.compile(
                r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
            )
            return int(expNameRegex.search(nm).groups()[0])
        def parseDimN(nm):
                expNameRegex = re.compile(
                    r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
                )
                return int(expNameRegex.search(nm).groups()[1])
        def parseDimK(nm):
                expNameRegex = re.compile(
                    r"(\d+)x(\d+)x(\d+)w(\d+)-(\d+)-(\d+)"
                )
                return int(expNameRegex.search(nm).groups()[2])
        def applyRemainderSize(df,nm,D,d):
            df[nm] = df[[D,d]].apply(lambda x: int(x[D]) % int(x[d]),axis=1)
            return df
        df = handle_timeouts(df)
        df["tileAmod32"] = df["m"]/8*df["k"]%32
        df["M"] = df["FakeNN JSON Name"].apply(parseDimM)
        df["N"] = df["FakeNN JSON Name"].apply(parseDimN)
        df["K"] = df["FakeNN JSON Name"].apply(parseDimK)
        df["mRem"] = df["M"] % df["m"]
        df["nRem"] = df["N"] % df["n"]
        df["kRem"] = df["K"] % df["k"]
        containsNaNs = df[df.isna().any(axis=1)]
        if not containsNaNs.empty:
            inspect_cols = [
            "FakeNN JSON Name",
            "M",
            "N",
            "K",
            "m",
            "n",
            "k",
            "mRem",
            ]  # customize as needed
            available_cols = [col for col in inspect_cols if col in df.columns]

            nan_rows = df[df.isna().any(axis=1)]
            print(nan_rows[available_cols])
            raise Exception("data frame of timed values contains NaNs!")
        
        
        df["mRem"] = df["mRem"].astype(int)
        df["nRem"] = df["nRem"].astype(int)
        df["kRem"] = df["kRem"].astype(int)
        # print(df["mRem"])
       # df = applyRemainderSize(df,"mRem","M","m") 
        # df = applyRemainderSize(df,"nRem","N","n") 
        # df = applyRemainderSize(df,"kRem","K","k") 
        df["mnkRem"] = df[["mRem","nRem","kRem"]].apply(lambda x: (int(x["mRem"]),int(x["nRem"]),int(x["kRem"])),axis=1)
        df["mnk"] = df[["m","n","k"]].apply(lambda x: int(x["m"])*int(x["n"])*int(x["k"]),axis=1)
        df["1/mRem"]=1/df["mRem"]
        df["m/mRem"]=df["m"]/df["mRem"]
        df["m % 8"]=df["m"] % 8
        df["niceM"] = df["m"].apply(lambda r: True if r % 8 == 0 else False)
        df["niceMRem"] = df["mRem"].apply(lambda r: True if r == 0 or r % 8 == 0 else False)
        df["howNice"] = df["mRem"].apply(lambda r: "zero" if r == 0 else ("divisBy8" if r % 8 == 0 else "mean"))
        df["bothNice"]=df[["niceM", "niceMRem"]].apply(lambda r: True if r["niceM"] and r["niceMRem"] else False,axis=1)
        df["Total CC Tiles"] = df["SSR Config Count"]
        df["FMADDsMULsPerCore"] = df["FMADDsMULs"] / df["Total CC Tiles"]
        df["1/FMADDS"]=1/df["FMADDsMULsPerCore"]
        df["hypotenuse"] = df[["1/mRem","FMADDsMULsPerCore"]].apply(lambda x: math.sqrt(x["1/mRem"]*x["1/mRem"]+x["FMADDsMULsPerCore"]*x["FMADDsMULsPerCore"]),axis=1)
            
        if not untimed:
            timed = df 
            timed=handle_timeouts(timed)             
            timedContainsNaNs = timed[timed.isna().any(axis=1)]
            if not timedContainsNaNs.empty:
                if timedContainsNaNs.shape[0] != 1:
                    print(timedContainsNaNs[["FakeNN JSON Name","m","n","k","FMADDsMULs","Total CC Tiles","SSR Config Count"]])
                    raise Exception("data frame of timed values contains NaNs!")
            timed["Time (cycles)"]=timed["Global Sim E2E_dma"]
            timed["n / k"]=timed["Avg n'_sz / k_size"]
            timed["comp/memxfer"]=timed["FMADDsMULsPerCore"]/timed["tileA"]+timed["tileB"] #+timed["tileC"]     
            timed["remainderTiles"] = timed["remainderTiles"].apply(lambda x: "000" if x == 0 else f"{x}")
            timed["symbolMarker"] = timed["remainderTiles"].apply(lambda x: "O" if x == "000" else "^")
            timed = timed.sort_values(by="symbolMarker", ascending=True)
            timed["flatColor"] = "pink"
            timed["timedData"] = True
            timed=timed.sort_values("Time (cycles)",ascending=True)
            timed = timed.reset_index(drop=True)
            best=timed["Global Sim E2E_dma"][0]
            timed["diff"] = timed["Time (cycles)"].apply(lambda x: (x - best)/best * 100)
            timed["n/k<1"] = timed["Avg n'_sz / k_size"].apply(lambda x: x < 1.0)
            timed["diff<0.5"] = timed["diff"].apply(lambda x: x <= 0.5)
            df = timed
        else:
            analyzed = df
            analyzed=handle_timeouts(analyzed)
            #analyzed["niceMRem"] = analyzed["mRem"].apply(lambda r: True if r == 0 or r > 8 else False)
            
            analyzed = addFakeTime(analyzed,df_timed)
            analyzed["Y/X"]=analyzed["Avg n'_sz / k_size"] * analyzed["Avg A'"]
            analyzed["timedData"] = False
            analyzed["hypotenuse"] = analyzed[["mRem","1/FMADDS"]].apply(lambda x: math.sqrt(x["mRem"]*x["mRem"]+x["1/FMADDS"]*x["1/FMADDS"]),axis=1)
            analyzed["Time (cycles)"]=analyzed["Global Sim E2E_dma"]
            analyzed["n / k"]=analyzed["Avg n'_sz / k_size"]
            analyzed["comp/memxfer"]=analyzed["FMADDsMULsPerCore"]/analyzed["tileA"]+analyzed["tileB"] #+timed["tileC"]
            
            df = analyzed
        return df