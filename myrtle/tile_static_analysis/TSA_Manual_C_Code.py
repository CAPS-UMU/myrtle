from tile_static_analysis.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, EnclosingSCFLoop, LoadCounts, sumLoadCounts, multByInt, givenLoopsCreateLoadCount, ComputeCoreTiles, LoadCounter, add
import pandas as pd
import pathlib
# from tile_static_analysis.TSA_Manual_C_Code_deprecated import TSA_C
# from tile_static_analysis.TSA_Manual_C_Code_deprecated import TSA_C

from tile_static_analysis.TSA_Quidditch import TSA_Quidditch

class TSA_C(TSA_Quidditch):
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
        # labels = list(df.columns) + self.getLoweringInfoColumnNames()
        # return pd.DataFrame(analyzed, columns=labels)
        cols = analyzed[0].keys()
        # print(cols)
        df = pd.DataFrame(analyzed, columns=cols)  
        # print(df)
        return df

    def analyze_option(self, d):
        # ['FakeNN JSON Name','M','N','K','m','n','k','JSON Name']
        inputSizes = MatmulInputs(m=d["M"], n=d["N"], k=d["K"])
        m_num = int(d["M"] / d["m"])
        n_num = int(d["N"] / d["n"])
        k_num = int(d["K"] / d["k"])
        l1TileSizes = TileSizes(m=d["m"], n=d["n"], k=d["k"], m_count=m_num, n_count=n_num, k_count=k_num)
        ccTileSizes = self.getComputeCoreTileSizes(l1TileSizes)
        # print(l1TileSizes)
        # print(ccTileSizes)
        # loads=self.getStreamingLoadsPerClusterTile(ccTileSizes)
        # loads=loads.mapMult(m_num).mapMult(n_num).mapMult(k_num)
        # print(f'final: {loads}')
        # print(self.getLoweringInfo(l1TileSizes,ccTileSizes))
        # print()
        # loweringInfo = self.getLoweringInfoAnnotation(inputSizes, l1TileSizes)
        loweringInfo = self.getLoweringInfo(l1TileSizes,ccTileSizes)
        d.update(loweringInfo)
        # add L3 load counting
        d["m_tiles"] = d["M"]/ d["m"]
        d["n_tiles"] = d["N"]/ d["n"]
        d["k_tiles"] = d["K"]/ d["k"]
        d["L3 Loads"] = (d["m_tiles"]*d["n_tiles"]*d["k_tiles"])*(d["m"]*d["k"]+d["k"]*d["n"]) + (d["m_tiles"]*d["n_tiles"])*(d["m"]*d["n"])
    
        return d
    
    """Given a tile with parallel dim of sz, what is the compute core's dim when there are 8 cores?"""
    def coreTileParDimSize(self, sz: int):
        usualSize = sz // 8
        remainder = sz % 8
        usualCount = (sz / usualSize) - remainder
        subtiles = {
            "totalCount" : usualCount + remainder,
            "usualCount": usualCount,
            "usualSize" : usualSize, # most of the compute core tiles have this size
            "unusualCount": remainder, # the remainder is spread out across the first remainder tiles (1 extra elt each)
            "unusualSize": usualSize + 1
        }
        assert subtiles["totalCount"] % 8 == 0
        return subtiles

    # return total number compute core tiles processed 
    # to complete execution of a matmul kernel
    def getCCTileCountsAndSizes(self, mat: MatmulInputs, sizes: TileSizes):
        m_count = mat.m // sizes.m  # tile the m dimension once
        k_count = mat.k // sizes.k  # tile the k dimension once
        n_count = mat.n // sizes.n  # tile the n dimension once
        # tile the m dimension again (for each computer core)
        cc_m_sz = self.coreTileParDimSize(sizes.m)
        ccTileInfo ={
            "cc_tile_count" : int(cc_m_sz["totalCount"] * k_count * n_count),
            "usualCount" : int(cc_m_sz["usualCount"] * k_count * n_count),
            "usualSize" : MatmulInputs(m=cc_m_sz["usualSize"],n=sizes.n, k=sizes.k),
            "unusualCount": int(cc_m_sz["unusualCount"] * k_count * n_count),
            "unusualSize": MatmulInputs(m=cc_m_sz["unusualSize"],n=sizes.n, k=sizes.k)
        }
        return ccTileInfo

    def unrollAndJamFactor(self, rowDim):
            return self.UaJF # fixed unroll and jam factor

    def unrollAndJamOuterLoops(self, rowDim):
        if rowDim == 1:
            return 1
        if self.unrollAndJamFactor(rowDim) != 1:
            return int(rowDim / self.unrollAndJamFactor(rowDim))
        else:
            return rowDim

    def simulate_peek_at_lowered_matmul_tiling(self, cc_tile: MatmulInputs):
        # for potential_factor in range(1, self.pipeline_depth * 2):
        expectedFMADDs = cc_tile.m * cc_tile.n * cc_tile.k
        m = cc_tile.m
        if m == 0:
            m = 1
        k = cc_tile.k
        n = cc_tile.n
        # create the hardware loop
        hLoop=HardwareLoop(loop_iters=k,body_size=self.unrollAndJamFactor(cc_tile.n))
        # look for an enclosing loop
        oLoop=EnclosingSCFLoop(iters=self.unrollAndJamOuterLoops(cc_tile.n))
        # every matmul is really a loop of matvecs...
        ooLoop = EnclosingSCFLoop(iters=cc_tile.m)
        res = expectedFMADDs == (
            (hLoop.body_size * hLoop.loop_iters) * oLoop.iters * ooLoop.iters
        )
        if not res:
            print(f'{cc_tile}:expected FMADD is {expectedFMADDs} = {cc_tile.m } * {cc_tile.n} * {cc_tile.k} * {ooLoop.iters} but got {(hLoop.body_size * hLoop.loop_iters) * oLoop.iters*ooLoop.iters} instead')
            print("\t",end='')
            print(hLoop)
            print("\t",end='')
            print(oLoop)
            print("\t",end='')
            print(ooLoop)

        return (res, hLoop, oLoop, ooLoop)

    # def getLoweringInfoAnnotation(self, mat: MatmulInputs, sizes: TileSizes):
    #     cc_tile_info = self.getCCTileCountsAndSizes(MatmulInputs,sizes)
    #     cc_tile_count = int(cc_tile_info["cc_tile_count"])
    #     info ={
    #         "SSR Config Count":cc_tile_count
    #     }
    #     # analyze usual tiles
    #     res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile_info["usualSize"])
    #     if not res:
    #         raise Exception("Lowering to snitch hardware loop failed!")
    #     info["LittleMPrime Little VecMat Runs"]=ooLoop.iters
    #     info["LittleMPrime UnrollAndJam Loop Iters"]=oLoop.iters
    #     info["LittleMPrime HW Loop Iters"]=hLoop.loop_iters
    #     info["LittleMPrime HW Loop Body Size"]=hLoop.body_size
    #     info["LittleMPrime Little M Prime"]=cc_tile_info["usualSize"].n
    

    #     oneUsualTile = givenLoopsCreateLoadCount(hLoop, oLoop, ooLoop)
    #     # print(oneUsualTile)
    #     # print(cc_tile_info["usualCount"])
    #     allUsualTiles = multByInt(oneUsualTile,int(cc_tile_info["usualCount"]))
    #     # analyze unusual tiles (tiles with m dimension increased by one to take on part of remainder)
    #     res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile_info["unusualSize"])
    #     if not res:
    #         raise Exception("Lowering to snitch hardware loop failed!")
    #     info["LittleMPrimeP1 Little VecMat Runs"]=ooLoop.iters
    #     info["LittleMPrimeP1 UnrollAndJam Loop Iters"]=oLoop.iters
    #     info["LittleMPrimeP1 HW Loop Iters"]=hLoop.loop_iters
    #     info["LittleMPrimeP1 HW Loop Body Size"]=hLoop.body_size
    #     info["LittleMPrimeP1 Little M Prime"]=cc_tile_info["unusualSize"].n
    #     oneUnUsualTile = LoadCounts
    #     oneUnUsualTile.intializeGivenLoops=(hLoop, oLoop, ooLoop)
    #     allUnUsualTiles = multByInt(oneUnUsualTile,int(cc_tile_info["unusualCount"]))
    #     # sum results of usual and unusual tiles
    #     allTiles = sumLoadCounts(allUsualTiles,allUnUsualTiles)
       
    #     info["Regular Loads"]=allTiles.regular_loads
    #     info["Total SSR Loads"]=allTiles.total_ssr_loads
    #     info["Not Reused SSR Loads"]=allTiles.not_resused_ssr_loads
    #     info["A SSR Reuse Loads"]=allTiles.a_operand_ssr_reuse_loads
    #     info["A SSR Start Reuse Loads"]=allTiles.a_operand_ssr_start_reuse_loads
    #     info["B SSR Loads"]=allTiles.b_operand_ssr_loads
    #     info["Little K"]=cc_tile_info["usualSize"].k
        
    #     return info

    def exportAnalysisToCSV(self, dispatchNickName, df):
        filename= f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_c_ana.csv"
        df.to_csv(
            filename,
            index=False,
        )
        print("\t",end='')
        print(
            
            "TSA: wrote analyzed search space to"
        )
        print("\t", end="")
        print(f"     {filename}")
        return filename

    def getComputeCoreTileSizes(self, l1: TileSizes):
        mPrime = l1.m // 8
        mHat = mPrime + 1
        remainderCount = l1.m % 8
        regularCount = 8-remainderCount 
        # print()
        # print(f"L1 tile sizes are: {l1.m}-{l1.n}-{l1.k}")
        # print(f"mPrime:{mPrime} mHat:{mHat} remainderCount:{remainderCount} regularCount:{regularCount}")
        assert (remainderCount + regularCount) == 8
        mPrime_tiles = TileSizes(m=mPrime,n=l1.n,k=l1.k,m_count=regularCount,n_count=1,k_count=1)
        mHat_tiles = TileSizes(m=mHat,n=l1.n,k=l1.k,m_count=remainderCount,n_count=1,k_count=1)
        ccTileInfo= ComputeCoreTiles(mPrime = mPrime_tiles, mHat= mHat_tiles)
        return ccTileInfo

    def getStreamingLoadsPerCoreTile(self, cc: TileSizes):
            # repeat m times:
                # repeat n/8 times:
                    # we load 8 initial values for C 1x8
                    # 8 fmadds cx, a, b, cx repeated in hardware loop (k-2) times
        c_regular_loads = 8 * (cc.n / 8) * cc.m
        a_ssr_loads = 8 * (cc.k -2) * (cc.n /8) * cc.m
        a_ssr_reuse_loads = 7 * (cc.k -2) * (cc.n /8) * cc.m
        b_ssr_loads = 8 * (cc.k -2) * (cc.n /8) * cc.m
        return LoadCounter(a_ssr=a_ssr_loads,a_ssr_reuse=a_ssr_reuse_loads, b_ssr=b_ssr_loads, c_regular=c_regular_loads)
    
    def getStreamingLoadsPerClusterTile(self, cc: ComputeCoreTiles):
        regSizeLoadCounting =self.getStreamingLoadsPerCoreTile(cc.mPrime)
        remSizeLoadCounting =self.getStreamingLoadsPerCoreTile(cc.mHat)
        #print(f"regular: {regSizeLoadCounting}")
        
        # print(regSizeLoadCounting)
        reg = regSizeLoadCounting.mapMult(cc.mPrime.m_count).mapMult(cc.mPrime.n_count).mapMult(cc.mPrime.k_count)
        rem = remSizeLoadCounting.mapMult(cc.mHat.m_count).mapMult(cc.mHat.n_count).mapMult(cc.mHat.k_count)
        # print(f"regular * m' * n * k = * {cc.mPrime.m_count} * {cc.mPrime.n_count} * {cc.mPrime.k_count}: {reg}")
        # print(f"remainder: {remSizeLoadCounting}")
        # print(f"remainder * mHat * n * k: {rem}")
        total = add(reg,rem)
        # print(f"total: {total}")
        # print()
        return total
        # print(reg)
        # reg = regSizeLoadCounting * cc.regularSize.m_count
        # rem = remSizeLoadCounting * cc.remainderSize.m_count
        # return LoadCounter(a_ssr=a_ssr_loads, b_ssr=b_ssr_loads, c_regular=c_regular_loads)

    def getLoweringInfo(self, l1Tiles: TileSizes, cc : ComputeCoreTiles):
            #print(f"{l1Tiles.m_count} {l1Tiles.n_count} {l1Tiles.m_count} {8} {cc.mPrime.n_count} {cc.mPrime.k_count}")
            cluster_tile_count = int(l1Tiles.m_count * l1Tiles.n_count * l1Tiles.k_count)
            cc_tile_count = cluster_tile_count * 8 * 1 * 1
            info ={
                "SSR Config Count":cc_tile_count
            }
            info["mPrime Little VecMat Runs"]=cc.mPrime.m
            info["mPrime UnrollAndJam Loop Iters"]=int(cc.mPrime.n / 8)
            info["mPrime HW Loop Iters"]=cc.mPrime.k
            info["mPrime HW Loop Body Size"]=8
            info["mPrime"]=cc.mPrime.m
            info["mHat Little VecMat Runs"]=cc.mHat.m
            info["mHat UnrollAndJam Loop Iters"]=int(cc.mHat.n / 8)
            info["mHat HW Loop Iters"]=cc.mHat.k
            info["mHat HW Loop Body Size"]=8
            info["mHat"]=cc.mHat.m
            loads=self.getStreamingLoadsPerClusterTile(cc)
            allTiles = loads.mapMult(cluster_tile_count)
            info["Regular Loads"]=allTiles.c_regular
            info["Total SSR Loads"]=allTiles.a_ssr + allTiles.b_ssr
            info["A Not Reused SSR Loads"]=allTiles.a_ssr - allTiles.a_ssr_reuse
            info["A SSR Reuse Loads"]=allTiles.a_ssr_reuse
            info["A SSR Start Reuse Loads"]=int(allTiles.a_ssr_reuse / 7)
            info["B SSR Loads"]=allTiles.b_ssr
            info["Little K"]=cc.mPrime.k
            assert(allTiles.a_ssr_reuse % 7 == 0)
            return info
            # analyze usual tiles
            res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile_info["usualSize"])
            if not res:
                raise Exception("Lowering to snitch hardware loop failed!")
            info["mPrime Little VecMat Runs"]=ooLoop.iters
            info["mPrime UnrollAndJam Loop Iters"]=oLoop.iters
            info["mPrime HW Loop Iters"]=hLoop.loop_iters
            info["mPrime HW Loop Body Size"]=hLoop.body_size
            info["mPrime"]=cc_tile_info["usualSize"].n
        

            oneUsualTile = givenLoopsCreateLoadCount(hLoop, oLoop, ooLoop)
            # print(oneUsualTile)
            # print(cc_tile_info["usualCount"])
            allUsualTiles = multByInt(oneUsualTile,int(cc_tile_info["usualCount"]))
            # analyze unusual tiles (tiles with m dimension increased by one to take on part of remainder)
            res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile_info["unusualSize"])
            if not res:
                raise Exception("Lowering to snitch hardware loop failed!")
            info["mHat Little VecMat Runs"]=ooLoop.iters
            info["mHat UnrollAndJam Loop Iters"]=oLoop.iters
            info["mHat HW Loop Iters"]=hLoop.loop_iters
            info["mHat HW Loop Body Size"]=hLoop.body_size
            info["mHat"]=cc_tile_info["unusualSize"].n
            oneUnUsualTile = LoadCounts
            oneUnUsualTile.intializeGivenLoops=(hLoop, oLoop, ooLoop)
            allUnUsualTiles = multByInt(oneUnUsualTile,int(cc_tile_info["unusualCount"]))
            # sum results of usual and unusual tiles
            allTiles = sumLoadCounts(allUsualTiles,allUnUsualTiles)
        
            info["Regular Loads"]=allTiles.regular_loads
            info["Total SSR Loads"]=allTiles.total_ssr_loads
            info["Not Reused SSR Loads"]=allTiles.not_resused_ssr_loads
            info["A SSR Reuse Loads"]=allTiles.a_operand_ssr_reuse_loads
            info["A SSR Start Reuse Loads"]=allTiles.a_operand_ssr_start_reuse_loads
            info["B SSR Loads"]=allTiles.b_operand_ssr_loads
            info["Little K"]=cc_tile_info["usualSize"].k
            
            return info       