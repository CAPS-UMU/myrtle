from tile_static_analysis.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, EnclosingSCFLoop, LoadCounts, sumLoadCounts, multByInt, givenLoopsCreateLoadCount
import pandas as pd
import pathlib
from tile_static_analysis.TSA_Quidditch import TSA_Quidditch

class TSA_C(TSA_Quidditch):
    def __init__(
        self,
        unrollAndJamFactor = 8,
    ):
        self.UaJF = unrollAndJamFactor

    def analyze_options(self, df):
        options_as_dicts = list(df.to_dict('records'))
        analyzed = list(map(lambda d: self.analyze_option(d), options_as_dicts))
        # labels = list(df.columns) + self.getLoweringInfoColumnNames()
        # return pd.DataFrame(analyzed, columns=labels)
        cols = analyzed[0].keys()
        print(cols)
        df = pd.DataFrame(analyzed, columns=cols)  
        print(df)
        return df

    def analyze_option(self, d):
        # ['FakeNN JSON Name','M','N','K','m','n','k','JSON Name']
        inputSizes = MatmulInputs(m=d["M"], n=d["N"], k=d["K"])
        tileSizes = TileSizes(m=d["m"], n=d["n"], k=d["k"])
        loweringInfo = self.getLoweringInfoAnnotation(inputSizes, tileSizes)
        d.update(loweringInfo)
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

    def getLoweringInfoAnnotation(self, mat: MatmulInputs, sizes: TileSizes):
        cc_tile_info = self.getCCTileCountsAndSizes(MatmulInputs,sizes)
        cc_tile_count = int(cc_tile_info["cc_tile_count"])
        info ={
            "SSR Config Count":cc_tile_count
        }
        # analyze usual tiles
        res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile_info["usualSize"])
        if not res:
            raise Exception("Lowering to snitch hardware loop failed!")
        info["LittleMPrime Little VecMat Runs"]=ooLoop.iters
        info["LittleMPrime UnrollAndJam Loop Iters"]=oLoop.iters
        info["LittleMPrime HW Loop Iters"]=hLoop.loop_iters
        info["LittleMPrime HW Loop Body Size"]=hLoop.body_size
        info["LittleMPrime Little M Prime"]=cc_tile_info["usualSize"].n
    

        oneUsualTile = givenLoopsCreateLoadCount(hLoop, oLoop, ooLoop)
        print(oneUsualTile)
        print(cc_tile_info["usualCount"])
        allUsualTiles = multByInt(oneUsualTile,int(cc_tile_info["usualCount"]))
        # analyze unusual tiles (tiles with m dimension increased by one to take on part of remainder)
        res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile_info["unusualSize"])
        if not res:
            raise Exception("Lowering to snitch hardware loop failed!")
        info["LittleMPrimeP1 Little VecMat Runs"]=ooLoop.iters
        info["LittleMPrimeP1 UnrollAndJam Loop Iters"]=oLoop.iters
        info["LittleMPrimeP1 HW Loop Iters"]=hLoop.loop_iters
        info["LittleMPrimeP1 HW Loop Body Size"]=hLoop.body_size
        info["LittleMPrimeP1 Little M Prime"]=cc_tile_info["unusualSize"].n
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
