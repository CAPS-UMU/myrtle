from tile_static_analysis.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, unrollAndJamFactor, EnclosingSCFLoop, unrollAndJamOuterLoops
import pandas as pd
import pathlib
from tile_static_analysis.TSA_Quidditch import TSA_Quidditch


class TSA_C(TSA_Quidditch):
    def __init__(
        self,
        unrollAndJamFactor = 8,
    ):
        self.unrollAndJamFactor = unrollAndJamFactor

    def analyze_options(self, df):
        options_as_tuples = list(df.itertuples(index=False, name=None))
        analyzed = list(map(lambda tup: self.analyze_option(tup), options_as_tuples))
        labels = list(df.columns) + self.getLoweringInfoColumnNames()
        return pd.DataFrame(analyzed, columns=labels)

    def analyze_option(self, tup):
        # ['FakeNN JSON Name','M','N','K','m','n','k','JSON Name']
        #     0              , 1,  2,  3 , 4 , 5 , 6 ,   7
        input = MatmulInputs(m=tup[1], n=tup[2], k=tup[3])
        tiles = TileSizes(m=tup[4], n=tup[5], k=tup[6])
        loweringInfo = self.getLoweringInfoAnnotation(input, tiles)
        return tup + loweringInfo

    def getLoweringInfoAnnotation(self, mat: MatmulInputs, sizes: TileSizes):
        print(f"{sizes}: I need to change logical input")
        logicalInput = self.getLogicalSizeAfterPadding(mat, sizes)
        print(f"{sizes}: I need to change cc tile count") 
        cc_tile_count, cc_tile, cluster_tile = self.getCCTileCount(logicalInput,sizes)
        # reality check
        left = cc_tile.m * cc_tile.n * cc_tile.k * cc_tile_count
        right = logicalInput.m* logicalInput.n * logicalInput.k
        if left != right:
            raise Exception(f'after getCCTileCount: {left} should = {right}')
        res, hLoop, oLoop, ooLoop = self.simulate_peek_at_lowered_matmul_tiling(cc_tile)
        if not res:
            raise Exception("Lowering to snitch hardware loop failed!")
        # count regular and streaming floating point loads
        # from scratchpad to register per compute core tile processed.
        regular_loads = hLoop.body_size*oLoop.iters*ooLoop.iters
        a_operand_ssr_reuse_loads = (hLoop.body_size-1)*hLoop.loop_iters*oLoop.iters*ooLoop.iters
        a_operand_ssr_start_reuse_loads = 1*hLoop.loop_iters*oLoop.iters*ooLoop.iters
        b_operand_ssr_loads = hLoop.body_size*hLoop.loop_iters*oLoop.iters*ooLoop.iters
        total_ssr_loads_per_core = (hLoop.body_size*2)*hLoop.loop_iters*oLoop.iters*ooLoop.iters
        assert total_ssr_loads_per_core == (a_operand_ssr_reuse_loads+a_operand_ssr_start_reuse_loads+b_operand_ssr_loads)
        # per cluster (and we only have 1 cluster of 8 cores)
        regular_loads = regular_loads * cc_tile_count
        a_operand_ssr_reuse_loads = a_operand_ssr_reuse_loads * cc_tile_count
        a_operand_ssr_start_reuse_loads = a_operand_ssr_start_reuse_loads * cc_tile_count
        b_operand_ssr_loads = b_operand_ssr_loads * cc_tile_count
        total_ssr_loads = total_ssr_loads_per_core * cc_tile_count
        return (cc_tile_count,ooLoop.iters,oLoop.iters,hLoop.loop_iters,hLoop.body_size,regular_loads,total_ssr_loads,a_operand_ssr_reuse_loads,a_operand_ssr_start_reuse_loads,b_operand_ssr_loads,cc_tile.n,cc_tile.k)
