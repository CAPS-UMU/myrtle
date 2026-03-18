from tile_static_analysis.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, EnclosingSCFLoop
import pandas as pd
import pathlib
from tile_static_analysis.TileSizeAnalyzer import TileSizeAnalyzer
#from myrtle.peek_at_snitch_assembly import peek_at_lowered_matvec_tiling

# in CSV file, we should have
# SSR Config Count
# VecMat Runs
# UnrollAndJam Loop Iters
# HW Loop Iters
# HW Loop Body Size


class TSA_Quidditch(TileSizeAnalyzer):

    def getLogicalSizeAfterPadding(self, mat: MatmulInputs, sizes: TileSizes):
        logicalSize = MatmulInputs(m=mat.m, n=mat.n, k=mat.k)
        if mat.m % sizes.m != 0:
            logicalSize.m = roundUpToNearestMultipleOf(mat.m, sizes.m)
        if mat.n % sizes.n != 0:
            logicalSize.n = roundUpToNearestMultipleOf(mat.n, sizes.n)
        if mat.k % sizes.k != 0:
            logicalSize.k = roundUpToNearestMultipleOf(mat.k, sizes.k)
        return logicalSize


    """Given a tile of size sz, what is the subtile size when there are 8 subtiles?"""
    def coreTileSize(self, sz: int):
        if sz % 8 != 0:
            raise Exception(f"tile size MUST be divisible by 8, yet I have {sz}!")
        else:
            return sz // 8

    # return total number of times microkernel runs to complete execution of linalg kernel
    def getMicroKernelCount(self, mat: MatmulInputs, sizes: TileSizes):
        k_count = mat.k // sizes.k  # tile the k dimension once
        n_count = mat.n // sizes.n  # tile the n dimension once
        # tile the n dimension again (for each computer core)
        micro_n_sz = self.coreTileSize(sizes.n)
        per_cluster_count = sizes.n // micro_n_sz
        assert per_cluster_count == 8
        micro_k_sz = mat.k // k_count
        cluster_tile = MatmulInputs(n=sizes.n, k=micro_k_sz)
        microkernel_tile = MatmulInputs(n=micro_n_sz, k=micro_k_sz)
        microkernel_count = k_count * n_count * per_cluster_count
        # ONLY FOR DEBUGGING VV
        # print(f"sizes is {sizes}")
        # print(f'per_cluster_count is {per_cluster_count}')
        # print(f"microkernel tile: {microkernel_tile}")
        # print(f"cluster tile: {cluster_tile}")
        # print(f"k_count: {k_count}. n_count: {n_count}. per_cluster_count: {per_cluster_count}")
        # print(f"how many microkernel tiles are there, then?")
        # print(f"{k_count*n_count*per_cluster_count} because {k_count*n_count*per_cluster_count*micro_k_sz*micro_n_sz} = {mat.k * mat.n}")
        # print(f"each core processes (k_count * n_count) = {k_count*n_count} of the {k_count*n_count*per_cluster_count} tiles, because {k_count*n_count*per_cluster_count}/{8}={k_count*n_count*per_cluster_count/8}")
        # ^^ ONLY FOR DEBUGGING
        # reality check
        left = k_count*n_count*per_cluster_count*micro_k_sz*micro_n_sz
        right = mat.k * mat.n
        if left != right:
            raise Exception(f'{left} should = {right}')
        return (microkernel_count, microkernel_tile, cluster_tile)

    # return total number compute core tiles processed 
    # to complete execution of a matmulT linalg kernel
    def getCCTileCount(self, mat: MatmulInputs, sizes: TileSizes):
        m_count = mat.m // sizes.m  # tile the m dimension once
        k_count = mat.k // sizes.k  # tile the k dimension once
        n_count = mat.n // sizes.n  # tile the n dimension once
        # tile the n dimension again (for each computer core)
        cc_n_sz = self.coreTileSize(sizes.n)
        cluster_n_count = sizes.n // cc_n_sz
        assert cluster_n_count == 8 # reality check
        cluster_tile = MatmulInputs(m=sizes.m, n=sizes.n, k=sizes.k)
        cc_tile = MatmulInputs(m=sizes.m, n=cc_n_sz, k=sizes.k)
        cc_tile_count = m_count * k_count * n_count * cluster_n_count
        left = cc_tile_count*sizes.m*sizes.k*cc_n_sz
        right = mat.m * mat.k * mat.n
        if left != right:
            raise Exception(f'{left} should = {right}')
        return (cc_tile_count, cc_tile, cluster_tile)

    def unrollAndJamFactor(self, rowDim):
        options = [7,6,5,4,3,2]
        factor = 1
        for option in options:
            if rowDim % option == 0:
                factor = option
                break
        return factor

    def unrollAndJamOuterLoops(self, rowDim):
        # print(f'outer loops is {rowDim} / {unrollAndJamFactor(rowDim)} which is {rowDim / unrollAndJamFactor(rowDim)}')
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
            (hLoop.body_size * hLoop.loop_iters) * oLoop.iters*ooLoop.iters
        )
        if not res:
            print(f'{cc_tile}:expected FMADD is {expectedFMADDs} but got {(hLoop.body_size * hLoop.loop_iters) * oLoop.iters*ooLoop.iters} instead')
            print("\t",end='')
            print(hLoop)
            print("\t",end='')
            print(oLoop)
            print("\t",end='')
            print(ooLoop)

        return (res, hLoop, oLoop, ooLoop)

    def getLoweringInfoAnnotation(self, mat: MatmulInputs, sizes: TileSizes):
        logicalInput = self.getLogicalSizeAfterPadding(mat, sizes)
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

    def getLoweringInfoColumnNames(self):
        return ["SSR Config Count","Little VecMat Runs","UnrollAndJam Loop Iters","HW Loop Iters","HW Loop Body Size","Regular Loads","Total SSR Loads","A SSR Reuse Loads","A SSR Start Reuse Loads","B SSR Loads","Little N Prime","Little K"]

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

    def exportAnalysisToCSV(self, dispatchNickName, df):
        filename= f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_ss_q_ana.csv"
        df.to_csv(
            filename,
            index=False,
        )
        print("\t",end='')
        print(
            
            f"TSA: wrote analyzed search space to {filename}"
        )
        return filename

