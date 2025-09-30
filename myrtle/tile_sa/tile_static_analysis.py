from tile_sa.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, unrollAndJamFactor, EnclosingSCFLoop, unrollAndJamOuterLoops
import pandas as pd
import pathlib
#from myrtle.peek_at_snitch_assembly import peek_at_lowered_matvec_tiling

# in CSV file, we should have
# SSR Config Count
# VecMat Runs
# UnrollAndJam Loop Iters
# HW Loop Iters
# HW Loop Body Size

def getLogicalSizeAfterPadding(mat: MatmulInputs, sizes: TileSizes):
    logicalSize = MatmulInputs(m=mat.m, n=mat.n, k=mat.k)
    if mat.m % sizes.m != 0:
        logicalSize.m = roundUpToNearestMultipleOf(mat.m, sizes.m)
    if mat.n % sizes.n != 0:
        logicalSize.n = roundUpToNearestMultipleOf(mat.n, sizes.n)
    if mat.k % sizes.k != 0:
        logicalSize.k = roundUpToNearestMultipleOf(mat.k, sizes.k)
    return logicalSize


"""Given a tile of size sz, what is the subtile size when there are 8 subtiles?"""


def coreTileSize(sz: int):
    if sz % 8 != 0:
        raise Exception(f"tile size MUST be divisible by 8, yet I have {sz}!")
    else:
        return sz // 8

# return total number of times microkernel runs to complete execution of linalg kernel
def getMicroKernelCount(mat: MatmulInputs, sizes: TileSizes):
    k_count = mat.k // sizes.k  # tile the k dimension once
    n_count = mat.n // sizes.n  # tile the n dimension once
    # tile the n dimension again (for each computer core)
    micro_n_sz = coreTileSize(sizes.n)
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
def getCCTileCount(mat: MatmulInputs, sizes: TileSizes):
    m_count = mat.m // sizes.m  # tile the m dimension once
    k_count = mat.k // sizes.k  # tile the k dimension once
    n_count = mat.n // sizes.n  # tile the n dimension once
    # tile the n dimension again (for each computer core)
    cc_n_sz = coreTileSize(sizes.n)
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

def yodel():
    print("yodelayheehooooo~~~~~~!")


def LoadCountingAnnColumnNames():
    columns = [
        "Regular Loads",
        "Total Streaming Loads",
        "Other Streaming Loads",
        "Start Reuse Streaming Loads",
        "Reused Streaming Loads",
        "Outer Loop Iters",
        "HW Loop Body",
        "HW Loop Iters",
        "Microkernel Count",
        "Microkernel Row Dim",
        "Microkernel Reduction Dim",
    ]
    return columns



# @dataclass
# class HardwareLoop:
#     """Class for keeping track of hardware loop characteristics"""
#     name: str = "frepOuter"#FrepOuter.name
#     loop_iters: int = 1 # number of times loop executes
#     body_size: int = 1    # number of instructions in body of the loop
# #unrollAndJamFactor


def simulate_peek_at_lowered_matvec_tiling(matvec: MatmulInputs):
    # for potential_factor in range(1, self.pipeline_depth * 2):
    expectedFMADDs = matvec.n * matvec.k
    # create linalg, then lower to assembly
    #linalg_mod, asm_mod, m, n, k = createMatmulTransposeB(1, matvec.n, matvec.k)
    m = matvec.m
    if m == 0:
        m = 1
    k = matvec.k
    n = matvec.n // 8
    if (n) * 8 != matvec.n:
        raise Exception("Sorry, only row dimensions divisible by 8 allowed!")
    
    # look for a hardware loop
    hLoop=HardwareLoop(loop_iters=k,body_size=unrollAndJamFactor(n))
    # look for an enclosing loop
    oLoop=EnclosingSCFLoop(iters=unrollAndJamOuterLoops(n))
    res = expectedFMADDs == (
        (hLoop.body_size * hLoop.loop_iters) * oLoop.iters * 8
    )
    return (res, hLoop, oLoop)

def simulate_peek_at_lowered_matmul_tiling(cc_tile: MatmulInputs):
    # for potential_factor in range(1, self.pipeline_depth * 2):
    expectedFMADDs = cc_tile.m * cc_tile.n * cc_tile.k
    m = cc_tile.m
    if m == 0:
        m = 1
    k = cc_tile.k
    n = cc_tile.n
    # create the hardware loop
    hLoop=HardwareLoop(loop_iters=k,body_size=unrollAndJamFactor(cc_tile.n))
    # look for an enclosing loop
    oLoop=EnclosingSCFLoop(iters=unrollAndJamOuterLoops(cc_tile.n))
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

# matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`
def getLoadCountingAnn(mat: MatmulInputs, sizes: TileSizes):
    logicalInput = getLogicalSizeAfterPadding(mat, sizes)
    # HWLoopRuns = the number of times the snitch hardware loop runs on a single core
    # logicalCount = the number of times a core-sized tile gets processed = (8 * outer L1 tiling loops)
    # this number could be different than the number of microkernel runs per core,
    # because if unroll and jam is performed, a microkernel must run more than once to process a single core-sized tile.
    logicalCount, microkernel_tile, cluster_tile = getMicroKernelCount(logicalInput, sizes)
    left = microkernel_tile.n * microkernel_tile.k * logicalCount
    right = logicalInput.n * logicalInput.k
    # reality check
    if left != right:
        raise Exception(f'after getMicroKernelCount: {left} should = {right}')
    res, hLoop, oLoop = simulate_peek_at_lowered_matvec_tiling(cluster_tile)
    if not res:
        raise Exception("Lowering to snitch hardware loop failed!")
    # outer_loop_iters is equivalent to the number of micro-kernel runs needed to process one core-sized tile
    # if unroll and jam was performed, this value > 1.
    outer_loop_iters = oLoop.iters
    # loads during micro kernel execution(s) per core
    regular_loads_per_core = outer_loop_iters*hLoop.body_size
    other_streaming_loads_per_core = outer_loop_iters*(hLoop.body_size)*hLoop.loop_iters
    total_streaming_loads_per_core = outer_loop_iters*(hLoop.body_size*2)*hLoop.loop_iters
    start_reuse_streaming_loads_per_core = outer_loop_iters*(1)*hLoop.loop_iters
    reused_streaming_loads_per_core = outer_loop_iters*(hLoop.body_size-1)*hLoop.loop_iters
    assert total_streaming_loads_per_core == (start_reuse_streaming_loads_per_core+reused_streaming_loads_per_core+other_streaming_loads_per_core)
    # per cluster
    regular_loads = regular_loads_per_core * logicalCount
    other_streaming_loads = other_streaming_loads_per_core * logicalCount
    total_streaming_loads = total_streaming_loads_per_core * logicalCount
    start_reuse_streaming_loads = start_reuse_streaming_loads_per_core * logicalCount
    reused_streaming_loads = reused_streaming_loads_per_core * logicalCount
    return(regular_loads,total_streaming_loads,other_streaming_loads,start_reuse_streaming_loads,reused_streaming_loads,outer_loop_iters, hLoop.body_size, hLoop.loop_iters, logicalCount,microkernel_tile.n ,microkernel_tile.k)

def getLoweringInfoAnnotation(mat: MatmulInputs, sizes: TileSizes):
    logicalInput = getLogicalSizeAfterPadding(mat, sizes)
    cc_tile_count, cc_tile, cluster_tile = getCCTileCount(logicalInput,sizes)
    # reality check
    left = cc_tile.m * cc_tile.n * cc_tile.k * cc_tile_count
    right = logicalInput.m* logicalInput.n * logicalInput.k
    if left != right:
        raise Exception(f'after getCCTileCount: {left} should = {right}')
    res, hLoop, oLoop, ooLoop = simulate_peek_at_lowered_matmul_tiling(cc_tile)
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

def getLoweringInfoColumnNames():
    return ["SSR Config Count","Little VecMat Runs","UnrollAndJam Loop Iters","HW Loop Iters","HW Loop Body Size","Regular Loads","Total SSR Loads","A SSR Reuse Loads","A SSR Start Reuse Loads","B SSR Loads","Little N Prime","Little K"]

def analyze_options(df):
    options_as_tuples = list(df.itertuples(index=False, name=None))
    analyzed = list(map(lambda tup: analyze_option(tup), options_as_tuples))
    labels = list(df.columns) + getLoweringInfoColumnNames()
    return pd.DataFrame(analyzed, columns=labels)

def analyze_option(tup):
    # ['FakeNN JSON Name','M','N','K','m','n','k','JSON Name']
    #     0              , 1,  2,  3 , 4 , 5 , 6 ,   7
    input = MatmulInputs(m=tup[1], n=tup[2], k=tup[3])
    tiles = TileSizes(m=tup[4], n=tup[5], k=tup[6])
    loweringInfo = getLoweringInfoAnnotation(input, tiles)
    return tup + loweringInfo

def exportAnalysisToCSV(dispatchNickName, df):
    filename= f"{pathlib.Path(__file__).parent.resolve()}/../out/{dispatchNickName}_searchSpace_analyzed.csv"
    df.to_csv(
        filename,
        index=False,
    )
    print("\t",end='')
    print(
        
        f"TSA: wrote analyzed search space to {filename}"
    )
    return filename

def main():
    input = MatmulInputs(n=1200, k=400)
    tiles = TileSizes(n=56, k=100)
    print()
    print(f"input: {input}")
    print(f"tiles: {tiles}")
    print()
    res = getLoadCountingAnn(input, tiles)    
    print(res)


if __name__ == "__main__":
    main()
