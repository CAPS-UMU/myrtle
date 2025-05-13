from utils import roundUpToNearestMultipleOf, InputMatrix, TileSizes
from peek_at_snitch_assembly import peek_at_lowered_matvec_tiling


def getLogicalSizeAfterPadding(mat: InputMatrix, sizes: TileSizes):
    logicalSize = InputMatrix(n=mat.n, k=mat.k)
    if mat.n % sizes.n != 0:
        logicalSize.n = roundUpToNearestMultipleOf(mat.n, sizes.n)
    if mat.k % sizes.k != 0:
        logicalSize.k = roundUpToNearestMultipleOf(mat.k, sizes.k)
    return logicalSize


"""Given a tile of size sz, what is the subtile size when there are 8 subtiles?"""


def coreTile(sz: int):
    if sz % 8 != 0:
        raise Exception(f"tile size MUST be divisible by 8, yet I have {sz}!")
    else:
        return sz // 8

# return total number of times microkernel runs to complete execution of linalg kernel
def getMicroKernelCount(mat: InputMatrix, sizes: TileSizes):
    k_count = mat.k // sizes.k  # tile the k dimension once
    n_count = mat.n // sizes.n  # tile the n dimension once
    # tile the n dimension again (for each computer core)
    micro_n_sz = coreTile(sizes.n)
    per_cluster_count = sizes.n // micro_n_sz
    assert per_cluster_count == 8
    micro_k_sz = mat.k // k_count
    cluster_tile = InputMatrix(n=sizes.n, k=micro_k_sz)
    microkernel_tile = InputMatrix(n=micro_n_sz, k=micro_k_sz)
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


def yodel():
    print("yodelayheehooooo~~~~~~!")


def LoadCountingAnnColumnNames():
    columns = [
        "Regular Loads",
        "Total Streaming Loads",
        "Start Reuse Streaming Loads",
        "Reused Streaming Loads",
        "Outer Loop Iters",
        "HW Loop Body",
        "HW Loop Iters",
        "Actual HW Loop Runs"
    ]
    return columns


# matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`
def getLoadCountingAnn(mat: InputMatrix, sizes: TileSizes):
    logicalInput = getLogicalSizeAfterPadding(mat, sizes)
    # logicalCount = the number of times a core-sized tile gets processed = (8 * outer L1 tiling loops)
    # this number could be different than the number of microkernel runs per core,
    # because if unroll and jam is performed, a microkernel must run more than once to process a single core-sized tile.
    logicalCount, microkernel_tile, cluster_tile = getMicroKernelCount(logicalInput, sizes)
    left = microkernel_tile.n * microkernel_tile.k * logicalCount
    right = logicalInput.n * logicalInput.k
    # reality check
    if left != right:
        raise Exception(f'after getMicroKernelCount: {left} should = {right}')
    res, hLoop, oLoop = peek_at_lowered_matvec_tiling(cluster_tile)
    if not res:
        raise Exception("Lowering to snitch hardware loop failed!")
    # outer_loop_iters is equivalent to the number of micro-kernel runs needed to process one core-sized tile
    # if unroll and jam was performed, this value > 1.
    outer_loop_iters = oLoop.iters if oLoop.exists else 1
    # loads during micro kernel execution(s) per core
    regular_loads_per_core = outer_loop_iters*hLoop.body_size
    total_streaming_loads_per_core = outer_loop_iters*(hLoop.body_size*2)*hLoop.loop_repeats
    start_reuse_streaming_loads_per_core = outer_loop_iters*(1)*hLoop.loop_repeats
    reused_streaming_loads_per_core = outer_loop_iters*(hLoop.body_size-1)*hLoop.loop_repeats
    assert total_streaming_loads_per_core == (start_reuse_streaming_loads_per_core+reused_streaming_loads_per_core+(outer_loop_iters*(hLoop.body_size)*hLoop.loop_repeats))
    # per cluster
    regular_loads = regular_loads_per_core * logicalCount
    total_streaming_loads = total_streaming_loads_per_core * logicalCount
    start_reuse_streaming_loads = start_reuse_streaming_loads_per_core * logicalCount
    reused_streaming_loads = reused_streaming_loads_per_core * logicalCount
    return(regular_loads,total_streaming_loads,start_reuse_streaming_loads,reused_streaming_loads,outer_loop_iters, hLoop.body_size, hLoop.loop_repeats, logicalCount*outer_loop_iters)

def main():
    yodel()
    yodel()
    input = InputMatrix(n=1200, k=400)
    tiles = TileSizes(n=56, k=100)
    print()
    print(f"input: {input}")
    print(f"tiles: {tiles}")
    print()
    res = getLoadCountingAnn(input, tiles)    
    print(res)


if __name__ == "__main__":
    main()
