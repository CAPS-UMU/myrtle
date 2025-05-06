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


def getMicroKernelCount(mat: InputMatrix, sizes: TileSizes):
    k_count = mat.k // sizes.k  # tile the k dimension once
    n_count = mat.n // sizes.n  # tile the n dimension once
    # tile the n dimension again (for each computer core)
    micro_n_sz = coreTile(sizes.n)
    per_cluster_count = sizes.n // micro_n_sz
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
        "Streaming Loads",
        "Reuse Marked Streaming Loads",
        "Outer Loop Iters",
        "HW Loop Body",
        "HW Loop Iters",
    ]
    return columns


# matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1`
def getLoadCountingAnn(mat: InputMatrix, sizes: TileSizes):
    logicalInput = getLogicalSizeAfterPadding(mat, sizes)
    count, microkernel_tile, cluster_tile = getMicroKernelCount(logicalInput, sizes)
    left = microkernel_tile.n * microkernel_tile.k * count
    right = logicalInput.n * logicalInput.k
    # reality check
    if left != right:
        raise Exception(f'after getMicroKernelCount: {left} should = {right}')
    res, hLoop, oLoop = peek_at_lowered_matvec_tiling(cluster_tile)
    if not res:
        raise Exception("Lowering to snitch hardware loop failed!")
    outer_loop_iters = oLoop.iters if oLoop.exists else 1
    regular_loads_per_micro = outer_loop_iters*hLoop.body_size
    streaming_loads_per_micro = outer_loop_iters*(hLoop.body_size+1)*hLoop.loop_repeats
    reuse_marked_streaming_loads_per_micro = outer_loop_iters*(1)*hLoop.loop_repeats
    return(regular_loads_per_micro*count,streaming_loads_per_micro*count,reuse_marked_streaming_loads_per_micro*count,outer_loop_iters, hLoop.body_size, hLoop.loop_repeats)

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
