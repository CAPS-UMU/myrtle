from dataclasses import dataclass
from xdsl.dialects.riscv_snitch import FrepOuter

def roundUpToNearestMultipleOf(num, row_dim):
  remainder = num % row_dim
  if (remainder != 0):
    return (num//row_dim) * row_dim + row_dim
  else:
    return num

@dataclass
class InputMatrix:
    """Class for keeping track of matrix dimensions in"""
    """matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)"""
    n: int = 1200
    k: int = 400

@dataclass
class TileSizes:
    """Class for keeping track of matrix-vector transpose tiling in each dimension;"""
    """matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)"""
    m: int = 1
    n : int = 40
    k : int = 100

@dataclass
class HardwareLoop:
    """Class for keeping track of hardware loop characteristics"""
    name: str = FrepOuter.name
    loop_repeats: int = 1 # number of times loop executes
    body_size: int = 1    # number of instructions in body of the loop

@dataclass
class EnclosingSCFLoop:
    """Class for keeping track of a potential loop surrounding the hardware"""
    name: str = "an enclosing loop"
    iters : int = 1   # number of times the enclosing loop executes
    exists : bool = False # whether the hardware loop is in fact enclosed by another loop

def unrollAndJamFactor(rowDim):
    options = [7,6,5,4,3,2]
    factor = 1
    for option in options:
        if rowDim % option == 0:
            factor = option
            break
    return factor

def unrollAndJamOuterLoops(rowDim):
    # print(f'outer loops is {rowDim} / {unrollAndJamFactor(rowDim)} which is {rowDim / unrollAndJamFactor(rowDim)}')
    return rowDim / unrollAndJamFactor(rowDim)