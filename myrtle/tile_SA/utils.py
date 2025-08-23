from dataclasses import dataclass
#from xdsl.dialects.riscv_snitch import FrepOuter

def roundUpToNearestMultipleOf(num, row_dim):
  remainder = num % row_dim
  if (remainder != 0):
    return (num//row_dim) * row_dim + row_dim
  else:
    return num

@dataclass
class MatmulInputs:
    """Class for keeping track of input dimensions of a"""
    """matmul transpose with type `<MxK>, <NxK> -> <MxN>`"""
    m: int = 1
    n: int = 1200
    k: int = 400

@dataclass
class TileSizes:
    """Class for keeping track of tiling in each dimension for a"""
    """matmul transpose with type `<MxK>, <NxK> -> <MxN>`"""
    m: int = 1
    n : int = 40
    k : int = 100

@dataclass
class HardwareLoop:
    """Class for keeping track of hardware loop characteristics"""
    name: str = "frepOuter"#FrepOuter.name
    loop_iters: int = 1 # number of times loop executes
    body_size: int = 1  # number of instructions in body of the loop
    # right now, we assume all the instructions inside the body are the SAME
    # we assume each instruction takes 2 operands: a, b, -> c
    # we assume operands a and b using SSRs, and c does not


@dataclass
class EnclosingSCFLoop:
    """Class for keeping track of a potential loop surrounding the hardware loop"""
    name: str = "an enclosing loop"
    iters : int = 1   # number of times the enclosing loop executes; 
                      # if iters == 1, enclosing loop DNE.

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
    if rowDim == 1:
        return 1
    if unrollAndJamFactor(rowDim) != 1:
        return int(rowDim / unrollAndJamFactor(rowDim))
    else:
        return rowDim