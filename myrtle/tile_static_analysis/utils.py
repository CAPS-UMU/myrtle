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
    """OR a regular matmul with type `<MxK>, <KxN> -> <MxN>`"""
    m: int = 1
    n: int = 1200
    k: int = 400

@dataclass
class TileSizes:
    """Class for keeping track of tiling in each dimension for a"""
    """matmul transpose with type `<MxK>, <NxK> -> <MxN>`"""
    """OR a regular matmul with type `<MxK>, <KxN> -> <MxN>`"""
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

@dataclass
class LoadCounts:
    """Class for keeping track of a regular vs streaming loads while processing one core tile"""
    regular_loads : int = 0
    a_operand_ssr_reuse_loads : int = 0
    a_operand_ssr_start_reuse_loads : int = 0
    b_operand_ssr_loads : int = 0
    not_resused_ssr_loads : int = 0
    total_ssr_loads : int = 0

def givenLoopsCreateLoadCount(hLoop : HardwareLoop, oLoop:EnclosingSCFLoop, ooLoop:EnclosingSCFLoop):
    initialized = LoadCounts()
    initialized.regular_loads = hLoop.body_size*oLoop.iters*ooLoop.iters
    initialized.a_operand_ssr_reuse_loads = (hLoop.body_size-1)*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    initialized.a_operand_ssr_start_reuse_loads = 1*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    initialized.b_operand_ssr_loads = hLoop.body_size*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    initialized.not_resused_ssr_loads = initialized.a_operand_ssr_start_reuse_loads + initialized.b_operand_ssr_loads
    initialized.total_ssr_loads = (hLoop.body_size*2)*hLoop.loop_iters*oLoop.iters*ooLoop.iters
    assert initialized.total_ssr_loads == (initialized.a_operand_ssr_reuse_loads+initialized.a_operand_ssr_start_reuse_loads+initialized.b_operand_ssr_loads)
    return initialized
    
def multByInt(left, i : int):
    prod = LoadCounts()
    prod.regular_loads = i*left.regular_loads
    prod.a_operand_ssr_reuse_loads = i*left.a_operand_ssr_reuse_loads
    prod.a_operand_ssr_start_reuse_loads = i*left.a_operand_ssr_start_reuse_loads
    prod.b_operand_ssr_loads = i*left.b_operand_ssr_loads
    prod.not_resused_ssr_loads = i*left.not_resused_ssr_loads
    prod.total_ssr_loads= i*left.total_ssr_loads
    return prod

def sumLoadCounts(left, other):
    sum = LoadCounts()
    sum.regular_loads = left.regular_loads + other.regular_loads
    sum.a_operand_ssr_reuse_loads = left.a_operand_ssr_reuse_loads + other.a_operand_ssr_reuse_loads
    sum.a_operand_ssr_start_reuse_loads = left.a_operand_ssr_start_reuse_loads + other.a_operand_ssr_start_reuse_loads
    sum.b_operand_ssr_loads = left.b_operand_ssr_loads + other.b_operand_ssr_loads
    sum.not_resused_ssr_loads = left.not_resused_ssr_loads + other.not_resused_ssr_loads
    sum.total_ssr_loads= left.total_ssr_loads + other.total_ssr_loads
    return sum