from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_memref_to_riscv,
    convert_scf_to_riscv_scf,
)
from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
    ConvertRiscvScfToRiscvCfPass,
)
from xdsl.backend.riscv.lowering.convert_snitch_stream_to_snitch import (
    ConvertSnitchStreamToSnitch,
)
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, func, linalg
from xdsl.dialects.builtin import AffineMap, AffineMapAttr, MemRefType, ModuleOp, f64
from xdsl.dialects.riscv import riscv_code
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.passes import PipelinePass
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms import (
    arith_add_fastmath,
    convert_linalg_to_loops,
    convert_linalg_to_memref_stream,
    convert_memref_stream_to_loops,
    convert_memref_stream_to_snitch_stream,
    convert_riscv_scf_for_to_frep,
    dead_code_elimination,
    loop_hoist_memref,
    lower_affine,
    memref_streamify,
    reconcile_unrealized_casts,
)
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.lower_snitch import LowerSnitchPass
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.riscv_scf_loop_range_folding import (
    RiscvScfLoopRangeFoldingPass,
)
from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation
from collections import Counter
from dataclasses import dataclass, field
from xdsl.transforms.test_lower_linalg_to_snitch import (
    LOWER_SNITCH_STREAM_TO_ASM_PASSES,
)
from xdsl.transforms.test_lower_linalg_to_snitch import (
    LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
    OPTIMISE_MEMREF_STREAM_PASSES,
)
from io import StringIO
from xdsl.dialects.riscv import RISCVAsmOperation, RsRsOffIntegerOperation, MulOp, LiOp
from xdsl.dialects.riscv_cf import ConditionalBranchOperation
from xdsl.dialects.riscv_func import FuncOp
from xdsl.dialects.riscv_snitch import FrepOuterOp
from xdsl.passes import ModulePass
from utils import InputMatrix, HardwareLoop, EnclosingSCFLoop


def createMatmulTransposeB(m=0, outputVectorEltCount=40, inputVectorEltCount=100):
    if m == 0:
        m = 1
    k = inputVectorEltCount
    n = outputVectorEltCount // 8
    if (n) * 8 != outputVectorEltCount:
        raise Exception("Sorry, only row dimensions divisible by 8 allowed!")
    ctx = Context()
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    a_shape2 = (m, k)
    b_shape2 = (n, k)
    c_shape2 = (m, n)
    # Let's try to recreate the parsed matmultranspose_b using an xDSL IR builder...
    a_type2 = MemRefType(f64, a_shape2)
    b_type2 = MemRefType(f64, b_shape2)
    c_type2 = MemRefType(f64, c_shape2)
    kernel_op2 = func.FuncOp("matmul_transpose_b", ((a_type2, b_type2, c_type2), ()))
    with ImplicitBuilder(kernel_op2.body) as (a2, b2, c2):
        # Add name hints to make it easier to track how values are lowered
        a2.name_hint = "A"
        b2.name_hint = "B"
        c2.name_hint = "C"
        body2 = Region(Block(arg_types=(f64, f64, f64)))
        linalg.GenericOp(
            inputs=(a2, b2),
            outputs=(c2,),
            body=body2,
            indexing_maps=(
                AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (m, k))),
                AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (n, k))),
                AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (m, n))),
            ),
            iterator_types=(
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.reduction(),
            ),
        )
        with ImplicitBuilder(body2) as (a_val2, b_val2, acc_old_val2):
            prod_val2 = arith.MulfOp(a_val2, b_val2).result
            acc_new_val2 = arith.AddfOp(acc_old_val2, prod_val2).result
            linalg.YieldOp(acc_new_val2)
            # Add more name hints to make it easier to track how values are lowered
            a_val2.name_hint = "a"
            b_val2.name_hint = "b"
            acc_old_val2.name_hint = "acc_old"
            prod_val2.name_hint = "prod"
            acc_new_val2.name_hint = "acc_new"
        func.ReturnOp()

    linalg_module = ModuleOp((kernel_op2,))
    my_linalg_to_snitch = PipelinePass(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemRefStreamPass(),
            arith_add_fastmath.AddArithFastMathFlagsPass(),
            *OPTIMISE_MEMREF_STREAM_PASSES,
            *LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
            convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
            *LOWER_SNITCH_STREAM_TO_ASM_PASSES,
        ]
    )

    my_asm_module = run_thru_all_passes(
        my_linalg_to_snitch.passes,
        linalg_module,
        ctx,
    )

    return (linalg_module, my_asm_module, m, n, k)


def look_for_frep(module: ModuleOp, output=StringIO()) -> (list, str):
    freps = []
    frepsAsHLs = []
    for op in module.body.walk():
        assert isinstance(op, RISCVAsmOperation) or (isinstance(op, FuncOp)), f"{op}"
        if op.name == FrepOuterOp.name:
            freps.append(op)
    for op in freps:
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)
            if "immediate" in op.max_rep.op.attributes:
                imm = op.max_rep.op.attributes["immediate"].value.data
                print(f"this frep was passed an immediate value of {imm}", file=output)
                frepsAsHLs.append(
                    HardwareLoop(loop_repeats=imm + 1, body_size=op.max_inst)
                )
            else:
                print(f"ERROR: op.max_rep.op: {op.max_rep.op}", file=output)
            print(f"max_inst is {op.max_inst}", file=output)
    if len(freps) == 0:
        print(f"ERROR no frep found in lowering!!", file=output)
    return (frepsAsHLs, output.getvalue())


def look_for_enclosing_loop(
    module: ModuleOp, output=StringIO()
) -> (EnclosingSCFLoop, str):
    mulsBefore = []
    mulsAfter = []
    sawFrep = False
    frepCount = 0
    branches = []
    for op in module.body.walk():
        assert isinstance(op, RISCVAsmOperation) or (isinstance(op, FuncOp)), f"{op}"
        if op.name == FrepOuterOp.name:
            frepCount = frepCount + 1
        if isinstance(op, ConditionalBranchOperation):
            branches.append(op)
    if len(branches) > 1:
        print(f"ERROR: more than one branch found!!", file=output)
    if frepCount == 0:
        print(f"ERROR no frep found in lowering!!", file=output)
    if frepCount > 1:
        print(f"ERROR more than one frep found in lowering!!", file=output)
    if len(branches) == 0:
        return (EnclosingSCFLoop(exists=False), output.getvalue())
    limb = branches[0]
    print(f"rs1: {limb.rs1}, rs2: {limb.rs2}", file=output)
    print(f"rs1.op: {limb.rs1.op}, rs2.op: {limb.rs2.op}", file=output)
    if limb.rs2.op.name != LiOp.name:
        return (EnclosingSCFLoop(exists=False), output.getvalue())
    imm = limb.rs2.op.attributes["immediate"].value.data
    return (EnclosingSCFLoop(limb.name, iters=imm, exists=True), output.getvalue())


def peek_at_lowered_matvec_tiling(matvec: InputMatrix):
    expectedFMADDs = matvec.n * matvec.k
    # create linalg, then lower to assembly
    linalg_mod, asm_mod, m, n, k = createMatmulTransposeB(1, matvec.n, matvec.k)
    # look for a hardware loop
    loops, commentary = look_for_frep(asm_mod)
    if len(loops) != 1:
        raise Exception(
            f"ERROR: expected only ONE hardware loop, but found {len(loops)} {commentary}"
        )
    # look for an enclosing loop
    oLoop, commentary2 = look_for_enclosing_loop(asm_mod)
    # print(riscv_code(asm_mod)) # DEBUGGING ONLY
    # check the number of FMADDs is as expected
    res = expectedFMADDs == (
        (loops[0].body_size * loops[0].loop_repeats) * oLoop.iters * 8
    )
    if not oLoop.exists:
        res = expectedFMADDs == (n * k * 8)
    return (res, loops[0], oLoop)


def run_thru_all_passes(
    passes: tuple[ModulePass, ...], module: ModuleOp, ctx: Context
) -> ModuleOp:
    res = module.clone()
    for p in passes:
        p.apply(ctx, res)
    return res


def main():
    print("\n|----------- HOODLE ----------|", end="\n")
    print(check_lowered_matvec_tiling(0, 40, 100))
    print(check_lowered_matvec_tiling(0, 64, 80))


if __name__ == "__main__":
    main()
