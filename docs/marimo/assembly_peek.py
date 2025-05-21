import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Compiling `linalg` to Snitch: **tile size exploration**

        matrix-vector transpose with type `<MxK>, <NxK> -> <MxN>` where `M = 1` (otherwise matmul)
        ```
        inputVectorEltCount = K
        outputVectorEltCount = N
        ```

        This notebook compiles a `linalg matmul_transpose_b` operation to RISC-V with extensions for [Snitch](https://pulp-platform.github.io/snitch/), a neural network accelerator.

        Afterwards, static analysis on the riscv assembly is performed.
        """
    )
    return


@app.cell
def _():
    # Import all the necessary functionality from xDSL for this notebook
    # If you see an error about xdsl not being defined run this cell manually

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
    from xdsl.context import MLContext
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
    return (
        AffineMap,
        AffineMapAttr,
        Attribute,
        Block,
        CanonicalizePass,
        ConvertRiscvScfToRiscvCfPass,
        ConvertSnitchStreamToSnitch,
        ImplicitBuilder,
        LowerSnitchPass,
        MLContext,
        MLIROptPass,
        MemRefType,
        ModuleOp,
        PipelinePass,
        RISCVRegisterAllocation,
        Region,
        RiscvScfLoopRangeFoldingPass,
        SSAValue,
        SnitchRegisterAllocation,
        TypedPtr,
        arith,
        arith_add_fastmath,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_linalg_to_loops,
        convert_linalg_to_memref_stream,
        convert_memref_stream_to_loops,
        convert_memref_stream_to_snitch_stream,
        convert_memref_to_riscv,
        convert_riscv_scf_for_to_frep,
        convert_scf_to_riscv_scf,
        dead_code_elimination,
        f64,
        func,
        get_all_dialects,
        linalg,
        loop_hoist_memref,
        lower_affine,
        memref_streamify,
        reconcile_unrealized_casts,
        riscv_code,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ### Myrtle Cost Model Experiments!!!
        Set your `inputVectorEltCount` and `outputVectorEltCount` variables to your desired values!
        """
    )
    return


@app.cell
def _(mo):
    min_val2 = 1
    max_val2 = 100
    m2 = mo.ui.slider(min_val2, max_val2, value=1, label="M")
    n2 = mo.ui.slider(min_val2, max_val2, value=40, label="N")
    k2 = mo.ui.slider(min_val2, max_val2, value=100, label="K")
    return k2, m2, max_val2, min_val2, n2


@app.cell
def _(k2, m2, mo, n2):
    mo.md(
        f"""
    Input desired L1-tile dimensions. For example, 1-40-100 (a simple case), or 1-64-80 (hardware loop itself inside a loop).
    This tile size will be divided into 8 even peices to deploy to the 8 cores.

    {m2}{m2.value}

    {n2}{n2.value}

    {k2}{k2.value}
    """
    )
    return


@app.cell
def _(
    AffineMap,
    AffineMapAttr,
    Block,
    ImplicitBuilder,
    LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
    LOWER_SNITCH_STREAM_TO_ASM_PASSES,
    MemRefType,
    ModuleOp,
    OPTIMISE_MEMREF_STREAM_PASSES,
    PipelinePass,
    Region,
    arith,
    arith_add_fastmath,
    convert_linalg_to_memref_stream,
    convert_riscv_scf_for_to_frep,
    f64,
    func,
    linalg,
    pipeline_accordion,
):
    def createMatmulTransposeB(
        m=0, outputVectorEltCount=40, inputVectorEltCount=100
    ):
        if m == 0:
            m = 1
        k = inputVectorEltCount
        n = outputVectorEltCount // 8
        print(
            f"Original Type: <MxK>, <NxK> -> <MxN> = <{m}x{k}>, <{outputVectorEltCount}x{k}> -> <{m}x{outputVectorEltCount}>"
        )
        print(
            f"\toutputVectorEltCount = {outputVectorEltCount} which for 8 snitch cores becomes {n}"
        )
        if (n) * 8 != outputVectorEltCount:
            raise Exception("Sorry, only row dimensions divisble by 8 allowed!")
        print(f'\tM is {m}, N = {n}, K = {k}')
        print(
            f"Adjusted Per-Core Type: <MxK>, <NxK> -> <MxN> = <{m}x{k}>, <{n}x{k}> -> <{m}x{n}>"
        )

        a_shape2 = (m, k)
        b_shape2 = (n, k)
        c_shape2 = (m, n)
        # Let's try to recreate the parsed matmultranspose_b using an xDSL IR builder...
        a_type2 = MemRefType(f64, a_shape2)
        b_type2 = MemRefType(f64, b_shape2)
        c_type2 = MemRefType(f64, c_shape2)
        kernel_op2 = func.FuncOp(
            "matmul_transpose_b", ((a_type2, b_type2, c_type2), ())
        )
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

        linalg_module2 = ModuleOp((kernel_op2,))
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

        my_asm_module, my_accordion = pipeline_accordion(
            tuple(("", p) for p in my_linalg_to_snitch.passes),
            linalg_module2,  # theModule
        )
        return (linalg_module2, my_asm_module, m, n, k, my_accordion)
    return (createMatmulTransposeB,)


@app.cell
def _(createMatmulTransposeB, k2, m2, mo, module_html, n2):
    linalg_mod, asm_mod, m, n, k, my_accordion = createMatmulTransposeB(m2.value, n2.value, k2.value)

    mo.md(f"""\

    **Linalg:**
    {module_html(linalg_mod)}
    """
    )
    return asm_mod, k, linalg_mod, m, my_accordion, n


@app.cell
def _(asm_html, asm_mod, mo, riscv_code):
    mo.md(
        f"""
    \
    **Snitch Assembly:**
    {asm_html(riscv_code(asm_mod))}
    """
    )
    return


@app.cell
def _(my_accordion):
    my_accordion
    return


@app.cell
def _(mo):
    mo.md("""### Static Analysis""")
    return


@app.cell
def _(IO, ModuleOp, dataclass):
    from io import StringIO
    from xdsl.dialects.riscv import RISCVAsmOperation, RsRsOffIntegerOperation, MulOp, LiOp
    from xdsl.dialects.riscv_cf import ConditionalBranchOperation
    from xdsl.dialects.riscv_snitch import FrepOuterOp

    @dataclass
    class HardwareLoop:
        """Class for keeping track of hardware loop characteristics"""
        name: str = FrepOuterOp.name
        loop_repeats: int = 1 # number of times loop executes
        body_size: int = 1    # number of instructions in body of the loop

        def total_cost(self) -> float:
            return self.unit_price * self.quantity_on_hand

    @dataclass
    class EnclosingSCFLoop:
        """Class for keeping track of a potential loop surrounding the hardware"""
        name: str = "an enclosing loop"
        iters : int = 1   # number of times the enclosing loop executes
        exists : bool = False # whether the hardware loop is in fact enclosed by another loop


    def look_for_frep(module: ModuleOp, output: IO[str] = StringIO()) -> (list, str):
        freps = []
        frepsAsHLs = []
        for op in module.body.walk():
            assert isinstance(op, RISCVAsmOperation), f"{op}"
            if op.name == FrepOuterOp.name:
                freps.append(op)
        for op in freps:
            asm = op.assembly_line()
            if asm is not None:
                print(asm, file=output)
                if 'immediate' in op.max_rep.op.attributes:
                    imm = op.max_rep.op.attributes['immediate'].value.data
                    print(f'{"\t\t"}this frep was passed an immediate value of {imm}',file=output)
                    frepsAsHLs.append(HardwareLoop(loop_repeats=imm + 1, body_size= op.max_inst))
                else:
                    print(f'{"\t\t"}ERROR: op.max_rep.op: {op.max_rep.op}',file=output)
                print(f'{"\t\t"}max_inst is {op.max_inst}',file=output)
        if len(freps) == 0:
            print(f'{"\t\t"}ERROR no frep found in lowering!!',file=output)
        return (frepsAsHLs, output.getvalue())

    def look_for_enclosing_loop(module: ModuleOp, output: IO[str] = StringIO()) -> (EnclosingSCFLoop,str):
        mulsBefore = []
        mulsAfter = []
        sawFrep = False
        frepCount = 0
        branches = []
        for op in module.body.walk():
            assert isinstance(op, RISCVAsmOperation), f"{op}"
            if op.name == FrepOuterOp.name:
                frepCount = frepCount + 1
            if isinstance(op, ConditionalBranchOperation): 
                branches.append(op)
        if len(branches) > 1:
            print(f'{"\t\t"}ERROR: more than one branch found!!',file=output)
        if frepCount == 0:   
            print(f'{"\t\t"}ERROR no frep found in lowering!!',file=output)
        if frepCount > 1:   
            print(f'{"\t\t"}ERROR more than one frep found in lowering!!',file=output)
        if len(branches) == 0:
            return (EnclosingSCFLoop(exists=False),output.getvalue())
        limb = branches[0] 
        print(f'rs1: {limb.rs1}, rs2: {limb.rs2}',file=output)
        print(f'rs1.op: {limb.rs1.op}, rs2.op: {limb.rs2.op}',file=output)
        if limb.rs2.op.name != LiOp.name:
            return (EnclosingSCFLoop(exists=False),output.getvalue())
        imm = limb.rs2.op.attributes['immediate'].value.data 
        return (EnclosingSCFLoop(limb.name, iters=imm,exists=True), output.getvalue())
    return (
        ConditionalBranchOperation,
        EnclosingSCFLoop,
        FrepOuterOp,
        HardwareLoop,
        LiOp,
        MulOp,
        RISCVAsmOperation,
        RsRsOffIntegerOperation,
        StringIO,
        look_for_enclosing_loop,
        look_for_frep,
    )


@app.cell
def _(createMatmulTransposeB, look_for_enclosing_loop, look_for_frep):
    def check_lowered_matvec_tiling(firstDim : int, outputVectorEltCount : int, inputVectorEltCount : int):
        if firstDim != 0 and firstDim != 1:
            raise Exception("MatVEC requires first dimension M = 1")
        if firstDim == 0:
            firstDim = 1;
        expectedFMADDs = outputVectorEltCount*inputVectorEltCount
        res = False
        # create linalg, then lower to assembly
        linalg_mod, asm_mod, m, n, k, accordion = createMatmulTransposeB(firstDim, outputVectorEltCount, inputVectorEltCount)
        # look for a hardware loop
        loops, commentary = look_for_frep(asm_mod)
        if len(loops) != 1:
            raise Exception(f'ERROR: expected only ONE hardware loop, but found {len(loops)} {"\n"}{commentary}')
        # look for an enclosing loop
        oLoop, commentary2 = look_for_enclosing_loop(asm_mod)
        if not oLoop.exists:
            res = expectedFMADDs == (n*k*8)
        res = expectedFMADDs == ((loops[0].body_size * loops[0].loop_repeats) * oLoop.iters*8)
        return (res,loops[0],oLoop)

    print("0-40-100:")
    print(check_lowered_matvec_tiling(0,40,100))
    print("0-64-40:")
    print(check_lowered_matvec_tiling(0,64,40))
    return (check_lowered_matvec_tiling,)


@app.cell
def _():
    # loops, results = look_for_frep(m, n, k, asm_mod)
    # theLoop, results2 = look_for_enclosing_loop(m, n, k, asm_mod)
    # print(loops)
    # print(theLoop)

    # mo.md(f"""\

    # **Did I find any freps?:**
    # {module_html(results)}
    # **What about an enclosing loop?**
    # {module_html(results2)}
    # """
    # )
    return


@app.cell
def _(mo):
    mo.md("""###Definitions and Imports needed to make the above work""")
    return


@app.cell
def _(MLContext, get_all_dialects):
    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx, dialect_factory, dialect_name


@app.cell
def _():
    from xdsl.transforms.test_lower_linalg_to_snitch import LOWER_SNITCH_STREAM_TO_ASM_PASSES
    from xdsl.transforms.test_lower_linalg_to_snitch import LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES, OPTIMISE_MEMREF_STREAM_PASSES
    return (
        LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
        LOWER_SNITCH_STREAM_TO_ASM_PASSES,
        OPTIMISE_MEMREF_STREAM_PASSES,
    )


@app.cell
def _():
    from dataclasses import dataclass, field
    return dataclass, field


@app.cell
def _(ModuleOp, mo):
    import html as htmllib

    def module_html(module: ModuleOp) -> str:
        return f"""\
        <div style="overflow-y: scroll; height:400px;"><small><code style="white-space: pre-wrap;">{htmllib.escape(str(module))}</code></small></div>
        """

    def asm_html(asm: str) -> str:
        return f"""\
        <div style="overflow-y: scroll; height:400px;">{mo.as_html(mo.ui.code_editor(
                asm, language="python", disabled=True
            ))}</div>
        """
    return asm_html, htmllib, module_html


@app.cell
def _():
    from collections import Counter
    return (Counter,)


@app.cell
def _(Counter, ModuleOp, ModulePass, PipelinePass, ctx, mo, module_html):
    def spec_str(p: ModulePass) -> str:
        if isinstance(p, PipelinePass):
            return ",".join(str(c.pipeline_pass_spec()) for c in p.passes)
        else:
            return str(p.pipeline_pass_spec())

    def pipeline_accordion(
        passes: tuple[tuple[mo.Html, ModulePass], ...], module: ModuleOp
    ) -> tuple[ModuleOp, mo.Html]:
        res = module.clone()
        d = []
        total_key_count = Counter(spec_str(p) for _, p in passes)
        d_key_count = Counter()
        for text, p in passes:
            p.apply(ctx, res)
            spec = spec_str(p)
            d_key_count[spec] += 1
            if total_key_count[spec] != 1:
                header = f"{spec} ({d_key_count[spec]})"
            else:
                header = spec
            html_res = module_html(res)
            d.append(mo.vstack(
                (
                    header,
                    text,
                    mo.md(html_res),
                )
            ))
        return (res, mo.carousel(d))
    return pipeline_accordion, spec_str


if __name__ == "__main__":
    app.run()
