# Formulas for Load Counting Explained

When lowering a tiled `linalg_matmul_transpose_b` operation with dimensions `<MxK>, <NxK> -> <MxN>` to snitch, there are two possibilities:

1. The tile dimensions result in a single snitch hardware loop (simple case)
2. The tile dimensions result in a snitch hardware loop enclosed inside a regular loop (slightly more complicated case)

## 1. Simple Case Example

- Original matmul transpose with dimensions `<1x400>, <1200x400> -> <1x1200>`

- We choose to tile dimensions `M-N-K` with sizes `1-40-100`, *which then gets tiled into 8 smaller pieces to deploy on each core*.
  ```
  Original Type: <MxK>, <NxK> -> <MxN> = <1x100>, <40x100> -> <1x40>
  ```

- The `matmul_tranpose_b` run on each compute core will therefore be of size `<1x100>, <5x100> -> <1x5>`
  ```
  Adjusted Per-Core Type: <MxK>, <NxK> -> <MxN> = <1x100>, <5x100> -> <1x5>
  ```

Using xDSL, we construct an MLIR Linalg Operation to lower:

```
builtin.module {
  func.func @matmul_transpose_b(%A : memref<1x100xf64>, %B : memref<5x100xf64>, %C : memref<1x5xf64>) {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, 
                                     affine_map<(d0, d1, d2) -> (d1, d2)>, 
                                     affine_map<(d0, d1, d2) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%A, %B : memref<1x100xf64>, memref<5x100xf64>) outs(%C : memref<1x5xf64>) {
    ^0(%a : f64, %b : f64, %acc_old : f64):
      %prod = arith.mulf %a, %b : f64
      %acc_new = arith.addf %acc_old, %prod : f64
      linalg.yield %acc_new : f64
    }
    func.return
  }
}
```

Note the streaming register setup from `snitch-allocate-registers` pass output:

```
riscv_func.func @matmul_transpose_b(%A : !riscv.reg<a0>, %B : !riscv.reg<a1>, %C : !riscv.reg<a2>) {
      %A_1 = riscv.mv %A : (!riscv.reg<a0>) -> !riscv.reg
      %B_1 = riscv.mv %B : (!riscv.reg<a1>) -> !riscv.reg
      %C_1 = riscv.mv %C : (!riscv.reg<a2>) -> !riscv.reg
      snitch_stream.streaming_region {
        patterns = [
          #snitch_stream.stride_pattern<ub = [100], strides = [8], repeat = 5>,
          #snitch_stream.stride_pattern<ub = [100, 5], strides = [8, 800]>
        ]
      } 
```

- streaming register `ft0` is configured to read the same element 5 times in a row due to its **repeat value of 5**
- streaming register `ft1` reads a different element (offset by a regular stride) every time

Resulting Snitch Assembly:

![0-40-100](/home/emily/myrtle/docs/md/0-40-100.png)

Close-up on hardware loop:

![](/home/emily/myrtle/docs/md/0-40-100-HL.png)

![](/home/emily/myrtle/docs/md/simple-rules.png)But wait! We started with a `matmul_transpose_b` of size  `<1x400>, <1200x400> -> <1x1200>`!

Since we tile dimensions `M-N-K` with sizes `1-40-100`, our original operation turns into

```
for k = 1 to 12 (since 1200 / 100 = 12)
for n = 1 to 10 (since 400 / 40 = 10)
for each of the 8 snitch cores
	matmul_transpose <1x100>, <5x100> -> <1x5>
```

**So total loads = 12 * 10 * 8 * `TotalLoadsPerCore` = 580800 Loads**

## 2. Slightly more complicated case

- Original matmul transpose with dimensions `<1x400>, <1200x400> -> <1x1200>`

- We choose to tile dimensions `M-N-K` with sizes `1-64-80`, *which then gets tiled into 8 smaller pieces to deploy on each core*.

  ```
  Original Type: <MxK>, <NxK> -> <MxN> = <1x80>, <64x80> -> <1x64>
  ```

- The `matmul_tranpose_b` run on each compute core will therefore be of size `<1x80>, <8x80> -> <1x8>`

  ```
  Adjusted Per-Core Type: <MxK>, <NxK> -> <MxN> = <1x80>, <8x80> -> <1x8>
  ```

Using xDSL, we construct an MLIR Linalg Operation to lower:

```
builtin.module {
  func.func @matmul_transpose_b(%A : memref<1x80xf64>, %B : memref<8x80xf64>, %C : memref<1x8xf64>) {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, 
                                     affine_map<(d0, d1, d2) -> (d1, d2)>, 
                                     affine_map<(d0, d1, d2) -> (d0, d1)>], 
                    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%A, %B : memref<1x80xf64>, memref<8x80xf64>) outs(%C : memref<1x8xf64>) {
    ^0(%a : f64, %b : f64, %acc_old : f64):
      %prod = arith.mulf %a, %b : f64
      %acc_new = arith.addf %acc_old, %prod : f64
      linalg.yield %acc_new : f64
    }
    func.return
  }
}
```

Note the streaming register setup from `snitch-allocate-registers` pass output:

```
riscv_func.func @matmul_transpose_b(%A : !riscv.reg<a0>, %B : !riscv.reg<a1>, %C : !riscv.reg<a2>) {
      %A_1 = riscv.mv %A : (!riscv.reg<a0>) -> !riscv.reg
      %B_1 = riscv.mv %B : (!riscv.reg<a1>) -> !riscv.reg
      %C_1 = riscv.mv %C : (!riscv.reg<a2>) -> !riscv.reg
      snitch_stream.streaming_region {
        patterns = [
          #snitch_stream.stride_pattern<ub = [2, 80], strides = [0, 8], repeat = 4>,
          #snitch_stream.stride_pattern<ub = [2, 80, 4], strides = [2560, 8, 640]>
        ]
      } ins(%A_1, %B_1 : !riscv.reg, !riscv.reg) 
```

- streaming register `ft0` is configured to read the same element 4 times in a row due to its **repeat value of 4**
- streaming register `ft1` reads a different element (offset by a regular stride) every time

Resulting Snitch Assembly:

Part 1:

pic!!

Part2:

pic!!

## Appendix

### 1. Simple Case: assembly as plain text

```
.text
.globl matmul_transpose_b
.p2align 2
 # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "a2", "zero"], "allocated_float": ["ft0", "ft1", "ft3", "ft4", "ft5", "ft6", "ft7"], "allocated_int": ["a0", "a1", "a2", "t0", "t1", "t2", "t3", "zero"]}
matmul_transpose_b:
    mv t2, a0
    mv t1, a1
    mv t0, a2
    li t3, 99
    scfgwi t3, 64                                # dm 0 dim 0 bound
    li t3, 8
    scfgwi t3, 192                               # dm 0 dim 0 stride
    li t3, 4
    scfgwi t3, 32                                # dm 0 repeat
    li t3, 4
    scfgwi t3, 65                                # dm 1 dim 0 bound
    li t3, 99
    scfgwi t3, 97                                # dm 1 dim 1 bound
    li t3, 800
    scfgwi t3, 193                               # dm 1 dim 0 stride
    li t3, -3192
    scfgwi t3, 225                               # dm 1 dim 1 stride
    scfgwi zero, 33                              # dm 1 repeat
    scfgwi t2, 768                               # dm 0 dim 0 source
    scfgwi t1, 801                               # dm 1 dim 1 source
    csrrsi zero, 1984, 1                         # SSR enable
    mv t1, t0
    fld ft7, 0(t1)                               # load double from memref of shape (1, 5)
    fld ft6, 8(t0)                               # load double from memref of shape (1, 5)
    fld ft5, 16(t0)                              # load double from memref of shape (1, 5)
    fld ft4, 24(t0)                              # load double from memref of shape (1, 5)
    fld ft3, 32(t0)                              # load double from memref of shape (1, 5)
    li t1, 99
    frep.o t1, 5, 0, 0
    fmadd.d ft7, ft0, ft1, ft7
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    mv t1, t0
    fsd ft7, 0(t1)                               # store double value to memref of shape (1, 5)
    fsd ft6, 8(t0)                               # store double value to memref of shape (1, 5)
    fsd ft5, 16(t0)                              # store double value to memref of shape (1, 5)
    fsd ft4, 24(t0)                              # store double value to memref of shape (1, 5)
    fsd ft3, 32(t0)                              # store double value to memref of shape (1, 5)
    csrrci zero, 1984, 1                         # SSR disable
    ret
```

### 2. Slightly More Complicated Case: assembly as plain text

```
.text
.globl matmul_transpose_b
.p2align 2
    # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "a2", "zero"], "allocated_float": ["ft0", "ft1", "ft3", "ft4", "ft5", "ft6"], "allocated_int": ["a0", "a1", "a2", "t0", "t1", "t2", "t3", "t4", "t5", "zero"]}
matmul_transpose_b:
    mv t3, a0
    mv t2, a1
    mv t0, a2
    li t1, 79
    scfgwi t1, 64                                # dm 0 dim 0 bound
    li t1, 1
    scfgwi t1, 96                                # dm 0 dim 1 bound
    li t1, 8
    scfgwi t1, 192                               # dm 0 dim 0 stride
    li t1, -632
    scfgwi t1, 224                               # dm 0 dim 1 stride
    li t1, 3
    scfgwi t1, 32                                # dm 0 repeat
    li t1, 3
    scfgwi t1, 65                                # dm 1 dim 0 bound
    li t1, 79
    scfgwi t1, 97                                # dm 1 dim 1 bound
    li t1, 1
    scfgwi t1, 129                               # dm 1 dim 2 bound
    li t1, 640
    scfgwi t1, 193                               # dm 1 dim 0 stride
    li t1, -1912
    scfgwi t1, 225                               # dm 1 dim 1 stride
    li t1, 8
    scfgwi t1, 257                               # dm 1 dim 2 stride
    scfgwi zero, 33                              # dm 1 repeat
    scfgwi t3, 800                               # dm 0 dim 1 source
    scfgwi t2, 833                               # dm 1 dim 2 source
    csrrsi zero, 1984, 1                         # SSR enable
    li t2, 2
    mv t1, zero
    # Constant folded riscv_cf.bge
scf_body_0_for:
    li t4, 4
    mul t4, t1, t4
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fld ft6, 0(t4)                               # load double from memref of shape (1, 8)
    li t4, 4
    mul t4, t1, t4
    addi t4, t4, 1
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fld ft5, 0(t4)                               # load double from memref of shape (1, 8)
    li t4, 4
    mul t4, t1, t4
    addi t4, t4, 2
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fld ft4, 0(t4)                               # load double from memref of shape (1, 8)
    li t4, 4
    mul t4, t1, t4
    addi t4, t4, 3
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fld ft3, 0(t4)                               # load double from memref of shape (1, 8)
    li t4, 79
    frep.o t4, 4, 0, 0
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    li t4, 4
    mul t4, t1, t4
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fsd ft6, 0(t4)                               # store double value to memref of shape (1, 8)
    li t4, 4
    mul t4, t1, t4
    addi t4, t4, 1
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fsd ft5, 0(t4)                               # store double value to memref of shape (1, 8)
    li t4, 4
    mul t4, t1, t4
    addi t4, t4, 2
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fsd ft4, 0(t4)                               # store double value to memref of shape (1, 8)
    li t4, 4
    mul t4, t1, t4
    addi t4, t4, 3
    li t5, 8
    mul t4, t4, t5                               # multiply by element size
    add t4, t0, t4
    fsd ft3, 0(t4)                               # store double value to memref of shape (1, 8)
    addi t1, t1, 1
    blt t1, t2, scf_body_0_for
scf_body_end_0_for:
    csrrci zero, 1984, 1                         # SSR disable
    ret
```

