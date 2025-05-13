# myrtle
tiling cost model for the [snitch cluster](https://ieeexplore.ieee.org/document/9216552) *and beyond!*

### new setup

```
uv venv
```

```
source .venv/bin/activate
```

```
uv sync
```

now you can run commands as normal!

To activate:

```
source .venv/bin/activate
```

To deactivate:

```
deactivate
```

Run commands in environment without activating the environment: `uv + command`, for example,

```
uv marimo edit docs/marimo
```

## Examples: Quidditch "Dispatches"

Each of these "dispatches"

- Consists of `linalg_matmul_transpose_b` with type `<MxK>, <NxK> -> <MxN>` where M = 1 (only matrix-vector operations)
- Followed by an element-wise addition

1. Generate tile size search space for Quidditch dispatch `main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64` with
   ```
   python3 tile_size_gen.py "dispatch_1" 400 1200
   ```

2. Generate tile size search space for Quidditch dispatch `main$async_dispatch_7_matmul_transpose_b_1x600x400_f64` with

   ```
   python3 tile_size_gen.py "dispatch_7" 600 400
   ```

3. Generate tile size search space for Quidditch dispatch `main$async_dispatch_8_matmul_transpose_b_1x600x600_f64` with

   ```
   python3 tile_size_gen.py "dispatch_8" 600 600
   ```

   

## Combining Static Analysis with empirical data

## Troubleshooting
1. ```
   UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
   ```
   Solution: Add `PyQt6` to `pyproject.toml`.