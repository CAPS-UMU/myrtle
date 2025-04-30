# myrtle
tiling cost model for the [snitch cluster](https://ieeexplore.ieee.org/document/9216552) *and beyond!*

### setup

Create your virtual environment:

```
python3 -m venv myenv
```

Activate it:

```
source myenv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

You're off to the races!

## Examples: Quidditch "Dispatches"

Each of these "dispatches"

- Consists of `linalg_matmul_transpose_b` with type `<MxK>, <NxK> -> <MxN>` where M = 1 (only matrix-vector operations)

- Followed by an element-wise addition
- Each dispatch is really a fusion of two NsNet Kernels

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

   

