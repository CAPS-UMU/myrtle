# myrtle
tiling cost model for the snitch cluster!

## Query Myrtle for a tile size

```
python3 myrtle.py <kernel> <mode> <output-tiles.json> <optional-bypass-gen.json>
```

where

- `<kernel>` is a string representing the **type of kernel** and its **input sizes**, formatted differently depending on the backend selected.
  - Quidditch: use the name of the iree dispatch to tile, for example, `"main\$async_dispatch_9_matmul_transpose_b_1x161x600_f64"`
  - Manual C code: use the string format `matmul_MxNxK_f64` where `M`, `N`, and `K`, are input dimensions, for ex, `"matmul_2x768x760_f64"`
- `<mode>` is the tile size selection mode, either
  - `"sflt"` - simple filtering tile selection
  - ~~`"scyc"` - simple cycle count predicted tile selection~~ (deprecated)
  - ~~`"svrcyc"` - SVR (support vector machine) cycle count predicted tile selection~~ (deprecated)
- `<output-tiles.json>` full path to where myrtle should store its output
- `<optional-bypass-gen.json>` is a file containing a search space that you would like myrtle to use instead of its own.

### Example runs

#### Manual C Code Backend

```
python3 myrtle/myrtle.py "matmul_2x768x760_f64" sflt test_output-disp-7.json
```

#### Quidditch Backend

```
python3 myrtle/myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" sflt test_output-disp-7.json
```

Bypassing tile search space generation:

```
python3 myrtle/myrtle.py "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64" sflt test_output-disp-1.json /home/hoppip/myrtle/sensitivity-analysis/holistic-data/dispatch_1_case1_everything.csv
```

```
python3 myrtle/myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" sflt test_output-disp-7.json /home/hoppip/myrtle/sensitivity-analysis/holistic-data/dispatch_7_case1_everything.csv
```

```
python3 myrtle/myrtle.py "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" sflt test_output-disp-8.json /home/hoppip/myrtle/sensitivity-analysis/holistic-data/dispatch_8_case1_everything.csv
```







