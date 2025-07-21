# myrtle
tiling cost model for the snitch cluster!

## query Myrtle for a tile size

```
python3 myrtle.py <dispatchName> <mode> <output-tiles.json> <optional-bypass-gen.json>
```

where

- `<dispatchName>` is the name of the iree dispatch to tile, for example, `"main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"`
- `<mode>` is the tile size selection mode, either
  - `"sflt"` - simple filtering tile selection
  - `"scyc"` - simple cycle count predicted tile selection
  - `"svrcyc"` - SVR (support vector machine) cycle count predicted tile selection

- `<output-tiles.json>` full path to where myrtle should store its output

- `<optional-bypass-gen.json>` is a file containing a search space that you would like myrtle to use instead of its own.

### Example runs

```
python3 myrtle/myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" svrcyc test_output-disp-7.json
```

Bypassing tile search space generation:

```
python3 myrtle/myrtle.py "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64" svrcyc test_output-disp-1.json /home/hoppip/myrtle/sensitivity-analysis/holistic-data/dispatch_1_case1_everything.csv
```

```
python3 myrtle/myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" svrcyc test_output-disp-7.json /home/hoppip/myrtle/sensitivity-analysis/holistic-data/dispatch_7_case1_everything.csv
```

```
python3 myrtle/myrtle.py "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" svrcyc test_output-disp-8.json /home/hoppip/myrtle/sensitivity-analysis/holistic-data/dispatch_8_case1_everything.csv
```







