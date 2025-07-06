# myrtle
tiling cost model for the snitch cluster!

## query Myrtle for a tile size

```
python3 myrtle.py <dispatchName> <mode> <output-tiles.json>
```

where

- `<dispatchName>` is the name of the iree dispatch to tile, for example, `"main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"`
- `<mode>` is the tile size selection mode, either
  - `"sflt"` - simple filtering tile selection
  - `"scyc"` - simple cycle count predicted tile selection
  - `"svrcyc"` - SVR (support vector machine) cycle count predicted tile selection

- `<output-tiles.json>` full path to where myrtle should store its output

### Example runs

```
python3 myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" svrcyc test_output.json
```

