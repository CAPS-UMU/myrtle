# How Square is Square Enough?

## Generating Search Spaces (top 10, sorted by L1 Usage most to least)

```
source .venv/bin/activate
```

### 128x128x128

```
python3 myrtle/myrtle.py "matmul_128x128x128_f64" sflt test_output-disp-7.json
```

Entire search space here: `myrtle/out/128x128x128wm-n-k_searchSpace_c_analyzed-myrtle-sflt-sorted-L1.csv`

manually trimmed to top 10

### 64x128x128

```
python3 myrtle/myrtle.py "matmul_64x128x128_f64" sflt test_output-disp-7.json
```

manually trimmed to top 10

### 32x128x128

```
python3 myrtle/myrtle.py "matmul_32x128x128_f64" sflt test_output-disp-7.json
```

manually trimmed to top 10

## Compiling all inside the snitch repo

Using script: `snitch_cluster/compile-three-top10-runs.sh`

```
bash many_gemms.sh 128x128x128wm-n-k_searchSpace_top10-sorted-L1.csv compile no no no > compile-128x128x128.txt; 
bash many_gemms.sh 64x128x128wm-n-k_searchSpace_top10-sorted-L1.csv compile no no no > compile-64x128x128.txt; 
bash many_gemms.sh 32x128x128wm-n-k_searchSpace_top10-sorted-L1.csv compile no no no > compile-32x128x128.txt; 
echo "DONE COMPILING"
```

## Running all inside the snitch repo

Using script: `snitch_cluster/run-three-top10-runs.sh`

```
bash many_gemms.sh 32x128x128wm-n-k_searchSpace_top10-sorted-L1.csv check run no no > run-32x128x128.txt; 
bash many_gemms.sh 64x128x128wm-n-k_searchSpace_top10-sorted-L1.csv check run no no > run-64x128x128.txt;
bash many_gemms.sh 128x128x128wm-n-k_searchSpace_top10-sorted-L1.csv check run no no > run-128x128x128.txt;
echo "Finished running three top 10 runs. It's a miracle."
```

## Extracting results inside snitch repo

```
bash many_gemms.sh 128x128x128wm-n-k_searchSpace_top10-sorted-L1.csv no no no extract; 
python combineKernelTimesIntoSingleCSV.py 128x128x128wm-n-k_searchSpace_top10-sorted-L1.csv;

bash many_gemms.sh 64x128x128wm-n-k_searchSpace_top10-sorted-L1.csv no no no extract;
python combineKernelTimesIntoSingleCSV.py 64x128x128wm-n-k_searchSpace_top10-sorted-L1.csv;

bash many_gemms.sh 32x128x128wm-n-k_searchSpace_top10-sorted-L1.csv no no no extract;
python combineKernelTimesIntoSingleCSV.py 32x128x128wm-n-k_searchSpace_top10-sorted-L1.csv;
```

## Let's do this again, but for a bunch more input sizes

```
python topTenFromMNK.py "inputSizes.txt" 
```
- [inputSizes.txt](inputSizes.txt)
- [topTenFomMNK.py](topTenFromMNK.py)

Search spaces csv files and accompanying compilation, run, and result extraction scripts will be stored in a folder called [top10](top10). Copy this folder to your snitch repo to run the experiments.

