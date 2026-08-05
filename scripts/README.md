# Utility Scripts for Using Myrtle + Graphing Data

### Create Tiling Scheme experiment to run on the Snitch Cluster

1. make a folder and input.txt containing the input matmul dimensions, for example 16x16x16
2. generate scripts to compile and run this matmul on the snitch cluster using the `createExperiment.py` script

```
cd scripts
mkdir ../16x16x16
echo "beta=0" > ../16x16x16/input.txt; echo "16x16x16" >> ../16x16x16/input.txt
python createExperiment.py "../16x16x16/input.txt" "../16x16x16" "_ss_c_rem_div_ana_pruned"
```

Instead of "_ss_c_rem_div_ana_pruned", you can use "all" to query myrtle for the full search space instead of a pruned one.

3. Copy the folder and all of its contents to the top level of the snitch repo directory

4. Modify the search space file (remove rows as desired) to make sure you only compile and run the tiling schemes you want

5. Compile and run from inside the `16x16x16` experiment folder*

   ```
   bash compile.sh
   bash run.sh
   bash extract.sh
   ```

   * make sure to set this environment variable before running the scripts!
     ```
     export gemmDir="/repo/sw/kernels/blas/gemm_boundary"

### Graph many data sets at once

1. Make an `.input file` with dimensions of + further info about each data set on a new line, formatted as follows:

   ```
   <MxNxK> <path-to-timed-data.csv> <path-to-analyzed-search-space.csv> <path-to-full-search-space.csv> <beta>
   ...
   <MxNxK> <path-to-timed-data.csv> <path-to-analyzed-search-space.csv> <path-to-full-search-space.csv> <beta>
   ```

   For example:

   ```
   384x384x384 /home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/both/timed/384x384x384-miniLM-results.csv /home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/both/untimed/ann-to-min-third-ssr-configs/384x384x384wm-n-k_ss_c_rem_div_ana_pruned.csv /home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/both/untimed/full/384x384x384wm-n-k_ss_c_rem_div.csv 0
   128x128x128 /home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/both/timed/128x128x128-bertTiny-results.csv /home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/both/untimed/ann-to-min-third-ssr-configs/128x128x128wm-n-k_ss_c_rem_div_ana_pruned.csv /home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/both/untimed/full/128x128x128wm-n-k_ss_c_rem_div.csv 0
   ```

2. Navigate to the `scripts` directory and launch the `graph-exp-data-beta.sh` script with the input file and desired output folder name. The output folder must be located inside the scripts directory.
   ```
   bash graph-exp-data-beta.sh dims-csv-name-line-by-line-expanded-SS.input expanded-SS
   ```

3. Copy this entire output folder to the `web` folder to view online from myrtle's github website.

### Most up to date graphs

We use input file `scripts/dims-csv-name-line-by-line-expanded-SS.input` and the bash script `scripts/graph-exp-data-beta.sh`.

Example graphing run:

```
clear;bash graph-exp-data-beta.sh dims-csv-name-line-by-line-expanded-SS.input expanded-SS-db
```

