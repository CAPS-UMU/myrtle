This is a temporary directory containing copies of data from branch [absolute-ranking](https://github.com/CAPS-UMU/myrtle/tree/cf55ab18f6d65fa5b7afa332909a73432c0fbca0/sensitivity-analysis/beta%3D0/spm-reg/timed). Eventually, `absolute-ranking` will be merged into main with the latest data. 

This directory contains copies from August 5th, to get a sense of the difference in cycle count measured for matmul runs on the `padding` vs `myrtle` branches.

We used `scripts/cycleDiff.py` to compare the gaboost (branch off of the myrtle branch)'s cycle counts to the `padding` branch's cycle counts.

Comparisons show a difference of less than 0.5% from the padding branch's cycle counts. Complete results located in [cycleDiffResults.txt](cycleDiffResults.txt)