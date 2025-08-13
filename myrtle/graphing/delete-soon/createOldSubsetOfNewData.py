import pandas as pd
import sys
# clear;python kernelTimeVsTotalTime.py 

user="emily"
user="hoppip"

args = sys.argv[1:]
print(f'old: {args[0]} new: {args[1]}')
old = args[0]
new = args[1]
old ="/home/emily/myrtle/sensitivity-analysis/holistic-data/dispatch_7_case1_everything.csv"
old="/home/emily/myrtle/sensitivity-analysis/holistic-data/dispatch_8_case1_everything.csv"
basename="csv_experiment_results-right-600x400"
basename='csv_experiment_results-probably-right-600x600'
new=f'/home/emily/myrtle/sensitivity-analysis/holistic-data/{basename}.csv'
dfo = pd.read_csv(old)
# dfo = dfo.loc[:, dfo.columns != 'Kernel Time' and dfo.columns != 'Total Time']
dfn = pd.read_csv(new)

subset = pd.merge(dfo[["JSON Name"]],dfn,on="JSON Name",how="left")
subset.to_csv(
            f'/home/emily/myrtle/sensitivity-analysis/holistic-data/{basename}-old-subset.csv',
            index=False,
        )
            