import pandas as pd
# clear;python kernelTimeVsTotalTime.py 

user="emily"
user="hoppip"

def rankEm(csvFile, nickname):
    df = pd.read_csv(csvFile)
    print(f'{nickname} 2nd run times.........')
    print("sorted by kernel time (fastest)")
    ranked = df.sort_values("Kernel Time", ascending=True)
    print(ranked)
    print("sorted by TOTAL time (fastest)")
    ranked = df.sort_values("Total Time", ascending=True)
    print(ranked)


csvFile = f'/home/{user}/myrtle/sensitivity-analysis/holistic-data/1x600x400wm-n-k-timed.csv'
nickname='600x400'
rankEm(csvFile,nickname)
print()
csvFile =f'/home/{user}/myrtle/sensitivity-analysis/holistic-data/1x600x400wm-n-k-timed-debug.csv'
nickname='600x400-DEBUG'
rankEm(csvFile,nickname)
print()
print()
csvFile =f'/home/{user}/myrtle/sensitivity-analysis/holistic-data/1x600x600wm-n-k-timed.csv'
nickname='600x600'
rankEm(csvFile,nickname)
print()
csvFile =f'/home/{user}/myrtle/sensitivity-analysis/holistic-data/1x600x600wm-n-k-timed-debug.csv'
nickname='600x600-DEBUG'
rankEm(csvFile,nickname)
print()