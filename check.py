import pandas as pd
import subprocess

goldenPath="/home/hoppip/myrtle/128x128x128-no-redundant-stores/128x128x128wm-n-k_ss_c_ana.csv"
changedPath="/home/hoppip/myrtle/128x128x128-redundant-stores/128x128x128wm-n-k_ss_c_ana.csv"

golden = pd.read_csv(goldenPath)
changed = pd.read_csv(changedPath)

golden[golden.columns].to_csv("left.csv",index=False)
changed[golden.columns].to_csv("right.csv",index=False)
res = subprocess.call(['diff']+[ "left.csv", "right.csv"])
if res == 0:
     print("OKAY.")
else:
     print("ERROR!!!")