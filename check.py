import pandas as pd
import subprocess
# implement overleaf forumla in the remainderTile tile analysis class
# check that it returns the same results for remainder tiles
# check its results for remainder tiles - HOW???

def check(g,c):
     golden = pd.read_csv(g)
     changed = pd.read_csv(c)

     golden[golden.columns].to_csv("left.csv",index=False)
     changed[golden.columns].to_csv("right.csv",index=False)
     return subprocess.call(['diff']+[ "left.csv", "right.csv"])

def main():
     goldenPath="128x128x128-no-redundant-stores/128x128x128wm-n-k_ss_c_ana.csv"
     changedPath="128x128x128-redundant-stores/128x128x128wm-n-k_ss_c_ana.csv"
     res = check(goldenPath,changedPath)
     goldenPath="myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"
     changedPath="myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
     res2 = check(goldenPath,changedPath)

     if res == 0 and res2 == 0:
          print("OKAY.")
     else:
          print("ERROR!!!")

if __name__ == "__main__":
    main()