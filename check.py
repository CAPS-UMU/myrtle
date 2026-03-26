import pandas as pd
import subprocess
# implement overleaf forumla in the remainderTile tile analysis class
# check that it returns the same results for remainder tiles
# check its results for remainder tiles - HOW???

def check(g,c):
     golden = pd.read_csv(g)
     changed = pd.read_csv(c)
     # golden = golden.apply(pd.to_numeric, errors='coerce')
     # changed = changed.apply(pd.to_numeric, errors='coerce')

     fixedOrder=["FakeNN JSON Name", "M", "N", "K", "m", "n", "k", "JSON Name", "Little K", "m_tiles", "n_tiles", "k_tiles","tileA", "Space Needed in L1", "Space Remaining", "tileB", "tileB_cc", "Mpad", "Kpad", "Reduction Dim", "Row Dim", "m Dim", "tileC", "Original Name", "tileC_cc", "Weight Matrix Tile Size", "tileA_cc", "Npad", "padding", "SSR Config Count", "mPrime Little VecMat Runs", "mPrime UnrollAndJam Loop Iters", "mPrime HW Loop Iters", "mPrime HW Loop Body Size", "mPrime", "mHat Little VecMat Runs", "mHat UnrollAndJam Loop Iters", "mHat HW Loop Iters", "mHat HW Loop Body Size", "mHat", "Regular Loads", "L3 Loads","Total SSR Loads", "A Not Reused SSR Loads", "A SSR Reuse Loads", "A SSR Start Reuse Loads", "B SSR Loads"]
     if len(fixedOrder) != len(golden.columns):
          raise Exception("golden doesn't have same number of cols as fixed Order!!!")
     fixedOrder=["FakeNN JSON Name", "M", "N", "K", "m", "n", "k", "JSON Name", "Little K", "m_tiles", "n_tiles", "k_tiles","tileA", "Space Needed in L1", "Space Remaining", "tileB", "tileB_cc", "Mpad", "Kpad", "Reduction Dim", "Row Dim", "m Dim", "tileC", "Original Name", "tileC_cc", "Weight Matrix Tile Size", "tileA_cc", "Npad", "padding", "SSR Config Count", "mPrime Little VecMat Runs", "mPrime UnrollAndJam Loop Iters", "mPrime HW Loop Iters", "mPrime HW Loop Body Size", "mPrime", "mHat Little VecMat Runs", "mHat UnrollAndJam Loop Iters", "mHat HW Loop Iters", "mHat HW Loop Body Size", "mHat", "Regular Loads", "Total SSR Loads", "A Not Reused SSR Loads", "A SSR Reuse Loads", "A SSR Start Reuse Loads", "B SSR Loads"]
     golden[fixedOrder].to_csv("left.csv",index=False)
     changed[fixedOrder].to_csv("right.csv",index=False)
     return subprocess.call(['diff']+[ "left.csv", "right.csv"])

def main():
     goldenPath="128x128x128-no-redundant-stores/128x128x128wm-n-k_ss_c_ana.csv"
     changedPath="128x128x128-redundant-stores/128x128x128wm-n-k_ss_c_ana.csv"
     res = check(goldenPath,changedPath)
     if res != 0:
          print("ERROR!!!")
          print(f'{goldenPath} differs from {changedPath}')

     # goldenPath="myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"
     # changedPath="myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
     # res2 = check(goldenPath,changedPath)
     res2 = 0
     if res2 != 0:
          print("ERROR!!!")
          print(f'{goldenPath} differs from {changedPath}')
     if res == 0 and res2 == 0:
          print("OKAY.")

if __name__ == "__main__":
    main()