
import sys
import pandas as pd

from itertools import product, islice

import numpy as np

import re

def main():
    filename="/home/hoppip/recent_snitch/snitch_cluster/256x256x256wm-n-k_searchSpace_c_analyzed-no-top-5"
    filename="/home/hoppip/recent_snitch/snitch_cluster/512x768x768wm-n-k_searchSpace_c_analyzed-myrtle-sflt-ranking"
    filename="288x1024x1024wm-n-k_searchSpace_bge_small"
    filename="512x512x512wm-n-k_searchSpace_distillbert"
    filename="768x768x768wm-n-k_searchSpace_roberta"
    filename="384x384x384wm-n-k_searchSpace_miniLM"
    filename="/home/hoppip/myrtle/sensitivity-analysis/manual-c-backend/review-cube"
    file=f"{filename}.csv"
  
    print("hello")
    df = pd.read_csv(file)     # read in the first CSV
    lowIsGood=False
    #by = "SSR Config Count"
    by="Space Needed in L1"
    df_sorted = df.sort_values(by=by, ascending=lowIsGood)
    cols = list(df_sorted.columns)
    cols.remove("JSON Name")
    cols.append("JSON Name")
    df_reordered = df_sorted[cols]
    df_reordered.to_csv(
        f"{filename}-sorted.csv",
        index=False,
    )

if __name__ == "__main__":
    main()