import pandas as pd

new = "1x400x200wm-n-k_searchSpace.csv"
old = "1x400x200wm-n-k_case1_searchSpaceOLD.csv"


df_new = pd.read_csv(new)
df_old = pd.read_csv(old)

print("This script proves the static analysis changes added for matmul")
print("Do NOT change previous vec mat results (regression test)")

act = pd.read_csv(new)  # read in the first CSV
pred = pd.read_csv(old)  # read in the second CSV

print(act.columns)
print(pred.columns)
print(act[["JSON Name","Regular Loads","HW Loop Iters"]])
print(pred[["JSON Name","Regular Loads","HW Loop Iters"]])
# combine the CSVs, only bringing in observed fields we care about
pred = pd.merge(
    pred,
    act[
        [
            "JSON Name",
            "SSR Config Count",
            "Little VecMat Runs",
            "UnrollAndJam Loop Iters",
            "HW Loop Body Size",
            "Total SSR Loads",
            "A SSR Reuse Loads",
            "A SSR Start Reuse Loads",
            "B SSR Loads",
            "Little N Prime",
            "Little K",
        ]
    ],
    on="JSON Name",
    how="inner",
)


new_cols = [
    "Total SSR Loads",
    "B SSR Loads",
    "A SSR Start Reuse Loads",
    "A SSR Reuse Loads",
    "UnrollAndJam Loop Iters",
    "HW Loop Body Size",
    "SSR Config Count",  
    "Little N Prime",
    "Little K",
]
old_cols = [
    "Total Streaming Loads",
    "Other Streaming Loads",
    "Start Reuse Streaming Loads",
    "Reused Streaming Loads",
    "Outer Loop Iters",
    "HW Loop Body",
    "Microkernel Count",
    "Microkernel Row Dim",
    "Microkernel Reduction Dim",
]

for (old_c,new_c) in zip(old_cols,new_cols):
    print(pred[["JSON Name", old_c, new_c]])

# cmp = pd.merge(old[["JSON Name","Streaming Loads"]], new[["JSON Name","Total SSR Loads"]],on="JSON Name",how="inner")
# print(cmp)
