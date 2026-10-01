import pandas as pd
import pathlib

fullSS="/home/emily/myrtle/scripts/out/expandedAnns/128x128x128_full.csv"
timedSS="/home/emily/myrtle/scripts/out/expandedAnns/128x128x128_timed.csv"

full=pd.read_csv(fullSS)
timed = pd.read_csv(timedSS)

untimed = full[~full["FakeNN JSON Name"].isin(timed["FakeNN JSON Name"])]
busyCores = untimed[untimed["m"]%8 == 0]
print(f"full: {full.shape}")
print(f"timed: {timed.shape}")
print(f"untimed: {untimed.shape}")
print(f"busyCores: {busyCores.shape}")
busyCores.sort_values("SSR Configs",ascending = False)
#rm_utPruned.to_csv("./out/remaindertilesWithFewerThan1024.csv", index=False)
filename = f"{pathlib.Path(__file__).parent.resolve()}/out/128x128x128-nearly-exhaustive.csv"
busyCores.to_csv(filename, index=False)