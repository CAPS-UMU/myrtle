import pandas as pd
import sys

def main():
    inputName = sys.argv[1]
    outputName = sys.argv[2]
    df = pd.read_csv(inputName)
    df_sorted = df.sort_values(
        by="Global Sim E2E_dma",
        ascending=True,
        key=lambda col: col.mask(col < 0, float("inf")),
    )
    df_sorted.to_csv(outputName, index=False)


if __name__ == "__main__":
    main()
