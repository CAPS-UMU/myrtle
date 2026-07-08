import pandas as pd
import sys

def main():
    inputName = sys.argv[1]
    outputName = sys.argv[2]
    df = pd.read_csv(inputName)
    print(f"Rows before removing timeouts: {len(df)}")
    df = df[df["Global Sim E2E_dma"] != -1]
    print(f"Rows after removing timeouts: {len(df)}")
    df_sorted = df.sort_values(by="Global Sim E2E_dma", ascending=True)
    df_sorted.to_csv(outputName, index=False)


if __name__ == "__main__":
    main()
