import sys
import re
import subprocess
import pandas as pd
import os


# python strip-df-to-only-timed-data.py absolute-path-to-file-to-strip.csv suffix
# Example Usage:
# clear;python strip-df-to-only-timed-data.py /home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv only-time

def main():
    inputFile = sys.argv[1]
    suffix=sys.argv[2]
    inputBaseName = os.path.basename(inputFile)[0:-4]
    loc=inputFile[0:len(inputFile)-len(inputBaseName)-4]
    outputFolder = f"{loc}/only-time"
    outputPath = f"{outputFolder}/{inputBaseName}.csv"
    df = pd.read_csv(inputFile)
    importantCols = ["FakeNN JSON Name","core0","core1","core2","core3","core4","core5","core6","core7","dma","Kernel Time"]
    stripped = df[importantCols]
    stripped.to_csv(outputPath,index=False)
    print("\tCSV Stripping:",end="\n\t")
    print(f"read in csv {inputBaseName}.csv",end="\n\t")
    print(f"wrote stripped csv to {outputPath}")
   

if __name__ == "__main__":
    main()