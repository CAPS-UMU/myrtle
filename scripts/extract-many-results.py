import sys
import re
import subprocess
import pandas as pd
# Generate folders containing search spaces and scripts for each M,N,K matmul dimension in the input text file.
def get_lines_from_file(file_name):
    """
    Opens a file, reads its contents, and returns a list of strings
    with the trailing newline characters removed.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            # .splitlines() is better than .readlines() because it 
            # automatically strips the '\n' from each string.
            return file.read().splitlines()
    except FileNotFoundError:
        raise Exception(f"Error: The file '{file_name}' was not found.")

# python extract-many-results.py outer-join-each.input 
def main():
    inputSizes = sys.argv[1]    
    lines = get_lines_from_file(inputSizes)
    expNameRegex = re.compile(
            r"(\d+)x(\d+)x(\d+)"
        )
    myresultsFldr="../myResults"
    timedResults=f"{myresultsFldr}/timed"
    untimedResults=f"{myresultsFldr}/untimed"
    # subprocess.call(['mkdir', myresultsFldr])
    # subprocess.call(['mkdir', timedResults])
    # subprocess.call(['mkdir', untimedResults])
    for line in lines:
        M_str, N_str, K_str = expNameRegex.search(line).groups()
        dims=f"{M_str}x{N_str}x{K_str}"
        outputFolder=f"../{dims}"
        prunedSuffix = "_ss_c_rem_div_ana_pruned"
        fixed = f"{dims}wm-n-k{prunedSuffix}"
        fixedSrc=f"{outputFolder}/{fixed}.csv"
        fixedDst=f"{untimedResults}/{fixed}.csv"
        fixedResultsSrc=f"{outputFolder}/{fixed}-results.csv"
        fixedResultsDst=f"{timedResults}/{fixed}-results.csv"
        # copy analyzed ss
        subprocess.call(['ls', fixedSrc])
        subprocess.call(['cp', fixedSrc,fixedDst])
        # copy timed ss
        subprocess.call(['ls', fixedResultsSrc])
        subprocess.call(['cp', fixedResultsSrc,fixedResultsDst])
        

        
if __name__ == "__main__":
    main()