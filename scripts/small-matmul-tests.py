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

# python small-matmul-tests.py small-matmul-tests.input
# python small-matmul-tests.py 125-matmul-tests.input
# python small-matmul-tests.py small-m-dim-matmul-tests.input
def main():
    inputSizes = sys.argv[1]    
    lines = get_lines_from_file(inputSizes)
    expNameRegex = re.compile(
            r"(\d+)x(\d+)x(\d+)"
        )  
    for line in lines:
        M_str, N_str, K_str = expNameRegex.search(line).groups()
        print(f'{M_str} {N_str} {K_str}')
        dims=f"{M_str}x{N_str}x{K_str}"
        inputSizesTxt=f"../{dims}/input.txt"
        outputFolder=f"../{dims}"
        subprocess.call(['rm', '-rf', outputFolder],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(['mkdir', outputFolder])
        f = open(inputSizesTxt, "w")
        print(dims, file=f)
        f.close()
        #"_ss_c_rem_div_ana_pr_sel_sflt"
        subprocess.call(["python", "topTenFromMNK.py", inputSizesTxt, outputFolder, "_ss_c_rem_div_ana_pr_sel_sflt", "0"])
        #subprocess.call(["python", "topTenFromMNK.py", inputSizesTxt, outputFolder, "_ss_c_rem_div_ana_pruned", "0"])
    f = open(f"{sys.argv[1]}.compile-and-run.sh", "w")
    # convenience script
    print("#!/bin/bash",file=f)
    for line in lines:
        print(f"cd ../{line}; bash compile.sh;",file=f)
    for line in lines:
        print(f"cd ../{line}; bash run.sh;",file=f)
    f.close()
if __name__ == "__main__":
    main()