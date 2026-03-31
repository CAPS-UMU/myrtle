import sys
import re
import subprocess
import pandas as pd

# EXAMPLE USAGE:
# clear;python strip-many-dfs.py /home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/

# Source - https://stackoverflow.com/a/3777308
# Posted by Manoj Govindan, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-03, License - CC BY-SA 4.0
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
        return f"Error: The file '{file_name}' was not found."

def main():
    inputFolder= sys.argv[1]
    f = open("temp.txt", "w")
    subprocess.call(['ls', inputFolder],stdout=f)
    f.close()
    expNameRegex = re.compile(
            r"(\d+)x(\d+)x(\d+)(.*)(.csv)"
        )
    lines = get_lines_from_file("temp.txt")
    for line in lines:
        try:
            M_str, N_str, K_str, nada, nadanada= expNameRegex.search(line).groups()
            # kernelName=f"matmul_{M_str}x{N_str}x{K_str}_f64"
            # print(kernelName)
            filePath=f"{inputFolder}/{line}"
            subprocess.call(['python', "strip-df-to-only-timed-data.py",filePath, ""])
        except AttributeError:
            print(f"skipping {line}")
    subprocess.call(["rm","-rf","temp.txt"])

if __name__ == "__main__":
    main()