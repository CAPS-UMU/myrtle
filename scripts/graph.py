import sys
from pathlib import Path
import subprocess

# python graph.py user="emily" fresh-ana output="hoodle"

# Default configurations
USER = "hoppip"
OUTPUT_DIR = "output"

# Parse CLI arguments
for arg in sys.argv[1:]:
    if arg.startswith("user="):
        val = arg.split("=", 1)[1].strip("'\"")
        if val:
            USER = val
    elif arg.startswith("output="):
        val = arg.split("=", 1)[1].strip("'\"")
        if val:
            OUTPUT_DIR = val

# Base directory hierarchy
HOME_DIR = f"/home/{USER}"
MYRTLE_DIR = f"{HOME_DIR}/myrtle"
SENSITIVITY_DIR = f"{MYRTLE_DIR}/sensitivity-analysis"
BETA_DIR = f"{SENSITIVITY_DIR}/beta=0"
SPM_REG_DIR = f"{BETA_DIR}/spm-reg"

# Search space subdirectories
TIMED_DIR = f"{SPM_REG_DIR}/timed"
DEFAULT_UNTIMED_DIR = f"{SPM_REG_DIR}/untimed"
DEFAULT_UNTIMED_TOO_BIG_DIR = f"{SPM_REG_DIR}/untimed/tooBig"
FRESH_ANA_DIR = f"{MYRTLE_DIR}/myrtle/out"

# Check CLI arguments for "fresh-ana"
if "fresh-ana" in sys.argv:
    UNTIMED_DIR = FRESH_ANA_DIR
else:
    UNTIMED_DIR = DEFAULT_UNTIMED_DIR

# Common file suffix patterns
ANA_SUFFIX = "wm-n-k_ss_c_rem_div_ana.csv"
FULL_SUFFIX = "wm-n-k_ss_c_rem_div.csv"
PRUNED_DMA_SUFFIX = "wm-n-k_rem_div_ana_pruned-results-dma-only.csv"

EXPERIMENTS = [
    "128x128x128",
    "256x256x256",
    "384x384x384",
    "512x512x512",
    "128x768x768",
    "192x384x384"
]

# Experiment-specific filenames for the timed search space
TIMED_FILES = {
    "128x128x128": f"{TIMED_DIR}/128x128x128-reg-SPM-results.csv",
    "256x256x256": f"{TIMED_DIR}/256x256x256wm-n-k_ss_c_rem_div_ana-results-dma-only.csv",
    "384x384x384": f"{TIMED_DIR}/384x384x384{PRUNED_DMA_SUFFIX}",
    "512x512x512": f"{TIMED_DIR}/512x512x512{PRUNED_DMA_SUFFIX}",
    "128x768x768": f"{TIMED_DIR}/128x768x768-roberta-reg-SPM-results.csv",
    "192x384x384": f"{TIMED_DIR}/192x384x384-bgeSmall-reg-SPM-results.csv",
}


def get_experiment_paths(exp: str) -> tuple[str, str, str]:
    """Returns (timed_path, analyzed_path, full_path) for a given experiment."""
    timed_path = TIMED_FILES[exp]
    analyzed_path = f"{DEFAULT_UNTIMED_TOO_BIG_DIR}/{exp}{ANA_SUFFIX}"
    full_path = analyzed_path #f"{UNTIMED_DIR}/{exp}{FULL_SUFFIX}"
    return timed_path, analyzed_path, full_path


def generate_input_file(output_filename: str = "experiments_input.txt") -> None:
    lines = []
    for exp in EXPERIMENTS:
        timed_path, analyzed_path, full_path = get_experiment_paths(exp)
        # Format: <experiment> <timed> <analyzed> <full> 0
        line = f"{exp} {timed_path} {analyzed_path} {full_path} 0"
        lines.append(line)

    output_content = "\n".join(lines) + "\n"

    Path(output_filename).write_text(output_content, encoding="utf-8")
    print(f"Generated '{output_filename}' (User: {USER}, untimed dir: {UNTIMED_DIR}):\n")
    print(output_content)


if __name__ == "__main__":
    input_file = "experiments_input.input"
    generate_input_file(input_file)
    subprocess.call(['bash', 'graph-exp-data-beta.sh', input_file, OUTPUT_DIR])