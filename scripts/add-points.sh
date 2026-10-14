#!/usr/bin/env bash

USER="hoppip"
MYHOME="/home/$USER/"
MYMYRTLE="${MYHOME}myrtle/"
TIMED="${MYMYRTLE}sensitivity-analysis/beta=0/spm-reg/timed/"
TO_ADD_DIR="${MYMYRTLE}toAdd"
OUT_DIR="${MYMYRTLE}scripts/out"

mkdir -p "$OUT_DIR"

# Parse invocation argument:
# - If passed "aggressive", pass "aggressive" to concatCSVs.py.
# - If no argument is passed (or default), pass "polite" to concatCSVs.py.
CONCAT_MODE_ARG="polite"
if [[ "$1" == "aggressive" ]]; then
    CONCAT_MODE_ARG="aggressive"
fi

# List of experiment files
EXP_FILES=(
    "${TIMED}128x128x128-reg-SPM-results.csv"
    "${TIMED}256x256x256wm-n-k_ss_c_rem_div_ana-results-dma-only.csv"
    "${TIMED}192x384x384-bgeSmall-reg-SPM-results.csv"
    "${TIMED}128x768x768-roberta-reg-SPM-results.csv"
    "${TIMED}384x384x384wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
    "${TIMED}512x512x512wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
)

# Helper function to invoke concatCSVs.py from MYMYRTLE
run_concat() {
    local left="$1"
    local right="$2"
    local output="$3"

    (
        cd "$MYMYRTLE" || exit 1
        python scripts/concatCSVs.py "$left" "$right" "$output" "$CONCAT_MODE_ARG"
    )
}

for exp_file in "${EXP_FILES[@]}"; do
    filename="$(basename "$exp_file")"

    # Extract MxNxK prefix (e.g., "128x128x128")
    if [[ "$filename" =~ ^([0-9]+x[0-9]+x[0-9]+) ]]; then
        mnk="${BASH_REMATCH[1]}"
    else
        echo "Could not extract MxNxK prefix from $filename. Skipping."
        continue
    fi

    # Find unprocessed CSV files in toAdd matching the MxNxKw prefix
    # Matches starting with "already-added-" are naturally excluded
    matching_csvs=()
    shopt -s nullglob
    for f in "$TO_ADD_DIR/${mnk}w"*.csv; do
        [[ -f "$f" ]] && matching_csvs+=("$f")
    done
    shopt -u nullglob

    # If no unprocessed CSVs exist in toAdd, skip this experiment
    if [[ ${#matching_csvs[@]} -eq 0 ]]; then
        echo "[$mnk] No unprocessed CSVs found in $TO_ADD_DIR with prefix '${mnk}w'. Skipping."
        continue
    fi

    echo "[$mnk] Found ${#matching_csvs[@]} file(s) to process in $TO_ADD_DIR."

    all_output="$OUT_DIR/${mnk}-all.csv"
    concat_failed=0

    # Step 1: Concatenate all matching toAdd CSVs into MxNxK-all.csv
    if [[ ${#matching_csvs[@]} -eq 1 ]]; then
        cp "${matching_csvs[0]}" "$all_output"
    else
        current_left="${matching_csvs[0]}"
        temp_files=()

        for ((i = 1; i < ${#matching_csvs[@]}; i++)); do
            right="${matching_csvs[i]}"
            if (( i == ${#matching_csvs[@]} - 1 )); then
                step_out="$all_output"
            else
                step_out=$(mktemp "$OUT_DIR/tmp_${mnk}_XXXXXX.csv")
                temp_files+=("$step_out")
            fi

            if ! run_concat "$current_left" "$right" "$step_out"; then
                echo "[$mnk] Error: concatCSVs.py failed while creating $all_output."
                concat_failed=1
                break
            fi
            current_left="$step_out"
        done

        # Clean up intermediate step files
        for tmp in "${temp_files[@]}"; do
            rm -f "$tmp"
        done

        if (( concat_failed != 0 )); then
            continue
        fi
    fi

    # Step 2: Concatenate base EXP file with MxNxK-all.csv -> MxNxK-updated.csv
    updated_output="$OUT_DIR/${mnk}-updated.csv"
    if ! run_concat "$exp_file" "$all_output" "$updated_output"; then
        echo "[$mnk] Error: concatCSVs.py failed while generating $updated_output."
        continue
    fi

    # Step 3: Overwrite EXP file with updated CSV
    if cp "$updated_output" "$exp_file"; then
        echo "[$mnk] Successfully updated $exp_file."

        # Step 4: Rename processed toAdd files with 'already-added-' prefix
        for orig_file in "${matching_csvs[@]}"; do
            base_name="$(basename "$orig_file")"
            dir_name="$(dirname "$orig_file")"
            mv "$orig_file" "$dir_name/already-added-$base_name"
        done
        echo "[$mnk] Renamed ${#matching_csvs[@]} file(s) in $TO_ADD_DIR to 'already-added-*'."
    else
        echo "[$mnk] Error: Failed to overwrite $exp_file. Source files were not renamed."
    fi
done