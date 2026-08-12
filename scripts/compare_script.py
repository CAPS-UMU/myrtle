import argparse
import sys
import pandas as pd

# example:
# python compare_script.py "/home/emily/myrtle/16x16x16 copy/16x16x16wm-n-k_ss_c_rem_div_ana_pruned.csv" "/home/emily/myrtle/16x16x16/16x16x16wm-n-k_ss_c_rem_div_ana_pruned.csv" 
# python compare_script.py "/home/emily/myrtle/16x16x16 copy/16x16x16wm-n-k_ss_c_rem_div_ana.csv" "/home/emily/myrtle/16x16x16/16x16x16wm-n-k_ss_c_rem_div_ana.csv" 


def compare_myrtle_outputs(old_csv_path: str, new_csv_path: str) -> bool:
    """
    Compares two CSV files output by myrtle for equivalence or column subset relationship.
    
    Equivalence / Subset conditions:
    1. Same set of unique IDs in "FakeNN JSON Name"
    2. Identical cell values for all shared columns and corresponding rows
    """
    # 1. Load CSVs into DataFrames
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)

    id_col = "FakeNN JSON Name"

    # Ensure key column exists in both files
    if id_col not in df_old.columns or id_col not in df_new.columns:
        print(f"Error: Missing required column '{id_col}' in one or both files.")
        return False

    # 2. Check Row IDs first
    old_ids = set(df_old[id_col])
    new_ids = set(df_new[id_col])

    if old_ids != new_ids:
        print(f"Mismatch in '{id_col}' row IDs:")
        print(f"  IDs in Old but not New: {old_ids - new_ids}")
        print(f"  IDs in New but not Old: {new_ids - old_ids}")
        return False

    # Set unique ID as index to align rows
    df_old = df_old.set_index(id_col)
    df_new = df_new.set_index(id_col)

    # 3. Check Column Relationships
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)

    cols_match = old_cols == new_cols
    old_is_subset = old_cols < new_cols  # Strict subset
    new_is_subset = new_cols < old_cols  # Strict subset

    if not (cols_match or old_is_subset or new_is_subset):
        print("Mismatch in columns (neither is a subset of the other):")
        print(f"  Columns only in Old: {old_cols - new_cols}")
        print(f"  Columns only in New: {new_cols - old_cols}")
        return False

    # Find the shared set of columns to compare data content
    shared_cols = sorted(list(old_cols & new_cols))

    # Align row ordering and select shared columns
    sorted_index = sorted(df_old.index)
    df_old_shared = df_old.reindex(index=sorted_index, columns=shared_cols)
    df_new_shared = df_new.reindex(index=sorted_index, columns=shared_cols)

    # 4. Compare Content Equality across Shared Columns
    if df_old_shared.equals(df_new_shared):
        if cols_match:
            print("SUCCESS: CSV outputs are completely equivalent!")
            return True
        elif old_is_subset:
            print("ALERT: Column subset relationship detected!")
            print(f"  The OLD CSV is a column subset of the NEW CSV.")
            print(f"  Shared data matches, but the NEW CSV contains {len(new_cols - old_cols)} additional column(s): {new_cols - old_cols}")
            return False
        elif new_is_subset:
            print("ALERT: Column subset relationship detected!")
            print(f"  The NEW CSV is a column subset of the OLD CSV.")
            print(f"  Shared data matches, but the OLD CSV contains {len(old_cols - new_cols)} additional column(s): {old_cols - new_cols}")
            return False
    else:
        print("FAILURE: Cell values differ in shared columns.")
        diff_mask = (df_old_shared != df_new_shared) & ~(df_old_shared.isna() & df_new_shared.isna())
        diff_locations = diff_mask.stack()[lambda x: x].index.tolist()
        print(f"Found differences in {len(diff_locations)} cell(s). First few (Row ID, Column):")
        for loc in diff_locations[:5]:
            print(f"  - ID: {loc[0]} | Column: '{loc[1]}'")
            print(f"    Old: {df_old_shared.loc[loc[0], loc[1]]}")
            print(f"    New: {df_new_shared.loc[loc[0], loc[1]]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Myrtle CSV output files for row/column content equivalence."
    )
    parser.add_argument("old_csv", type=str, help="Path to the reference/old CSV file")
    parser.add_argument("new_csv", type=str, help="Path to the updated/new CSV file")

    args = parser.parse_args()

    # Perform comparison
    is_equivalent = compare_myrtle_outputs(args.old_csv, args.new_csv)

    # Exit with code 0 if fully equivalent, or 1 if mismatched / subset
    sys.exit(0 if is_equivalent else 1)


if __name__ == "__main__":
    main()