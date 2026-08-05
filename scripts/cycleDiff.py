import pandas as pd

# Load the CSV files
file_a = '/home/emily/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/192x384x384-bgeSmall-reg-SPM-results.csv'
file_b = '/home/emily/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/gaboost/192x384x384-bgeSmall-reg-SPM-results-GABoost.csv'

file_a = '/home/emily/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/128x128x128-reg-SPM-results.csv'
file_b = '/home/emily/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/gaboost/128x128x128-reg-SPM-results-GABoost.csv'
# file_a = '/home/emily/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/128x768x768-roberta-reg-SPM-results.csv'
# file_b = '/home/emily/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/gaboost/128x768x768-roberta-reg-SPM-results-GABoost.csv'

df_a = pd.read_csv(file_a)
df_b = pd.read_csv(file_b)

key_col = 'FakeNN JSON Name'
metric_col = 'Global Sim E2E_dma'

# Convert metric columns to numeric (coercing invalid non-numeric values to NaN)
df_a[metric_col] = pd.to_numeric(df_a[metric_col], errors='coerce')
df_b[metric_col] = pd.to_numeric(df_b[metric_col], errors='coerce')

# 1. Filter out rows where "Global Sim E2E_dma" equals -1
df_a = df_a[df_a[metric_col] != -1]
df_b = df_b[df_b[metric_col] != -1]

# 2. Count how many valid rows from A have matching "FakeNN JSON Name" values in B
matching_a_rows = df_a[key_col].isin(df_b[key_col]).sum()
print(f"Number of valid rows from A present in B (excluding -1 values): {matching_a_rows}\n")

# 3. Merge common rows between A and B on the key column
merged_df = pd.merge(
    df_a[[key_col, metric_col]], 
    df_b[[key_col, metric_col]], 
    on=key_col, 
    suffixes=('_A', '_B')
)

# 4. Calculate actual differences (B - A) and percentage differences relative to file A
merged_df['Diff'] = merged_df[f'{metric_col}_B'] - merged_df[f'{metric_col}_A']
merged_df['Abs_Diff'] = merged_df['Diff'].abs()

merged_df['Pct_Diff (%)'] = (merged_df['Diff'] / merged_df[f'{metric_col}_A']) * 100
merged_df['Abs_Pct_Diff (%)'] = merged_df['Pct_Diff (%)'].abs()

# 5. Output row-by-row comparison
print("Comparison for common rows:")
cols_to_display = [
    key_col, 
    f'{metric_col}_A', 
    f'{metric_col}_B', 
    'Diff', 
    'Pct_Diff (%)', 
    'Abs_Pct_Diff (%)'
]
print(merged_df[cols_to_display].to_string(index=False))

# 6. Calculate and print summary statistics
mean_diff = merged_df['Diff'].mean()
mean_abs_diff = merged_df['Abs_Diff'].mean()
mean_pct_diff = merged_df['Pct_Diff (%)'].mean()
mean_abs_pct_diff = merged_df['Abs_Pct_Diff (%)'].mean()

print("\n--- Summary ---")
print(f"Average Difference (B - A): {mean_diff:.2f}")
print(f"Average Absolute Difference: {mean_abs_diff:.2f}")
print(f"Average Percentage Difference: {mean_pct_diff:.2f}%")
print(f"Average Absolute Percentage Difference: {mean_abs_pct_diff:.2f}%")