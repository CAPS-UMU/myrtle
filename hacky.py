import pandas as pd
import plotly.express as px

# 1. Load your datasets
# Replace 'file1.csv' and 'file2.csv' with your actual filenames
# df1 = pd.read_csv('/home/hoppip/myrtle/sensitivity-analysis/manual-c-backend/128x128x128wm-n-k_top10_l1.csv')
# df2 = pd.read_csv('/home/hoppip/myrtle/sensitivity-analysis/no-redundant-stores/128x128x128/128x128x128wm-n-k_top10_c_L1-results.csv')
df1=pd.read_csv('/home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/128x128x128wm-n-k_ss_c_ana-results-redundant.csv')
df2 = pd.read_csv('/home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv')
# add rank to each
df1_sorted = df1.sort_values(by="Kernel Time", ascending=True)
df1_sorted["absoluteRank"] = range(1, int(df1_sorted.shape[0] + 1))
df2_sorted = df2.sort_values(by="Kernel Time", ascending=True)
df2_sorted["absoluteRank"] = range(1, int(df2_sorted.shape[0] + 1))
# 2. Merge the data
# We use 'on' for the common column (e.g., 'Date') 
# 'how=inner' ensures we only plot points that exist in both files
df_merged = pd.merge(df1_sorted, df2_sorted, on='FakeNN JSON Name', suffixes=('_RedundantStores', '_NoRedundantStores'))
df_merged['Difference'] = df_merged['Kernel Time_RedundantStores'] - df_merged['Kernel Time_NoRedundantStores']
df_merged['% Change'] = (df_merged['Difference'] / df_merged['Kernel Time_NoRedundantStores']) * 100
df_merged['RankDiff'] = df_merged['absoluteRank_RedundantStores'] - df_merged['absoluteRank_NoRedundantStores']
df_merged['Status'] = df_merged['Difference'].apply(
    lambda x: 'Slow Down' if x < 0 else 'Same or Better'
)

df_merged.sort_values(by="Kernel Time_RedundantStores")
# 3. Create the interactive figure
# Here we compare 'Sales_File1' against 'Sales_File2' over 'Date'
fig1 = px.scatter(df_merged, 
              x='FakeNN JSON Name', 
              y=['Kernel Time_RedundantStores', 'Kernel Time_NoRedundantStores'],
              symbol='Status',  # Use the column we created above
              symbol_sequence=['circle', 'x'], # Normal = circle, High = x
              title='Kernel Time With and Without Redundant Stores',
              hover_name='FakeNN JSON Name',
              hover_data={
                 'RankDiff':True,
                 'absoluteRank_RedundantStores': True,         # Show the raw value from File A
                 'absoluteRank_NoRedundantStores': True,         # Show the raw value from File B
             },
              labels={'value': 'Kernel Time (cycles)', 'variable': 'Source File'})

# 4. Enhance the layout (Optional)
fig1.update_layout(hovermode='x unified')

# 5. Export to an interactive HTML file
fig1.write_html("L3_stores_comparison_scatter.html")
print("Success! Your interactive graph is saved as 'L3_stores_comparison_scatter.html'.")


# 4. Create the Bar Graph
# We can use color to show if the difference is positive or negative
fig2 = px.bar(df_merged, 
             x='FakeNN JSON Name', 
             y='Difference',
             title='Speedup from Removing Redundant Stores',
             color='Difference',
             color_continuous_scale='RdBu', # Red for negative, Blue for positive
             hover_name='FakeNN JSON Name',
             hover_data={
                 'Difference': ':.2f',    # Format to 2 decimal places
                 '% Change': ':.1f',      # Format to 1 decimal place
                 'Kernel Time_RedundantStores': True,         # Show the raw value from File A
                 'Kernel Time_NoRedundantStores': True,         # Show the raw value from File B
                 'RankDiff':True,
                 'absoluteRank_RedundantStores': True,         # Show the raw value from File A
                 'absoluteRank_NoRedundantStores': True,         # Show the raw value from File B
                 'FakeNN JSON Name': False       # Hide Category if it's already on the X-axis
             },
            labels={'Difference': 'Redundant - No Redundant (cycles)','value': 'Kernel Time (cycles)', 'variable': 'Source File'})

# 5. Add a horizontal line at 0 for clarity
fig2.add_hline(y=0, line_dash="dash", line_color="black")

# 6. Save as HTML
fig2.write_html("L3_stores_difference_analysis.html")

print("Success! Open 'L3_stores_difference_analysis.html' to see the comparison.")


# 4. Create the Bar Graph
# We can use color to show if the difference is positive or negative
fig3 = px.bar(df_merged, 
             x='FakeNN JSON Name', 
             y='RankDiff',
             title='Rank Difference after Removing Redundant Stores',
             color='RankDiff',
             color_continuous_scale='RdBu', # Red for negative, Blue for positive
             hover_name='FakeNN JSON Name',
             hover_data={
                 'RankDiff': ':.2f',    # Format to 2 decimal places
                 'absoluteRank_RedundantStores': True,         # Show the raw value from File A
                 'absoluteRank_NoRedundantStores': True,         # Show the raw value from File B
                 'FakeNN JSON Name': False       # Hide Category if it's already on the X-axis
             },
            labels={'RankDiff': 'Rank Difference','value': 'Rank (1 is fastest)', 'variable': 'Source File'})

# 5. Add a horizontal line at 0 for clarity
fig3.add_hline(y=0, line_dash="dash", line_color="black")

# 6. Save as HTML
fig3.write_html("L3_stores_RankDiff_analysis.html")
print("Success! Open 'L3_stores_RankDiff_analysis.html' to see the comparison.")

with open('redundant_vs_no_redundant_stores_report.html', 'w') as f:
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))

print("Placed all three graphs in a single file here: 'redundant_vs_no_redundant_stores_report.html' :D")