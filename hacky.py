import pandas as pd
import plotly.express as px

# 1. Load your datasets
# Replace 'file1.csv' and 'file2.csv' with your actual filenames
df1 = pd.read_csv('/home/hoppip/myrtle/sensitivity-analysis/manual-c-backend/128x128x128wm-n-k_top10_l1.csv')
df2 = pd.read_csv('/home/hoppip/myrtle/sensitivity-analysis/no-redundant-stores/128x128x128/128x128x128wm-n-k_top10_c_L1-results.csv')

# 2. Merge the data
# We use 'on' for the common column (e.g., 'Date') 
# 'how=inner' ensures we only plot points that exist in both files
df_merged = pd.merge(df1, df2, on='FakeNN JSON Name', suffixes=('_RedundantStores', '_NoRedundantStores'))

# 3. Create the interactive figure
# Here we compare 'Sales_File1' against 'Sales_File2' over 'Date'
fig = px.scatter(df_merged, 
              x='FakeNN JSON Name', 
              y=['Kernel Time_RedundantStores', 'Kernel Time_NoRedundantStores'],
              title='Comparison of Data Sets',
              labels={'value': 'Kernel Time (cycles)', 'variable': 'Source File'})

# 4. Enhance the layout (Optional)
fig.update_layout(hovermode='x unified')

# 5. Export to an interactive HTML file
fig.write_html("L3_stores_comparison.html")

print("Success! Your interactive graph is saved as 'L3_stores_comparison.html'.")