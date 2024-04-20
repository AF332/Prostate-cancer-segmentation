import pandas as pd
import plotly.express as px

# Load the spreadsheet data
file_path = r"F:\Results\Results.xlsx"
data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets to find the correct one

# Display the names of the sheets and the first few rows of each to identify the right one
sheet_overview = {sheet_name: data[sheet_name].head() for sheet_name in data.keys()}
print(sheet_overview)

# Extract data from 'Model 1' sheet
model_1_data = data['Best Model']

# Create a scatter plot with Plotly
fig = px.line(model_1_data, x='Threshold Value', y='IoU Score', title='Threshold vs IoU Score for Model 6',
                 labels={"Threshold": "Threshold", "IoU Score": "IoU Score"},
                 markers=True,
                 template="plotly_white")
fig.show()
fig.write_image("Threshold_vs_IoU_Score.png")