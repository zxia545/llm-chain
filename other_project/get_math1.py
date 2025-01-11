import os
import json
import pandas as pd

# Function to process JSON files in the folder and combine results into a single row for each file
def process_json_files(base_path):
    combined_results = []

    for file_name in os.listdir(base_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(base_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Create a row with the file name and all key-value pairs
            row = {"File Name": file_name}
            row.update(data)  # Add all key-value pairs from JSON
            combined_results.append(row)

    # Convert combined results to a DataFrame
    df = pd.DataFrame(combined_results)

    # Save to a single Excel file
    output_path = os.path.join(base_path, "combined_results.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Combined results saved to {output_path}")

# Define the base path for your folder
base_folder = "/path/to/your/folder"
process_json_files(base_folder)
