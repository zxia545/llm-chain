import os
import json
import pandas as pd

# Function to process JSON files in the folder and combine results into one Excel file
def process_json_files(base_path):
    combined_results = []

    for file_name in os.listdir(base_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(base_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract key-value pairs and add the file name as a column
            for key, value in data.items():
                combined_results.append({"File Name": file_name, "Key": key, "Value": value})

    # Convert combined results to a DataFrame
    df = pd.DataFrame(combined_results)

    # Save to a single Excel file
    output_path = os.path.join(base_path, "combined_results.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Combined results saved to {output_path}")

# Define the base path for your folder
base_folder = "/path/to/your/folder"
process_json_files(base_folder)
