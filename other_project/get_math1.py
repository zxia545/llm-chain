import os
import json
import pandas as pd

# Function to process JSON files in the folder
def process_json_files(base_path):
    for file_name in os.listdir(base_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(base_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Convert JSON keys and values to a DataFrame
            df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])

            # Save to an Excel file with the same name as the JSON file
            excel_file_name = os.path.splitext(file_name)[0] + '.xlsx'
            output_path = os.path.join(base_path, excel_file_name)
            df.to_excel(output_path, index=False)
            print(f"Processed {file_name} and saved to {excel_file_name}")

# Define the base path for your folder
base_folder = "/path/to/your/folder"
process_json_files(base_folder)
