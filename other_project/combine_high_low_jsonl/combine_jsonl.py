import json

# Function to combine multiple JSONL files into a single JSONL file
def combine_jsonl(input_jsonl_files, output_jsonl):
    combined_data = []

    # Iterate over each file in the input list
    for file_path in input_jsonl_files:
        with open(file_path, 'r') as file:
            combined_data.extend([json.loads(line) for line in file])

    # Write combined data to the output file
    with open(output_jsonl, 'w') as output_file:
        for item in combined_data:
            output_file.write(json.dumps(item) + '\n')

    print(f"Combined JSONL file saved as: {output_jsonl}")

# Example usage
# Replace with your actual list of JSONL file paths and the desired output file path
input_files = [
    'file1.jsonl',
    'file2.jsonl',
    'file3.jsonl'
]
output_file = 'combined_output.jsonl'
combine_jsonl(input_files, output_file)
