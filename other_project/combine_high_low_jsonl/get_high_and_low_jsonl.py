import json
import argparse
# Function to split JSONL based on labels
def split_jsonl(input_jsonl, label_jsonl):
    # Open and read input JSONL and label JSONL files
    with open(input_jsonl, 'r') as input_file:
        input_data = [json.loads(line) for line in input_file]

    with open(label_jsonl, 'r') as label_file:
        label_data = [json.loads(line) for line in label_file]

    # Initialize lists for HIGH and LOW items
    high_items = []
    low_items = []

    # Create a lookup dictionary for labels by idx
    label_dict = {item['idx']: item['output'].strip().lower() for item in label_data}

    # Process each item in the input file
    for item in input_data:
        idx = item['idx']
        if idx in label_dict:
            if label_dict[idx] == 'high':
                high_items.append(item)
            elif label_dict[idx] == 'low':
                low_items.append(item)

    # Save HIGH and LOW items to separate JSONL files
    base_name = input_jsonl.split('.')[0]
    high_output = f"{base_name}_HIGH.jsonl"
    low_output = f"{base_name}_LOW.jsonl"

    with open(high_output, 'w') as high_file:
        for item in high_items:
            high_file.write(json.dumps(item) + '\n')

    with open(low_output, 'w') as low_file:
        for item in low_items:
            low_file.write(json.dumps(item) + '\n')

    print(f"Files saved: {high_output}, {low_output}")

# Example usage
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparse.add_argument('-i', '--input_jsonl', type=str)
    argparse.add_argument('-l', '--label_jsonl', type=str)
    args = argparser.parse_args()
    
    
    split_jsonl(args.input_jsonl, args.label_jsonl)