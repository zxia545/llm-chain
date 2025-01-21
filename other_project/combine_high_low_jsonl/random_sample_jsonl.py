import json
import random

# Function to randomly sample 100K items from a JSONL file
def sample_jsonl(input_jsonl, output_jsonl, sample_size=100000):
    # Read all data from the input JSONL file
    with open(input_jsonl, 'r') as input_file:
        data = [json.loads(line) for line in input_file]

    # Randomly sample the specified number of items
    sampled_data = random.sample(data, min(sample_size, len(data)))

    # Write the sampled data to the output JSONL file
    with open(output_jsonl, 'w') as output_file:
        for item in sampled_data:
            output_file.write(json.dumps(item) + '\n')

    print(f"Sampled {len(sampled_data)} items saved to: {output_jsonl}")

# Example usage
# Replace 'input.jsonl' and 'output_sampled.jsonl' with your actual file paths
input_file = 'input.jsonl'
output_file = 'output_sampled.jsonl'
sample_jsonl(input_file, output_file)
