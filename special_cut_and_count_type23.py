import re
import json
from utils import read_jsonl, write_jsonl

def process_jsonl(input_file, output_file, dataset_type):
    cutoff_keywords = {
        "Infinity-Instruct": ["Final_Answer", "Final Answer"],
        "MAmmoTH": ["Final_Solution", "Final Solution"],
        "WizardCoder": ["Refactored_Code", "Refactored Code"]
    }

    if dataset_type not in cutoff_keywords:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    keywords = cutoff_keywords[dataset_type]
    fail_count = 0
    total_count = 0
    
    not_long_count = 0

    processed_data = []
    wrong_data = []

    for record in read_jsonl(input_file):
        total_count += 1
        idx = record.get("idx")
        question = record.get("q", "")
        response = record.get("response", "")

        # Find the cutoff point
        cut_position = None
        for keyword in keywords:
            match = re.search(re.escape(keyword), response, re.IGNORECASE)
            if match:
                cut_position = match.end()
                break

        if cut_position:
            # Trim leading colons or whitespace after the cutoff point
            while cut_position < len(response) and response[cut_position] in [":", " ", "\n"]:
                cut_position += 1
            refined_response = response[cut_position:].strip()
            
            if len(refined_response) < 50:
                fail_count += 1
                not_long_count += 1
                wrong_data.append({
                    "idx": idx,
                    "input": question,
                    "output": response
                })
                continue
            
            
            processed_data.append({
                "idx": idx,
                "input": question,
                "output": refined_response
            })
        else:
            refined_response = response.strip()  # Keep the entire response if no keyword is found
            fail_count += 1
            wrong_data.append({
                "idx": idx,
                "input": question,
                "output": response
            })

        
    output_file_name = output_file.split("/")[-1]
    path_to_output = "/".join(output_file.split("/")[:-1])
    wrong_file = f"rerun_{output_file_name}"
    # write_jsonl(f"{path_to_output}/{wrong_file}", wrong_data)

    # Write processed data to a new JSONL file
    # write_jsonl(output_file, processed_data)

    # Print failure statistics
    failure_rate = (fail_count / total_count) * 100 if total_count else 0
    
    input_file_name = input_file.split("/")[-1]
    
    print(f"Processing complete. \nFor jsonl File{input_file_name}: \nFailed to split {fail_count} out of {total_count} responses ({failure_rate:.2f}%).")
    print(f"Failed to split {not_long_count} out of {total_count} responses because they were not long enough: rate is {(not_long_count/total_count*100)}%.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Infinity-Instruct", "MAmmoTH", "WizardCoder"], help="Type of dataset to process.")

    args = parser.parse_args()

    process_jsonl(args.input_file, args.output_file, args.dataset_type)
