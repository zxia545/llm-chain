import re
import json
from utils import read_jsonl, write_jsonl

def process_jsonl(input_file):

    processed_data = []
    
    idx_set = set()
    repeat_count = 0 
    for record in read_jsonl(input_file):
        idx = record.get("idx")
        if idx in idx_set:
            repeat_count += 1
            continue
        idx_set.add(idx)
        processed_data.append(record)
        

    
    # sort by idx
    processed_data = sorted(processed_data, key=lambda x: x["idx"])
    
    write_jsonl(input_file, processed_data, append=False)
    
    print(f"Processed {len(processed_data)} records from {input_file}.")
    print(f"Removed {repeat_count} repeated idxs.")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    args = parser.parse_args()

    process_jsonl(args.input_file)
