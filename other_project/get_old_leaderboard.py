import os
import json
import pandas as pd

# Function to process each folder
def process_folders(base_path):
    results = []

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Process the JSON data
                    averages = calculate_averages(data)
                    averages["Folder"] = folder_name
                    results.append(averages)

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)
    df = df[["Folder", "ARC", "HellaSwag", "TruthfulQA", "Winogrande", "GSM8k", "MMLU", "Avg"]]
    output_path = os.path.join(base_path, "result.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

# Function to calculate averages for groups
def calculate_averages(data):
    group_definitions = {
        "ARC": {"key": "arc_challenge", "metric": "acc_norm,none"},
        "HellaSwag": {"key": "hellaswag", "metric": "acc_norm,none"},
        "TruthfulQA": {"key": "truthfulqa_mc2", "metric": "acc,none"},
        "Winogrande": {"key": "winogrande", "metric": "acc,none"},
        "GSM8k": {"key": "gsm8k", "metric": "exact_match,strict-match"},
        "MMLU": {
            "key": "mmlu",
            "subgroups": [
                "mmlu_humanities",
                "mmlu_social_sciences",
                "mmlu_other",
                "mmlu_stem"
            ]
        }
    }

    group_averages = {}
    total_average = []

    for group_name, task_info in group_definitions.items():
        scores = []
        if "subgroups" in task_info:
            for subgroup in task_info["subgroups"]:
                if subgroup in data["results"]:
                    scores.append(data["results"][subgroup].get("acc,none", 0))
        else:
            if task_info["key"] in data["results"]:
                scores.append(data["results"][task_info["key"]].get(task_info["metric"], 0))
        
        group_avg = sum(scores) / len(scores) if scores else 0
        group_averages[group_name] = group_avg * 100
        total_average.extend(scores)

    group_averages["Avg"] = (sum(total_average) / len(total_average) * 100) if total_average else 0
    return group_averages

# Define the base path for your folders
base_folder = "/path/to/current/folder"
process_folders(base_folder)
