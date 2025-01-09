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
    df = df[["Folder", "IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU-PRO", "Avg"]]
    output_path = os.path.join(base_path, "result.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

# Function to calculate averages for groups
def calculate_averages(data):
    group_definitions = {
        "IFEval": ["leaderboard_ifeval"],
        "BBH": data["group_subtasks"]["leaderboard_bbh"],
        "MATH": data["group_subtasks"]["leaderboard_math_hard"],
        "GPQA": data["group_subtasks"]["leaderboard_gpqa"],
        "MUSR": data["group_subtasks"]["leaderboard_musr"],
        "MMLU-PRO": ["leaderboard_mmlu_pro"]
    }

    group_averages = {}
    total_average = []

    for group_name, tasks in group_definitions.items():
        scores = []
        for task in tasks:
            if task in data["results"]:
                task_data = data["results"][task]
                if task == "leaderboard_ifeval":
                    acc_keys = [key for key in task_data.keys() if "acc,none" in key]
                    scores.extend([task_data[key] for key in acc_keys])
                elif "acc_norm,none" in task_data:
                    scores.append(task_data["acc_norm,none"])
                elif "exact_match,none" in task_data:
                    scores.append(task_data["exact_match,none"])
                elif "acc,none" in task_data:
                    scores.append(task_data["acc,none"])
        
        group_avg = sum(scores) / len(scores) if scores else 0
        group_averages[group_name] = group_avg * 100
        total_average.extend(scores)

    group_averages["Avg"] = (sum(total_average) / len(total_average) * 100) if total_average else 0
    return group_averages

# Define the base path for your folders
base_folder = "/path/to/current/folder"
process_folders(base_folder)
