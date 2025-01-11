import json
import pandas as pd

# Load the JSON file
json_file = "results_2025-01-10T12-32-23.787829.json"
with open(json_file, "r") as f:
    data = json.load(f)

results = data["results"]
group_subtasks = data["group_subtasks"]

# Define task mapping
tasks = {
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
            "mmlu_stem",
        ],
    },
}

# Helper function to calculate averages
def calculate_average(task_key, metric, subgroups=None):
    if subgroups:
        scores = []
        for subgroup in subgroups:
            subtasks = group_subtasks.get(subgroup, [])
            for subtask in subtasks:
                scores.append(results[subtask].get(metric, 0))
        return sum(scores) / len(scores) if scores else None
    else:
        return results[task_key].get(metric, None)

# Process each task
output_data = []
for task_name, task_info in tasks.items():
    if "subgroups" in task_info:
        avg_score = calculate_average(
            task_info["key"], "acc,none", subgroups=task_info["subgroups"]
        )
    else:
        avg_score = calculate_average(task_info["key"], task_info["metric"])
    output_data.append({"Task": task_name, "Average Score": avg_score})

# Create a DataFrame and save to Excel
df = pd.DataFrame(output_data)
output_file = "leaderboard_results.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
