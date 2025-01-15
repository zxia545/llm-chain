import os
import json
import pandas as pd

def process_eval_results(folder_path, output_excel):
    results = []

    # Traverse the folder structure
    evalplus_result_path = os.path.join(folder_path, "evalplus_result")
    if not os.path.exists(evalplus_result_path):
        print(f"Folder 'evalplus_result' not found in {folder_path}")
        return

    for dataset_type in os.listdir(evalplus_result_path):
        dataset_path = os.path.join(evalplus_result_path, dataset_type)

        if os.path.isdir(dataset_path):
            for file_name in os.listdir(dataset_path):
                if file_name.endswith("eval_results.json"):
                    file_path = os.path.join(dataset_path, file_name)

                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    eval_data = data.get("eval", {})
                    total_count = len(eval_data)
                    base_win_count = 0
                    plus_win_count = 0

                    for key, value in eval_data.items():
                        if value.get("base_status") == "pass":
                            base_win_count += 1
                        if value.get("plus_status") == "pass":
                            plus_win_count += 1

                    base_accuracy = (base_win_count / total_count * 100) if total_count > 0 else 0
                    plus_accuracy = (plus_win_count / total_count * 100) if total_count > 0 else 0

                    # Add results for this file
                    results.append({
                        "FileName": file_name,
                        dataset_type: f"{base_accuracy:.2f}%",
                        f"{dataset_type}++": f"{plus_accuracy:.2f}%",
                        "Average": None
                    })

    # Merge and calculate averages
    df = pd.DataFrame(results)
    columns = [col for col in df.columns if col not in ["FileName", "Average"]]

    for index, row in df.iterrows():
        accuracies = [float(row[col].strip('%')) for col in columns if row[col] is not None]
        df.at[index, "Average"] = f"{(sum(accuracies) / len(accuracies)):.2f}%"

    # Save to Excel
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

# Example usage
folder_path = "path/to/input_folder"  # Replace with the actual path
output_excel = "eval_results.xlsx"
process_eval_results(folder_path, output_excel)
