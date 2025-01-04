import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion

def construct_type1_message(dataset_type, question):
    """
    Constructs the message structure for Type1 interaction based on dataset type.
    """
    if dataset_type == "Infinity-Instruct":
        return [
            {"role": "system", "content": "You are a highly knowledgeable assistant providing detailed, contextually accurate answers to diverse user queries."},
            {"role": "user", "content": question}
        ]
    elif dataset_type == "MAmmoTH":
        return [
            {"role": "system", "content": "You are a mathematical problem-solving assistant. Provide step-by-step solutions and explain your reasoning clearly."},
            {"role": "user", "content": question}
        ]
    elif dataset_type == "WizardCoder":
        return [
            {"role": "system", "content": "You are a programming assistant. Solve coding challenges, provide code implementations, and explain the logic behind them."},
            {"role": "user", "content": question}
        ]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Infinity-Instruct", "MAmmoTH", "WizardCoder"],
                        help="Type of dataset being processed.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model for vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for vLLM.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    args = parser.parse_args()

    # Step 1: Start vLLM server for the chosen model
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)

    # Step 2: Prepare reading/writing
    data_list = list(read_jsonl(args.input_jsonl))
    output_data = []

    # Step 3: Process each record and call the model
    api_base = f"http://localhost:{args.port}"
    for record in data_list:
        question = record["input"]
        idx = record.get("idx", None)

        try:
            messages = construct_type1_message(args.dataset_type, question)
            answer = chat_completion(api_base, args.model_name, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            answer = f"[Error calling LLM] {str(e)}"

        output_data.append({
            "idx": idx,
            "input": question,
            "output": answer
        })

    # Step 4: Write results to output directory
    model_dir = os.path.join("outputs", args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    output_file = os.path.join(model_dir, os.path.basename(args.input_jsonl))
    write_jsonl(output_file, output_data)
    print(f"[INFO] Output saved to {output_file}")

    # Step 5: Stop the server
    stop_vllm_server(process)

if __name__ == "__main__":
    main()
