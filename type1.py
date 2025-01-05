import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
import logging
import time

# Configure logger
logging.basicConfig(level=logging.INFO, filename=f'type1_running_{time.time()}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the script...")

def construct_type1_message(dataset_type, question):
    """
    Constructs the message structure for Type1 interaction based on dataset type.
    """
    if dataset_type == "Infinity-Instruct":
        return [
            {"role": "system", "content": "You are an AI assistant designed to provide accurate, clear, complete, and helpful answers to user instructions."},
            {"role": "user", "content": question}
        ]
    elif dataset_type == "MAmmoTH":
        question_lower = question.lower()
        if "program" in question_lower or "python" in question_lower:
            return [
                {"role": "system", "content": "You are a mathematician and educator. Solve the following math problem with accurate, complete, and clear explanations."},
                {"role": "user", "content": question}
            ]
        else:
            return [
                {"role": "system", "content": "You are a mathematician and educator. Solve the following math problem with accurate, complete, and clear explanations. For every question, break down your reasoning into a logical chain of steps, and provide the final answer only after completing the reasoning."},
                {"role": "user", "content": question}
            ]
    elif dataset_type == "WizardCoder":
        return [
            {"role": "system", "content": "You are an expert programmer and problem solver. Your task is to provide correct, efficient, readable, and well-structured code solutions to programming problems, adhering to best coding practices throughout."},
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
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for concurrent processing.")
    args = parser.parse_args()

    # Step 1: Start vLLM server for the chosen model
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)

    # Step 2: Prepare reading/writing
    data_list = list(read_jsonl(args.input_jsonl))
    model_dir = os.path.join("outputs", args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    output_file = os.path.join(model_dir, os.path.basename(args.input_jsonl))

    # Load existing output JSONL if it exists
    if os.path.exists(output_file):
        logger.info(f"[INFO] Loading existing results from {output_file}")
        existing_results = list(read_jsonl(output_file))
        existing_ids = {record["idx"] for record in existing_results}
        data_list = [record for record in data_list if record.get("idx") not in existing_ids]
        logger.info(f"[INFO] {len(data_list)} new records will be processed.")
    else:
        logger.info(f"[INFO] No existing results found. Processing all records.")

    output_data = []

    # Step 3: Multithreaded processing
    api_base = f"http://localhost:{args.port}/v1"

    def save_partial_results():
        if output_data:
            write_jsonl(output_file, output_data, append=True)
            output_data.clear()

    def process_record(record):
        question = record["input"]
        idx = record.get("idx", None)

        try:
            messages = construct_type1_message(args.dataset_type, question)
            logger.info(f"[INFO] Processing record {idx}...")
            answer = chat_completion(api_base, args.model_name, messages, max_tokens=2048, temperature=0.7)
        except Exception as e:
            answer = f"[Error calling LLM] {str(e)}"
            logger.error(f"[ERROR] Failed to process record {idx}: {str(e)}")
        
        logger.info(f"[INFO] Completed record {idx}.")
        return {"idx": idx, "input": question, "output": answer}

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_record, record) for record in data_list]
        pre_time = time.time()
        for i, future in enumerate(futures, start=1):
            output_data.append(future.result())
            # Save intermediate results every 2000 records
            if i % 2000 == 0:
                current_time = time.time()
                save_partial_results()
                logger.warning(f"[INFO] Processed {i} records in {current_time - pre_time:.2f}s.")
                pre_time = current_time

    # Save any remaining records
    save_partial_results()

    logger.info(f"[INFO] Output saved to {output_file}")

    # Step 5: Stop the server
    stop_vllm_server(process)

if __name__ == "__main__":
    main()
