import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
import logging
import time

# Configure logger
logging.basicConfig(level=logging.INFO, filename=f'type1{time.strftime("%d_%H_%M_%S")}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the script...")

def construct_type1_message(dataset_type: str, question: str):
    """
    Build a single-turn prompt (system + user) for Type-1 generation.
    """
    if dataset_type == "Magpie_Math_Instruct":
        sys_msg = (
            "You are a mathematician and educator. Solve the following math problem with accurate, "
            "complete, and clear explanations. Break down your reasoning into a logical chain of steps, "
            "and provide the final answer only after completing the reasoning."
        )
    elif dataset_type == "Anthropology":
        sys_msg = (
            "You are a cultural anthropologist and educator. Analyse the question below using relevant "
            "anthropological theories and ethnographic evidence. Present your reasoning in clear stages, "
            "compare differing perspectives if useful, and give a concise conclusion only after the full analysis."
        )
    elif dataset_type == "Economics":
        sys_msg = (
            "You are an economist and lecturer. Provide a rigorous, data-oriented solution to the question below. "
            "State any assumptions, apply appropriate economic models, work through the logic step-by-step, "
            "and give the final answer or policy implication after completing the reasoning."
        )
    elif dataset_type == "Law":
        sys_msg = (
            "You are an experienced attorney and legal scholar. Craft a well-structured legal analysis of the "
            "question below. Identify controlling statutes and case law, develop arguments, consider counter-"
            "arguments, and conclude with a succinct holding only after your step-by-step reasoning."
        )
    elif dataset_type == "Philosophy":
        sys_msg = (
            "You are a professor of philosophy. Construct a rigorous argument addressing the question below. "
            "Outline key positions, evaluate counter-arguments, and defend your conclusion. Present your reasoning "
            "step-by-step before stating your final position."
        )
    elif dataset_type == "WizardCoder":
        sys_msg = (
            "You are an expert programmer and problem solver. Provide correct, efficient, readable, and well-"
            "structured code solutions to programming problems, following best practices throughout."
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": question},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Type of dataset being processed.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model for vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for vLLM.")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Folder name for the output.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for concurrent processing.")
    args = parser.parse_args()
    
    # check if output folder exist or not
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)
    output_folder_path = args.output_folder_path
    
    # Step 1: Start vLLM server for the chosen model
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)

    # Step 2: Prepare reading/writing
    data_list = list(read_jsonl(args.input_jsonl))

    output_file = os.path.join(output_folder_path, f'type1_{os.path.basename(args.input_jsonl)}')

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
        return {"idx": idx, "input": question, "output": answer, "domain": args.dataset_type}

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
