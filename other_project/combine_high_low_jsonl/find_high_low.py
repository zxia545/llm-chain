import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
import logging
import time

# Configure logger
logging.basicConfig(level=logging.INFO, filename=f'type1_running_{time.strftime("%d_%H_%M_%S")}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the script...")


def construct_prompt():
    """
    Returns a system prompt for guiding the LLM to classify the Q&A pair
    as LOW or HIGH quality, including a rule to check if the question is math-related.
    """
    return (
        "You are a helpful assistant that classifies user questions and answers as LOW quality or HIGH quality.\n"
        "You only reply with a single word: LOW or HIGH.\n\n"
        "Classification rules:\n"
        "1) If the Question is not clearly math-related or the user is asking about something irrelevant to mathematics, respond with LOW.\n"
        "2) If the Question or Answer is nonsensical, contradictory, incomplete, or severely incorrect, respond with LOW.\n"
        "3) If the Question or Answer contains an excessive amount of irrelevant or redundant information—such as repeated symbols, random characters, or other unnecessary formatting—respond with LOW.\n"
        "4) Otherwise respond with HIGH.\n\n"
        "Remember: only output one word: LOW or HIGH. Do not provide any explanation or additional commentary."
    )

def construct_type1_message(dataset_type, question, old_answer):
    """
    Constructs the message structure for Type1 interaction based on dataset type.
    """
    input_text = question
    output_text = old_answer
    s_prompt = construct_prompt()
    result_list = [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": (
            f"Question:\n{input_text}\n\n"
            f"Answer:\n{output_text}\n\n"
            "Please respond with LOW or HIGH (only output one word: LOW or HIGH)."
        )}
    ]
    return result_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Magpie_Math_Instruct_part1", "Magpie_Math_Instruct_part2", "Magpie_Math_Instruct_part3", "Magpie_Math_Instruct_part4"],
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
        old_answer = record["output"]
        idx = record.get("idx", None)

        try:
            messages = construct_type1_message(args.dataset_type, question, old_answer)
            print(messages)
            logger.info(f"[INFO] Processing record {idx}...")
            answer = chat_completion(api_base, args.model_name, messages, max_tokens=10, temperature=0.7)
        except Exception as e:
            answer = f"[Error calling LLM] {str(e)}"
            logger.error(f"[ERROR] Failed to process record {idx}: {str(e)}")
        
        logger.info(f"[INFO] Completed record {idx}.")
        return {"idx": idx,"answer":answer, "input": question, "output": old_answer}

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
