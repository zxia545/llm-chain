import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion, start_vllm_server_with_gpus, allocate_gpus
from concurrent.futures import ThreadPoolExecutor
import time 
import logging
import re

logging.basicConfig(level=logging.WARNING, filename=f'type2_with_self_rerun_{time.strftime("%d_%H_%M_%S")}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.warning("Starting the script...")

# This file require user save the input name in the folder in rerun_cut_type2_step3_xxx.jsonl it will 

def process_jsonl(input_file, output_file):
    processed_data = []
    llm_ignore_data = []

    for record in read_jsonl(input_file):
        total_count += 1
        idx = record.get("idx")
        question = record.get("q", "")
        if "a_std" in record:
            answer = record.get("a_std", "")
        elif "a" in record:
            answer = record.get("a", "")
        doubts = record.get("t", "")
        response = record.get("response", "")

        if "[LLM Error]" in response:
            logger.warning(f'Index {idx} has LLM Error - It maybe too long that pass the max token limit')
            llm_ignore_data.append(record)
            continue
        else:
            final_input = (
                "You are a mathematician and educator dedicated to resolving doubts about math solutions. "
                "Provide clear, step-by-step explanations to logically address each doubt.\n\n"
                f"Math Problem: {question}\n"
                f"Solution: {answer}\n\n\n"
                f"Doubts about the solution: {doubts}\n\n"
                "Please address the doubts.\n\n"
            )
            
            final_output = response
            processed_data.append(
                {
                    'idx': idx,
                    'input': final_input,
                    'output': final_output
                }
            )
        
    output_file_name = output_file.split("/")[-1]
    path_to_output = "/".join(output_file.split("/")[:-1])

    write_jsonl(output_file, processed_data, append=True)
    logger.warning(f"[INFO] Processed data has been saved to {output_file}")
    
    llm_error_file_path = f"{path_to_output}/llm_error_{output_file_name}"
    if len(llm_ignore_data) > 0:
        write_jsonl(llm_error_file_path, llm_ignore_data, append=True)
        logger.warning(f"[INFO] LLM Error data has been saved to {llm_error_file_path}")


def construct_messages(dataset_type, step, question=None, answer=None, doubts=None):
    """
    Construct role-based messages for LLM interactions based on dataset type and step.
    The prompts are tailored to improve quality by reinforcing focus, structure, and depth,
    encouraging thoughtful doubts and meaningful improvements.
    """
    if dataset_type == "Magpie_Math_Instruct":
        # ------------------ Step 1 ------------------
        if step == 1:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a mathematician and educator. Solve the following math problem with accurate, complete, and clear explanations. "
                        "Break down your reasoning into a logical chain of steps, and provide the final answer only after completing the reasoning."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]


        elif step == 2:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant. You will read the math problem and a solution provided by another AI assistant. "
                        # "Ask specific questions about any steps or justifications that are unclear, incomplete, or appear incorrect."
                        "If any step in the solution is unclear, ask specific questions about those parts."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Math Problem: {question}\n"
                        f"Here is the solution:\n{answer}\n\n"
                        "List at most 3 specific questions about this solution."
                    )
                }
            ]
            
        elif step == 3:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a mathematician and educator dedicated to resolving doubts about math solutions. "
                        "Provide clear, step-by-step explanations to logically address each doubt. "
                    )
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                },
                {
                    "role": "user",
                    "content": (
                        f"Doubts about the solution: {doubts}\n\n"
                        "Please address the doubts.\n\n"
                    )
                }
            ]
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
 
def save_partial_results(file_path, data, append=False):
    if data:
        write_jsonl(file_path, data, append=append)
        data.clear()
        
        
def process_record(api_base, llm_model_name, dataset_type, step, record):
    question = record.get("q", record.get("input"))
    answer = record.get("a", None)
    doubts = record.get("t", None)
    idx = record.get("idx", None)

    try:
        messages = construct_messages(dataset_type, step, question=question, answer=answer, doubts=doubts)
        response = chat_completion(api_base, llm_model_name, messages, max_tokens=2048, temperature=0.7)
    except Exception as e:
        response = f"[LLM Error] {str(e)}"
        
    if step == 1:
        result = {"idx": idx, "q": question, "a": response, "t": None}
    elif step == 2:
        result = {"idx": idx, "q": question, "a": answer, "t": response}
    elif step == 3:
        result = {"idx": idx, "q": question, "a": answer, "t": doubts, "response": response}
    return result

def refine_list(data_list, jsonl_path):
    # Load existing output JSONL if it exists
    if os.path.exists(jsonl_path):
        logger.warning(f"[INFO] Loading existing results from {jsonl_path}")
        existing_results = list(read_jsonl(jsonl_path))
        existing_ids = {record["idx"] for record in existing_results}
        data_list = [record for record in data_list if record.get("idx") not in existing_ids]
        logger.warning(f"[INFO] {len(data_list)} new records will be processed.")
    else:
        logger.warning(f"[INFO] No existing results found. Processing all records.")
    return data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Infinity-Instruct", "Magpie_Math_Instruct", "WizardCoder"], help="Type of dataset being processed.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Original Q input JSONL.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Final output will be like 'cut_type2_step3_xxx.jsonl'")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Folder name for the output.")
    # parser.add_argument("--wrong_jsonl", type=str, required=True, help="Wrong output will be like 'rerun_cut_type2_step3_xxx.jsonl'")
    parser.add_argument("--llm1_model", type=str, required=True, help="Model path for LLM1.")
    parser.add_argument("--llm2_model", type=str, required=True, help="Model path for LLM2.")
    parser.add_argument("--llm1_name", type=str, default="LLM1", help="Name for LLM1.")
    parser.add_argument("--llm2_name", type=str, default="LLM2", help="Name for LLM2.")
    parser.add_argument("--port1", type=int, default=8000, help="Port for LLM1.")
    parser.add_argument("--port2", type=int, default=8001, help="Port for LLM2.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for concurrent processing.")
    # parser.add_argument("--bypass_init", type=bool, default=False, help="Bypass the initial process")
    args = parser.parse_args()

    output_folder_path = args.output_folder_path
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # check if output jsonl is a path to jsonl or just the name if just the name then append the output_folder_path
    if not args.output_jsonl.startswith("/"):
        args.output_jsonl = f"{args.output_folder_path}/{args.output_jsonl}"
        

    # check does output file exist
    is_output_file_exists = os.path.exists(args.output_jsonl)
    
    
    latest_vllm_process_id = None
    """
    !!! This is the first time running the script !!! 
    """
    ###############################################################################################################################################################
    if not is_output_file_exists:
        # Step 1: q -> LLM1 -> a
        logger.warning("[SECTION1] - Start the init data processing and generate the output jsonl file")
        logger.warning("[INFO] Step1: q -> LLM1 -> a")
        process_llm1 = start_vllm_server(args.llm1_model, args.llm1_name, args.port1, args.gpu)
        data_list = list(read_jsonl(args.input_jsonl))
        
        step1_file = f"{output_folder_path}/type2_step1_{os.path.basename(args.input_jsonl)}"
        step1_data = []
        api_base_llm1 = f"http://localhost:{args.port1}"
        
        # Load existing output JSONL if it exists
        data_list = refine_list(data_list, step1_file)

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_record, api_base_llm1, args.llm1_name, args.dataset_type, 1, record) for record in data_list]
            for i, future in enumerate(futures, start=1):
                step1_data.append(future.result())
                if i % 2000 == 0:
                    save_partial_results(step1_file, step1_data, append=True)

        save_partial_results(step1_file, step1_data, append=True)
        stop_vllm_server(process_llm1)
        # Step 2: <q, a> -> LLM2 -> t
        logger.warning("[INFO] Step2: <q, a> -> LLM2 -> t")
        process_llm2 = start_vllm_server(args.llm2_model, args.llm2_name, args.port2, args.gpu)
        step2_file = f"{output_folder_path}/type2_step2_{os.path.basename(args.input_jsonl)}"
        step2_data = []
        step1_data_reloaded = list(read_jsonl(step1_file))
        api_base_llm2 = f"http://localhost:{args.port2}"
        
        # Load existing output JSONL if it exists
        step1_data_reloaded = refine_list(step1_data_reloaded, step2_file)
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_record, api_base_llm2, args.llm2_name, args.dataset_type, 2, record) for record in step1_data_reloaded]
            for i, future in enumerate(futures, start=1):
                step2_data.append(future.result())
                if i % 2000 == 0:
                    save_partial_results(step2_file, step2_data, append=True)

        save_partial_results(step2_file, step2_data, append=True)
        stop_vllm_server(process_llm2)

        # Step 3: <q, a, t> -> LLM1 -> a'
        logger.warning("[INFO] Step3: <q, a, t> -> LLM1 -> a'")
        latest_vllm_process_id = start_vllm_server(args.llm1_model, args.llm1_name, args.port1, args.gpu)
        step3_file = f"{output_folder_path}/type2_step3_{os.path.basename(args.input_jsonl)}"
        step3_data = []
        step2_data_reloaded = list(read_jsonl(step2_file))

        # Load existing output JSONL if it exists
        step2_data_reloaded = refine_list(step2_data_reloaded, step3_file)
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_record, api_base_llm1, args.llm1_name, args.dataset_type, 3, record) for record in step2_data_reloaded]
            for i, future in enumerate(futures, start=1):
                step3_data.append(future.result())
                if i % 2000 == 0:
                    save_partial_results(step3_file, step3_data, append=True)

        save_partial_results(step3_file, step3_data, append=True)
        # This is the final step to process the jsonl file
        process_jsonl(step3_file, args.output_jsonl)
    else:
        logger.warning(f"[SECTION1] - Bypass the initial data generation")
        if not is_output_file_exists:
            logger.warning(f"[WARNING] The output file {args.output_jsonl} does not exist, please run the script without bypass_init to generate the output file")
        else:
            logger.warning(f"[GOOD] The output file {args.output_jsonl} already exist. Can run the script without bypass_init to regenerate the output file")
    
    stop_vllm_server(latest_vllm_process_id)

    
if __name__ == "__main__":
    main()
