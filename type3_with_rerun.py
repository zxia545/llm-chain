import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion, start_vllm_server_with_gpus, allocate_gpus, get_training_data
import logging
import time
import re

logging.basicConfig(level=logging.WARNING, filename=f'type3_with_self_rerun_{time.strftime("%d_%H_%M_%S")}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.warning("Starting the script...")


def construct_messages(dataset_type, step, question=None, std_answer=None, confusion=None):
    """
    Construct role-based messages for LLM interactions based on dataset type and step.
    System prompts are tailored to encourage deep critical thinking and probing doubt.
    """

    if dataset_type == "Magpie_Math_Instruct":
        # --------------------------- Step 1 ---------------------------
        if step == 1:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant. You will read a correct solution to the following math problem. "
                        "If you have any questions or confusions about the solution, please ask specific questions regarding those parts."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Math Problem: {question}\n"
                        f"Here is the solution:\n{std_answer}\n\n"
                        "List at most 3 specific questions about this solution."
                    )
                }
            ]




        # --------------------------- Step 2 ---------------------------
        elif step == 2:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a mathematician and educator dedicated to resolving confusions about math solutions. "
                        "Provide clear, step-by-step explanations to logically address each confusion. "
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Math Problem: {question}\n"
                        f"Solution: {std_answer}\n\n\n"
                        f"Confusions about the solution: {confusion}\n\n"
                        "Please address the confusions.\n\n"
                    )
                }
            ]




    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")




def save_partial_results(file_path, data, append=False):
    if data:
        write_jsonl(file_path, data, append=append)
        data.clear()

def process_record(api_base, llm_model, dataset_type, step, record):
    question = record.get("input", record.get("q"))
    std_answer = record.get("output", record.get("a_std"))
    confusion = record.get("t", None)
    idx = record.get("idx", None)

    try:
        messages = construct_messages(dataset_type, step, question=question, std_answer=std_answer, confusion=confusion)
        response = chat_completion(api_base, llm_model, messages, max_tokens=2048, temperature=0.7)
    except Exception as e:
        response = f"[LLM Error] {str(e)}"
        
    if step == 1:
        result = {"idx": idx, "q": question, "a_std": std_answer, "t": response}
    elif step == 2:
        result = {"idx": idx, "q": question, "a_std": std_answer, "t": confusion, "response": response}
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
    # parser.add_argument("--wrong_jsonl", type=str, required=True, help="Wrong output will be like 'rerun_cut_type2_step3_xxx.jsonl'")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Folder name for the output.")
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
        
    # check if output file exists
    is_output_file_exists = os.path.exists(args.output_jsonl)
    
    
    latest_vllm_process_id = None
    """
    !!! This is the first time running the script !!! 
    """
    ###############################################################################################################################################################
    if  not is_output_file_exists:
        logger.warning("[INFO] Step1: <q, a_std> -> LLM2 -> t")
        process_llm2 = start_vllm_server(args.llm2_model, args.llm2_name, args.port2, args.gpu)
        data_list = list(read_jsonl(args.input_jsonl))
        step1_file = f"{output_folder_path}/type3_step1_{os.path.basename(args.input_jsonl)}"
        step1_data = []
        api_base_llm2 = f"http://localhost:{args.port2}"
        
        # Refine data list based on existing results
        data_list = refine_list(data_list, step1_file)

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_record, api_base_llm2, args.llm2_name, args.dataset_type, 1, record) for record in data_list]
            for i, future in enumerate(futures, start=1):
                step1_data.append(future.result())
                if i % 2000 == 0:
                    save_partial_results(step1_file, step1_data, append=True)

        save_partial_results(step1_file, step1_data, append=True)
        stop_vllm_server(process_llm2)

        # Step 2: <q, a_std, t> -> LLM1 -> a'
        logger.warning("[INFO] Step2: <q, a_std, t> -> LLM1 -> a'")
        latest_vllm_process_id = start_vllm_server(args.llm1_model, args.llm1_name, args.port1, args.gpu)
        step2_file = f"{output_folder_path}/type3_step2_{os.path.basename(args.input_jsonl)}"
        step2_data = []
        step1_data_reloaded = list(read_jsonl(step1_file))
        api_base_llm1 = f"http://localhost:{args.port1}"
        
        # Refine data list based on existing results
        step1_data_reloaded = refine_list(step1_data_reloaded, step2_file)
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_record, api_base_llm1, args.llm1_name, args.dataset_type, 2, record) for record in step1_data_reloaded]
            for i, future in enumerate(futures, start=1):
                step2_data.append(future.result())
                if i % 2000 == 0:
                    save_partial_results(step2_file, step2_data, append=True)

        save_partial_results(step2_file, step2_data, append=True)
        
        # This is the final step to process the jsonl file
        get_training_data(step2_file, args.output_jsonl)
        
        
    else:
        logger.warning(f"[SECTION1] - Bypass the initial data generation")
        if not is_output_file_exists:
            logger.warning(f"[WARNING] The output file {args.output_jsonl} does not exist, please run the script without bypass_init to generate the output file")
        else:
            logger.warning(f"[GOOD] The output file {args.output_jsonl} already exist. Can run the script without bypass_init to regenerate the output file")
            
            
        logger.warning(f'[INFO] start vllm server with {args.gpu} gpus')
        latest_vllm_process_id = start_vllm_server(args.llm1_model, args.llm1_name, args.port1, args.gpu)

   
    
    stop_vllm_server(latest_vllm_process_id)

if __name__ == "__main__":
    main()