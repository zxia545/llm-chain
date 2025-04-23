import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion, start_vllm_server_with_gpus, allocate_gpus, get_training_data
from concurrent.futures import ThreadPoolExecutor
import time 
import logging
import re

logging.basicConfig(level=logging.WARNING, filename=f'type2{time.strftime("%d_%H_%M_%S")}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.warning("Starting the script...")



def construct_messages(dataset_type: str,
                       step: int,
                       question: str | None = None,
                       answer:   str | None = None,
                       confusion: str | None = None):
    """
    Build multi-turn prompts for the 3-step Type-2 pipeline
    across Math, Anthropology, Economics, Law, and Philosophy.
    """

    T = {  # ───────────────── DOMAIN TEMPLATES ──────────────────
        "Magpie_Math_Instruct": dict(
            teacher_sys = (
                "You are a mathematician and educator. Solve the following math problem with accurate, "
                "complete, and clear explanations. Break down your reasoning into a logical chain of steps, "
                "and provide the final answer only after completing the reasoning."
            ),
            student_sys = (
                "You are a math student. You will read the math problem and a solution provided by your math teacher. "
                "If you have any questions or confusions about the solution, ask specific questions about those parts."
            ),
            resolver_sys = (
                "You are a mathematician and educator dedicated to resolving confusions about math solutions. "
                "Provide clear, step-by-step explanations to logically address each confusion."
            ),
            lbl_q = "Math Problem",
            lbl_a = "Here is the solution",
            lbl_c = "Confusions about the solution",
        ),

        "Anthropology": dict(
            teacher_sys = (
                "You are a cultural anthropologist and educator. Analyse the following question using relevant "
                "anthropological theories and ethnographic evidence. Present your reasoning in clear stages, "
                "compare differing perspectives where appropriate, and give a concise conclusion only after the full analysis."
            ),
            student_sys = (
                "You are an anthropology student. Read the anthropologist’s answer carefully. "
                "List any aspects that remain unclear, contradictory, or insufficiently supported."
            ),
            resolver_sys = (
                "You are a cultural anthropologist clarifying your previous analysis. "
                "Resolve each listed confusion with evidence-based, step-by-step explanations."
            ),
            lbl_q = "Anthropology Question",
            lbl_a = "Anthropological Analysis",
            lbl_c = "Confusions about the analysis",
        ),

        "Economics": dict(
            teacher_sys = (
                "You are an economist and lecturer. Provide a rigorous, data-oriented answer to the question below. "
                "State assumptions, apply appropriate economic models, work through the logic step-by-step, "
                "and present the final answer or policy implication only after completing the reasoning."
            ),
            student_sys = (
                "You are an economics student reviewing the economist’s answer. "
                "Identify any unclear assumptions, missing steps, or alternative interpretations."
            ),
            resolver_sys = (
                "You are an economist clarifying your previous analysis. "
                "Address each uncertainty with detailed reasoning, equations if helpful, and real-world examples."
            ),
            lbl_q = "Economics Question",
            lbl_a = "Economic Analysis",
            lbl_c = "Confusions about the analysis",
        ),

        "Law": dict(
            teacher_sys = (
                "You are an experienced attorney and legal scholar. Draft a well-structured legal analysis answering "
                "the question below. Cite relevant statutes and case law, consider counter-arguments, and conclude "
                "with a succinct holding only after your step-by-step reasoning."
            ),
            student_sys = (
                "You are a law student analysing the attorney’s answer. "
                "List any ambiguities, unsupported assertions, or unresolved issues that need clarification."
            ),
            resolver_sys = (
                "You are an attorney clarifying your legal analysis. "
                "Respond to each ambiguity systematically, citing authority and reasoning."
            ),
            lbl_q = "Legal Question",
            lbl_a = "Attorney’s Analysis",
            lbl_c = "Confusions about the analysis",
        ),

        "Philosophy": dict(
            teacher_sys = (
                "You are a professor of philosophy. Produce a rigorous answer to the question below. "
                "Present key positions, evaluate counter-arguments, and defend your conclusion. "
                "Lay out your reasoning step-by-step before stating your final stance."
            ),
            student_sys = (
                "You are a philosophy student critiquing the professor’s argument. "
                "List any hidden premises, logical gaps, or ambiguities that require clarification."
            ),
            resolver_sys = (
                "You are a philosopher clarifying your argument. "
                "Address each doubt with transparent, step-by-step reasoning and illustrative examples if useful."
            ),
            lbl_q = "Philosophical Question",
            lbl_a = "Philosopher’s Argument",
            lbl_c = "Confusions about the argument",
        ),
    }

    if dataset_type not in T:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'")

    P = T[dataset_type]      # shorthand to this domain’s template
    Q, A, C = P["lbl_q"], P["lbl_a"], P["lbl_c"]

    # ───────────────────── STEP-SPECIFIC PROMPTS ──────────────────────
    if step == 1:   # teacher produces initial answer
        return [
            {"role": "system", "content": P["teacher_sys"]},
            {"role": "user",   "content": question},
        ]

    if step == 2:   # student raises confusions
        return [
            {"role": "system", "content": P["student_sys"]},
            {"role": "user",
             "content": (
                 f"{Q}: {question}\n"
                 f"{A}:\n{answer}\n\n"
                 "List your confusions."
             )},
        ]

    if step == 3:   # resolver clarifies
        return [
            {"role": "system", "content": P["resolver_sys"]},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
            {"role": "user",
             "content": (
                 f"{C}: {confusion}\n\n"
                 "Please address each one."
             )},
        ]

    raise ValueError("step must be 1, 2, or 3")

 
def save_partial_results(file_path, data, append=False):
    if data:
        write_jsonl(file_path, data, append=append)
        data.clear()
        
        
def process_record(api_base, llm_model_name, dataset_type, step, record):
    question = record.get("q", record.get("input"))
    answer = record.get("a", None)
    confusion = record.get("t", None)
    idx = record.get("idx", None)

    try:
        messages = construct_messages(dataset_type, step, question=question, answer=answer, confusion=confusion)
        response = chat_completion(api_base, llm_model_name, messages, max_tokens=2048, temperature=0.7)
    except Exception as e:
        response = f"[LLM Error] {str(e)}"
        
    if step == 1:
        result = {"idx": idx, "q": question, "a": response, "t": None, "domain": dataset_type}
    elif step == 2:
        result = {"idx": idx, "q": question, "a": answer, "t": response, "domain": dataset_type}
    elif step == 3:
        result = {"idx": idx, "q": question, "a": answer, "t": confusion, "response": response, "domain": dataset_type}
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
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset being processed.")
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
        llm2_gpu_num = args.gpu
        if "Qwen2.5-7B" in args.llm2_model:
            llm2_gpu_num = 4
        process_llm2 = start_vllm_server(args.llm2_model, args.llm2_name, args.port2, llm2_gpu_num)
        step2_file = f"{output_folder_path}/type2_step2_{os.path.basename(args.input_jsonl)}"
        step2_data = []
        step1_data_reloaded = list(read_jsonl(step1_file))
        api_base_llm2 = f"http://localhost:{args.port2}"
        
        # Load existing output JSONL if it exists
        step1_data_reloaded = refine_list(step1_data_reloaded, step2_file)
        
        num_max_workers = args.threads
        if "Qwen2.5-7B" in args.llm2_model:
            num_max_workers = 32
        with ThreadPoolExecutor(max_workers=num_max_workers) as executor:
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
        get_training_data(step3_file, args.output_jsonl)
    else:
        logger.warning(f"[SECTION1] - Bypass the initial data generation")
        if not is_output_file_exists:
            logger.warning(f"[WARNING] The output file {args.output_jsonl} does not exist, please run the script without bypass_init to generate the output file")
        else:
            logger.warning(f"[GOOD] The output file {args.output_jsonl} already exist. Can run the script without bypass_init to regenerate the output file")
    
    stop_vllm_server(latest_vllm_process_id)

    
if __name__ == "__main__":
    main()
