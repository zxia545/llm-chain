import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
from concurrent.futures import ThreadPoolExecutor
import time 
import logging


logging.basicConfig(level=logging.INFO, filename=f'type1_running_{time.time()}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the script...")
def construct_messages(dataset_type, step, question=None, answer=None, doubts=None):
    """
    Construct role-based messages for LLM interactions based on dataset type and step.
    The prompts are tailored to improve quality by reinforcing focus, structure, and depth,
    encouraging thoughtful doubts and meaningful improvements.
    """
    if dataset_type == "Infinity-Instruct":
        if step == 1:
            return [
                {"role": "system", "content": "You are an AI assistant designed to provide accurate, clear, complete, and helpful answers to user instructions."},
                {"role": "user", "content": question}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are an AI assistant. You will read the question and an answer provided by another AI assistant. If there is anything in the answer you find unclear, incomplete, or confusing, ask specific questions to better understand those parts."},
                {"role": "user", "content": f"Question: {question}\nHere is the answer:\n{answer}\n\nPlease list any questions you have about details or reasoning you do not fully understand."}
            ]
        elif step == 3:
            return [
                {"role": "system", "content": "You are an AI assistant designed to provide accurate, clear, complete, and helpful answers to user instructions."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": f"You are tasked with improving an answer based on the questions provided. Update and refine the original answer to address these questions clearly and effectively.\nQuestion: {question}\nPrevious Answer: {answer}\nFeedback: {doubts}\n\nPlease update the answer accordingly:"}
            ]
    elif dataset_type == "MAmmoTH":
        if step == 1:
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
        elif step == 2:
            return [
                {"role": "system", "content": "You are an AI assistant. You will read the math problem and a solution provided by another AI assistant. If any step in the solution is unclear, lacks justification, or appears incomplete, ask specific questions to clarify or better understand those parts."},
                {"role": "user", "content": f"Math Problem: {question}\nHere is the solution:\n{answer}\n\nPlease list your questions about the reasoning steps or details you do not fully understand."}
            ]
        elif step == 3:
            question_lower = question.lower()
            if "program" in question_lower or "python" in question_lower:
                return [
                    {"role": "system", "content": "You are a mathematician and educator. Solve the following math problem with accurate, complete, and clear explanations."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": f"You are tasked with improving a math solution based on the questions provided. Refine and enhance the original solution to address these questions, ensuring accuracy, logical reasoning, and clear explanations.\nMath Problem: {question}\nPrevious Solution: {answer}\nFeedback: {doubts}\n\nPlease update the solution accordingly:"}
                ]
            else:
                return [
                    {"role": "system", "content": "You are a mathematician and educator. Solve the following math problem with accurate, complete, and clear explanations. For every question, break down your reasoning into a logical chain of steps, and provide the final answer only after completing the reasoning."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": f"You are tasked with improving a math solution based on the questions provided. Refine and enhance the original solution to address these questions, ensuring accuracy, logical reasoning, and clear explanations.\nMath Problem: {question}\nPrevious Solution: {answer}\nFeedback: {doubts}\n\nPlease update the solution accordingly:"}
                ]
    elif dataset_type == "WizardCoder":
        if step == 1:
            return [
                {"role": "system", "content": "You are an expert programmer and problem solver. Your task is to provide correct, efficient, readable, and well-structured code solutions to programming problems, adhering to best coding practices throughout."},
                {"role": "user", "content": question}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are an AI assistant. You will read the programming problem and a proposed code solution. If there is any part of the solution or its reasoning you find unclear or confusing, ask specific questions to clarify those parts."},
                {"role": "user", "content": f"Programming Problem: {question}\nHere is the code solution:\n{answer}\n\nPlease list your questions about any unclear logic, implementation detail, or part of the solution you do not fully understand."}
            ]
        elif step == 3:
            return [
                {"role": "system", "content": "You are an expert programmer and problem solver. Your task is to provide correct, efficient, readable, and well-structured code solutions to programming problems, adhering to best coding practices throughout."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": f"You are tasked with improving a code solution based on the questions provided. Refactor, correct, or enhance the code to address these questions, ensuring it is correct, efficient, readable, and adheres to best practices.\nProgramming Problem: {question}\nPrevious Code Solution: {answer}\nFeedback: {doubts}\n\nPlease update the code solution accordingly."}
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Infinity-Instruct", "MAmmoTH", "WizardCoder"], help="Type of dataset being processed.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Original Q input JSONL.")
    parser.add_argument("--llm1_model", type=str, required=True, help="Model path for LLM1.")
    parser.add_argument("--llm2_model", type=str, required=True, help="Model path for LLM2.")
    parser.add_argument("--llm1_name", type=str, default="LLM1", help="Name for LLM1.")
    parser.add_argument("--llm2_name", type=str, default="LLM2", help="Name for LLM2.")
    parser.add_argument("--port1", type=int, default=8000, help="Port for LLM1.")
    parser.add_argument("--port2", type=int, default=8001, help="Port for LLM2.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for concurrent processing.")
    args = parser.parse_args()

    # Step 1: q -> LLM1 -> a
    print("[INFO] Step1: q -> LLM1 -> a")
    process_llm1 = start_vllm_server(args.llm1_model, args.llm1_name, args.port1, args.gpu)
    data_list = list(read_jsonl(args.input_jsonl))
    step1_file = f"outputs/type2_step1_{os.path.basename(args.input_jsonl)}"
    step1_data = []
    api_base_llm1 = f"http://localhost:{args.port1}"

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_record, api_base_llm1, args.llm1_name, args.dataset_type, 1, record) for record in data_list]
        for i, future in enumerate(futures, start=1):
            step1_data.append(future.result())
            if i % 2000 == 0:
                save_partial_results(step1_file, step1_data, append=True)

    save_partial_results(step1_file, step1_data, append=True)
    stop_vllm_server(process_llm1)

    # Step 2: <q, a> -> LLM2 -> t
    print("[INFO] Step2: <q, a> -> LLM2 -> t")
    process_llm2 = start_vllm_server(args.llm2_model, args.llm2_name, args.port2, args.gpu)
    step2_file = f"outputs/type2_step2_{os.path.basename(args.input_jsonl)}"
    step2_data = []
    step1_data_reloaded = list(read_jsonl(step1_file))
    api_base_llm2 = f"http://localhost:{args.port2}"

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_record, api_base_llm2, args.llm2_name, args.dataset_type, 2, record) for record in step1_data_reloaded]
        for i, future in enumerate(futures, start=1):
            step2_data.append(future.result())
            if i % 2000 == 0:
                save_partial_results(step2_file, step2_data, append=True)

    save_partial_results(step2_file, step2_data, append=True)
    stop_vllm_server(process_llm2)

    # Step 3: <q, a, t> -> LLM1 -> a'
    print("[INFO] Step3: <q, a, t> -> LLM1 -> a'")
    process_llm1_step3 = start_vllm_server(args.llm1_model, args.llm1_name, args.port1, args.gpu)
    step3_file = f"outputs/type2_step3_{os.path.basename(args.input_jsonl)}"
    step3_data = []
    step2_data_reloaded = list(read_jsonl(step2_file))

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_record, api_base_llm1, args.llm1_name, args.dataset_type, 3, record) for record in step2_data_reloaded]
        for i, future in enumerate(futures, start=1):
            step3_data.append(future.result())
            if i % 2000 == 0:
                save_partial_results(step3_file, step3_data, append=True)

    save_partial_results(step3_file, step3_data, append=True)
    stop_vllm_server(process_llm1_step3)

    print("[INFO] Type2 pipeline complete.")

if __name__ == "__main__":
    main()
