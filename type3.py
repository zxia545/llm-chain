import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion

def construct_messages(dataset_type, step, question=None, std_answer=None, doubts=None):
    """
    Construct role-based messages for LLM interactions based on dataset type and step.
    System prompts are tailored to encourage deep critical thinking and probing doubt.
    """
    if dataset_type == "Infinity-Instruct":
        if step == 1:
            return [
                {"role": "system", "content": "You are a critical and analytical assistant. Your role is to evaluate the given standard answer with extreme scrutiny. Assume nothing and question everything. Your analysis must be rigorous, identifying any potential ambiguity, gaps in logic, or areas needing clarification."},
                {"role": "user", "content": f"Question: {question}\nStandard Answer: {std_answer}\nIdentify doubts or areas needing improvement."}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are a masterful assistant refining responses based on detailed feedback. Your role is to produce a deeply thoughtful, comprehensive, and accurate revised answer, addressing all highlighted concerns and doubts with precision."},
                {"role": "user", "content": f"Question: {question}\nStandard Answer: {std_answer}\nFeedback: {doubts}\nProvide a refined and improved answer."}
            ]
    elif dataset_type == "MAmmoTH":
        if step == 1:
            return [
                {"role": "system", "content": "You are a rigorous mathematical reviewer. Your role is to critically evaluate the provided solution, questioning its logic and identifying any inaccuracies or unclear reasoning. Scrutinize every detail."},
                {"role": "user", "content": f"Question: {question}\nStandard Answer: {std_answer}\nIdentify issues or suggest areas for improvement."}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are an expert mathematician tasked with refining solutions. Your role is to ensure absolute correctness, clarity, and completeness, addressing all feedback comprehensively."},
                {"role": "user", "content": f"Question: {question}\nStandard Answer: {std_answer}\nFeedback: {doubts}\nRefine and improve the solution."}
            ]
    elif dataset_type == "WizardCoder":
        if step == 1:
            return [
                {"role": "system", "content": "You are a critical code reviewer. Your task is to rigorously analyze the provided code for potential bugs, inefficiencies, and unclear logic. Be extremely skeptical and leave no detail unchecked."},
                {"role": "user", "content": f"Problem: {question}\nStandard Answer: {std_answer}\nHighlight doubts, issues, or suggestions for improvement."}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are an expert programmer refining code. Your role is to address all identified issues, ensuring the code is flawless, efficient, and easy to understand."},
                {"role": "user", "content": f"Problem: {question}\nStandard Answer: {std_answer}\nFeedback: {doubts}\nRefactor and enhance the code."}
            ]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Infinity-Instruct", "MAmmoTH", "WizardCoder"], help="Type of dataset being processed.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Original Q + standard answer JSONL.")
    parser.add_argument("--llm1_model", type=str, required=True, help="Model path for LLM1.")
    parser.add_argument("--llm2_model", type=str, required=True, help="Model path for LLM2.")
    parser.add_argument("--port1", type=int, default=8000, help="Port for LLM1.")
    parser.add_argument("--port2", type=int, default=8001, help="Port for LLM2.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs.")
    args = parser.parse_args()

    # Step 1: <q, a_std> -> LLM2 -> t
    print("[INFO] Step1: <q, a_std> -> LLM2 -> t")
    process_llm2 = start_vllm_server(args.llm2_model, args.port2, args.gpu)
    data_list = list(read_jsonl(args.input_jsonl))
    step1_data = []
    api_base_llm2 = f"http://localhost:{args.port2}"

    for record in data_list:
        q = record["input"]
        std_answer = record["output"] if "output" in record else record.get("std_answer", "")
        idx = record.get("idx", None)
        messages = construct_messages(args.dataset_type, step=1, question=q, std_answer=std_answer)
        try:
            t = chat_completion(api_base_llm2, args.llm2_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            t = f"[LLM2 Error] {str(e)}"
        step1_data.append({"idx": idx, "q": q, "a_std": std_answer, "t": t})

    step1_file = os.path.join("outputs", f"type3_step1_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step1_file, step1_data)
    print(f"[INFO] Step1 output saved to {step1_file}")
    stop_vllm_server(process_llm2)

    # Step 2: <q, a_std, t> -> LLM1 -> a'
    print("[INFO] Step2: <q, a_std, t> -> LLM1 -> a'")
    process_llm1 = start_vllm_server(args.llm1_model, args.port1, args.gpu)
    step2_data = []
    api_base_llm1 = f"http://localhost:{args.port1}"

    step1_data_reloaded = list(read_jsonl(step1_file))
    for record in step1_data_reloaded:
        q = record["q"]
        std_answer = record["a_std"]
        t = record["t"]
        idx = record["idx"]
        messages = construct_messages(args.dataset_type, step=2, question=q, std_answer=std_answer, doubts=t)
        try:
            a_prime = chat_completion(api_base_llm1, args.llm1_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            a_prime = f"[LLM1 Error] {str(e)}"

        step2_data.append({"idx": idx, "q": q, "a_prime": a_prime})

    step2_file = os.path.join("outputs", f"type3_step2_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step2_file, step2_data)
    print(f"[INFO] Step2 (final) output saved to {step2_file}")
    stop_vllm_server(process_llm1)

    print("[INFO] Type3 pipeline complete.")

if __name__ == "__main__":
    main()
