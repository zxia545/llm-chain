import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion

def construct_messages(dataset_type, step, question=None, answer=None, doubts=None):
    """
    Construct role-based messages for LLM interactions based on dataset type and step.
    The prompts are tailored to improve quality by reinforcing focus, structure, and depth,
    encouraging thoughtful doubts and meaningful improvements.
    """
    if dataset_type == "Infinity-Instruct":
        if step == 1:
            return [
                {"role": "system", "content": "You are a highly analytical and contextually aware assistant. Provide a detailed, comprehensive answer that directly addresses the query, offering examples, insights, and a structured explanation. Ensure every aspect of the query is covered with depth and clarity."},
                {"role": "user", "content": question}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are a critical reviewer tasked with identifying ambiguities, gaps, or areas for improvement in an answer. Question the logic, demand clarity, and suggest meaningful enhancements to ensure the answer fully satisfies the query."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nWhat clarifications, corrections, or improvements would you suggest?"}
            ]
        elif step == 3:
            return [
                {"role": "system", "content": "You are a thoughtful and thorough assistant revising an answer based on detailed feedback. Ensure your response is precise, comprehensive, and directly addresses all highlighted concerns."},
                {"role": "user", "content": f"Question: {question}\nPrevious Answer: {answer}\nFeedback: {doubts}\nRevise the answer to address the feedback comprehensively and clearly."}
            ]
    elif dataset_type == "MAmmoTH":
        if step == 1:
            return [
                {"role": "system", "content": "You are an expert mathematician delivering a step-by-step solution. Provide clear logic, detailed explanations, and illustrative examples to ensure complete understanding."},
                {"role": "user", "content": question}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are a meticulous mathematical reviewer. Analyze the solution critically, highlighting logical inconsistencies, numerical errors, or unclear steps. Suggest meaningful refinements."},
                {"role": "user", "content": f"Question: {question}\nSolution: {answer}\nIdentify any issues, errors, or areas for improvement."}
            ]
        elif step == 3:
            return [
                {"role": "system", "content": "You are a mathematical problem-solver refining solutions. Address all identified feedback, ensuring logical consistency, numerical accuracy, and clarity in your response."},
                {"role": "user", "content": f"Question: {question}\nPrevious Solution: {answer}\nFeedback: {doubts}\nImprove the solution, addressing the feedback thoroughly."}
            ]
    elif dataset_type == "WizardCoder":
        if step == 1:
            return [
                {"role": "system", "content": "You are a coding expert solving programming challenges. Provide correct, efficient, and well-explained solutions, detailing your thought process and ensuring clarity."},
                {"role": "user", "content": question}
            ]
        elif step == 2:
            return [
                {"role": "system", "content": "You are a critical code reviewer analyzing a solution. Highlight inefficiencies, logical flaws, or unclear sections. Provide actionable suggestions for improvement."},
                {"role": "user", "content": f"Problem: {question}\nCode: {answer}\nWhat corrections, optimizations, or clarifications do you recommend?"}
            ]
        elif step == 3:
            return [
                {"role": "system", "content": "You are a programming expert refining code. Implement corrections and optimizations, ensuring the code is efficient, accurate, and easy to understand."},
                {"role": "user", "content": f"Problem: {question}\nPrevious Code: {answer}\nFeedback: {doubts}\nRefactor and enhance the code based on the feedback."}
            ]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["Infinity-Instruct", "MAmmoTH", "WizardCoder"], help="Type of dataset being processed.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Original Q input JSONL.")
    parser.add_argument("--llm1_model", type=str, required=True, help="Model path for LLM1.")
    parser.add_argument("--llm2_model", type=str, required=True, help="Model path for LLM2.")
    parser.add_argument("--port1", type=int, default=8000, help="Port for LLM1.")
    parser.add_argument("--port2", type=int, default=8001, help="Port for LLM2.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs.")
    args = parser.parse_args()

    # Step 1: q -> LLM1 -> a
    print("[INFO] Step1: q -> LLM1 -> a")
    process_llm1_step1 = start_vllm_server(args.llm1_model, args.port1, args.gpu)

    data_list = list(read_jsonl(args.input_jsonl))
    step1_data = []
    api_base_llm1 = f"http://localhost:{args.port1}"
    for record in data_list:
        q = record["input"]
        idx = record.get("idx", None)
        messages = construct_messages(args.dataset_type, step=1, question=q)
        try:
            a = chat_completion(api_base_llm1, args.llm1_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            a = f"[LLM1 Error] {str(e)}"
        step1_data.append({"idx": idx, "q": q, "a": a})

    step1_file = os.path.join("outputs", f"type2_step1_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step1_file, step1_data)
    print(f"[INFO] Step1 output saved to {step1_file}")
    stop_vllm_server(process_llm1_step1)

    # Step 2: <q, a> -> LLM2 -> t
    print("[INFO] Step2: <q, a> -> LLM2 -> t")
    process_llm2 = start_vllm_server(args.llm2_model, args.port2, args.gpu)

    step2_data = []
    step1_data_reloaded = list(read_jsonl(step1_file))
    api_base_llm2 = f"http://localhost:{args.port2}"
    for record in step1_data_reloaded:
        q = record["q"]
        a = record["a"]
        idx = record["idx"]
        messages = construct_messages(args.dataset_type, step=2, question=q, answer=a)
        try:
            t = chat_completion(api_base_llm2, args.llm2_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            t = f"[LLM2 Error] {str(e)}"
        step2_data.append({"idx": idx, "q": q, "a": a, "t": t})

    step2_file = os.path.join("outputs", f"type2_step2_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step2_file, step2_data)
    print(f"[INFO] Step2 output saved to {step2_file}")
    stop_vllm_server(process_llm2)

    # Step 3: <q, a, t> -> LLM1 -> a'
    print("[INFO] Step3: <q, a, t> -> LLM1 -> a'")
    process_llm1_step3 = start_vllm_server(args.llm1_model, args.port1, args.gpu)
    step3_data = []
    step2_data_reloaded = list(read_jsonl(step2_file))

    for record in step2_data_reloaded:
        q = record["q"]
        a = record["a"]
        t = record["t"]
        idx = record["idx"]
        messages = construct_messages(args.dataset_type, step=3, question=q, answer=a, doubts=t)
        try:
            a_prime = chat_completion(api_base_llm1, args.llm1_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            a_prime = f"[LLM1 Error] {str(e)}"
        step3_data.append({"idx": idx, "q": q, "a_prime": a_prime})

    step3_file = os.path.join("outputs", f"type2_step3_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step3_file, step3_data)
    print(f"[INFO] Step3 (final) output saved to {step3_file}")
    stop_vllm_server(process_llm1_step3)

    print("[INFO] Type2 pipeline complete.")

if __name__ == "__main__":
    main()
