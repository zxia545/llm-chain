# type2.py
import argparse
import os
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="Original Q input JSONL.")
    parser.add_argument("--llm1_model", type=str, required=True, help="Model path for LLM1.")
    parser.add_argument("--llm2_model", type=str, required=True, help="Model path for LLM2.")
    parser.add_argument("--port1", type=int, default=8000, help="Port for LLM1.")
    parser.add_argument("--port2", type=int, default=8001, help="Port for LLM2.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs.")
    args = parser.parse_args()

    # ---------------------------------------------------
    # Step1: q -> LLM1 -> a
    # ---------------------------------------------------
    print("[INFO] Step1: q -> LLM1 -> a")
    process_llm1_step1 = start_vllm_server(args.llm1_model, args.port1, args.gpu)

    data_list = list(read_jsonl(args.input_jsonl))
    step1_data = []
    api_base_llm1 = f"http://localhost:{args.port1}"
    for record in data_list:
        q = record["input"]
        idx = record.get("idx", None)
        messages = [
            {"role": "system", "content": "You are LLM1. Provide your direct answer."},
            {"role": "user", "content": q}
        ]
        try:
            a = chat_completion(api_base_llm1, args.llm1_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            a = f"[LLM1 Error] {str(e)}"
        step1_data.append({"idx": idx, "q": q, "a": a})

    step1_file = os.path.join("outputs", f"type2_step1_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step1_file, step1_data)
    print(f"[INFO] Step1 output saved to {step1_file}")
    stop_vllm_server(process_llm1_step1)

    # ---------------------------------------------------
    # Step2: <q, a> -> LLM2 -> t
    # ---------------------------------------------------
    print("[INFO] Step2: <q, a> -> LLM2 -> t")
    process_llm2 = start_vllm_server(args.llm2_model, args.port2, args.gpu)

    step2_data = []
    step1_data_reloaded = list(read_jsonl(step1_file))
    api_base_llm2 = f"http://localhost:{args.port2}"
    for record in step1_data_reloaded:
        q = record["q"]
        a = record["a"]
        idx = record["idx"]
        messages = [
            {"role": "system", "content": "You are LLM2. Read the question and the answer. Provide doubts or questions."},
            {"role": "user", "content": f"Question: {q}\nAnswer: {a}\nWhat doubts or questions do you have?"}
        ]
        try:
            t = chat_completion(api_base_llm2, args.llm2_model, messages, max_tokens=256, temperature=0.2)
        except Exception as e:
            t = f"[LLM2 Error] {str(e)}"
        step2_data.append({"idx": idx, "q": q, "a": a, "t": t})

    step2_file = os.path.join("outputs", f"type2_step2_{os.path.basename(args.input_jsonl)}")
    write_jsonl(step2_file, step2_data)
    print(f"[INFO] Step2 output saved to {step2_file}")
    stop_vllm_server(process_llm2)

    # ---------------------------------------------------
    # Step3: <q, a, t> -> LLM1 -> a'
    # ---------------------------------------------------
    print("[INFO] Step3: <q, a, t> -> LLM1 -> a'")
    process_llm1_step3 = start_vllm_server(args.llm1_model, args.port1, args.gpu)  # reuse port1 for LLM1
    step3_data = []
    step2_data_reloaded = list(read_jsonl(step2_file))

    for record in step2_data_reloaded:
        q = record["q"]
        a = record["a"]
        t = record["t"]
        idx = record["idx"]
        messages = [
            {"role": "system", "content": "You are LLM1. You have your old answer and LLM2's doubts. Revise your answer."},
            {"role": "user", "content": f"Question: {q}\nPrevious Answer: {a}\nLLM2 Doubts: {t}\nRevise your answer now."}
        ]
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
