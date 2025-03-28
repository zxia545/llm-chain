#!/usr/bin/env python
import argparse
import json
import time
from utils import (
    read_jsonl,
    write_jsonl,
    start_vllm_server,
    stop_vllm_server,
    chat_completion,
    create_output_directory,
)

def construct_eval_prompt():
    return (
        "You are an evaluator assessing the quality of an answer and explanation in response to a question. "
        "Your goal is to determine how factually accurate, reasonable, and non-misleading the response is. "
        "Assign a score from 1 to 5 based on the following criteria:\n\n"
        "1 = The response is clearly misleading, mostly incorrect, or seriously flawed\n"
        "2 = The response has significant factual issues or confusion\n"
        "3 = The response is ambiguous, partially correct, or hard to verify\n"
        "4 = The response is mostly accurate with minor issues\n"
        "5 = The response is accurate, reliable, and not misleading in any way\n\n"
        "Your response must ONLY be a dictionary in the following format:\n"
        "{\n  \"score\": your_score (1-5),\n  \"explanation\": your_reasoning\n}"
    )

def construct_eval_user_prompt(data_entry):
    question = data_entry.get("input", "")
    response = data_entry.get("output", "")
    prompt = (
        f"Question:\n{question}\n\n"
        f"Response (Answer + Explanation):\n{response}\n\n"
        "Evaluate how accurate and trustworthy the response is. "
        "Provide a score (1-5) and a detailed explanation of your reasoning. "
        "Your response must be a dictionary like this:\n"
        "{\n  \"score\": your_score (1-5),\n  \"explanation\": your_reasoning\n}"
    )
    return prompt

import json
import re

def extract_score(response_text):
    """
    Attempt to extract the score and explanation from the LLM's response.
    First, try to parse the response as JSON. If that fails, use a regex to extract a score from
    a pattern like: "score": xx,
    Returns (score, explanation) if a valid score (1-5) is found; otherwise, (None, "").
    """
    # Try parsing as JSON
    try:
        eval_dict = json.loads(response_text)
        score = eval_dict.get("score")
        explanation = eval_dict.get("explanation", "")
        if isinstance(score, int):
            if 1 <= score <= 5:
                return score, explanation
        elif isinstance(score, str):
            try:
                score_val = int(score.strip())
                if 1 <= score_val <= 5:
                    return score_val, explanation
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: use regex to search for pattern like "score": xx,
    match = re.search(r'"score"\s*:\s*(\d+)', response_text)
    if match:
        try:
            score_val = int(match.group(1))
            if 1 <= score_val <= 5:
                return score_val, ""
        except Exception:
            pass

    return None, ""


def process_item(data_item, api_base, model_name, max_tokens=256, temperature=0.7, max_retries=10):
    """
    Processes a single JSON data entry by constructing the evaluation prompt and sending it to the LLM.
    If a valid score is not returned, it will retry up to max_retries times.
    """
    system_prompt = construct_eval_prompt()
    user_prompt = construct_eval_user_prompt(data_item)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    attempts = 0
    score = None
    eval_response = ""
    explanation = ""
    
    while attempts < max_retries and score is None:
        eval_response = chat_completion(
            api_base=api_base,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        score, explanation = extract_score(eval_response)
        attempts += 1
        if score is None:
            time.sleep(1)  # Wait a moment before retrying

    if score is None:
        # Assign a default score if no valid response is generated.
        score = 1
        explanation = "No valid evaluation output received after multiple attempts."

    data_item["eval_feedback"] = eval_response
    data_item["eval_score"] = score
    data_item["eval_attempts"] = attempts
    data_item["eval_explanation"] = explanation
    return data_item

def filter_jsonl(input_file, output_file, filter_rate, api_base, model_name, max_tokens=256, temperature=0.7):
    """
    Reads the input JSONL file, evaluates each data entry using the LLM evaluation prompt,
    sorts the entries by their evaluation score (highest first), and writes out the top 
    'filter_rate' fraction of items to the output JSONL file.
    """
    processed_items = []
    for data_item in read_jsonl(input_file):
        processed = process_item(data_item, api_base, model_name, max_tokens, temperature)
        processed_items.append(processed)
    
    # Sort the processed items in descending order of evaluation score.
    processed_items.sort(key=lambda x: x.get("eval_score", 0), reverse=True)
    
    # Determine the number of items to retain based on the filter rate.
    retain_count = int(len(processed_items) * filter_rate)
    filtered_items = processed_items[:retain_count]
    
    write_jsonl(output_file, filtered_items)
    print(f"[INFO] Filtered {len(filtered_items)} items written to {output_file}.")

def main():
    parser = argparse.ArgumentParser(description="Filter a JSONL file using LLM evaluation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--filter_rate", type=float, default=0.7, help="Fraction of top scored items to retain (e.g., 0.7 for top 70%).")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the LLM model to host.")
    parser.add_argument("--model_name", type=str, default="my-llm-model", help="The served model name for vLLM.")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the vLLM server on.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens for LLM responses.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for LLM responses.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index for the vLLM server.")
    
    args = parser.parse_args()
    api_base = f"http://localhost:{args.port}"
    
    # Start the vLLM server.
    server_process = start_vllm_server(args.llm_model_path, args.model_name, args.port, gpu=args.gpu)
    
    try:
        filter_jsonl(
            input_file=args.input_file,
            output_file=args.output_file,
            filter_rate=args.filter_rate,
            api_base=api_base,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    finally:
        stop_vllm_server(server_process)

if __name__ == "__main__":
    main()
