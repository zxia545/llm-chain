import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    filename=f'two_stage_pipeline_{time.strftime("%d_%H_%M_%S")}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting the two-stage pipeline script...")

# ---------------------------------------------------------------------------
#  1) CLASSIFICATION PROMPT / MESSAGE
# ---------------------------------------------------------------------------

def construct_classification_prompt():
    """
    Returns a system prompt instructing the LLM to classify (Q, A) as LOW, HIGH, or REFINE:
      - LOW if the Q&A is not math-related at all, or it is completely nonsense.
      - HIGH if it is already good math Q&A with no extraneous repeated symbols or random characters.
      - REFINE if it is valid math Q&A but has extraneous repeated symbols, random chars, or minor issues.
    The model must respond with exactly one word: LOW, HIGH, or REFINE.
    """
    return (
        "You are a strict classifier of math Q&A quality. You only reply with one word: LOW, HIGH, or REFINE.\n\n"
        "Rules:\n"
        "1) If the question or answer is not math-related, or is completely nonsense, or has no valid content, respond with LOW.\n"
        "2) If the question and answer are clearly math-related and appear correct, well-formed, and do not contain "
        "   extraneous repeated symbols or random characters, respond with HIGH.\n"
        "3) If the question and answer are math-related but have extraneous repeated symbols, random characters, or "
        "   other fixable issues, respond with REFINE.\n\n"
        "Remember: only output LOW, HIGH, or REFINE, with no additional text."
    )

def construct_classification_message(question, answer):
    """
    Constructs the message to send to the LLM for classification.
    """
    system_content = construct_classification_prompt()
    user_content = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Classify as LOW, HIGH, or REFINE (only output one word)."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ---------------------------------------------------------------------------
#  2) REFINING PROMPT / MESSAGE
# ---------------------------------------------------------------------------

def construct_refine_prompt():
    """
    Returns a system prompt to refine question and answer by removing extraneous symbols 
    and random characters. The model should keep math-related content intact.
    """
    return (
        "You are a helpful assistant that refines a math question and answer.\n"
        "You will remove repeated symbols, random characters, or irrelevant text, while preserving "
        "the original meaning and correctness.\n\n"
        "Output format (exactly):\n"
        "Refined Question: <cleaned version of the question>\n"
        "Refined Answer: <cleaned version of the answer>\n\n"
        "No explanations. No extra commentary."
    )

def construct_refine_message(question, answer):
    """
    Builds the conversation for refinement.
    """
    system_content = construct_refine_prompt()
    user_content = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Please refine them by removing extraneous symbols or random characters, but keep them valid and math-related."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ---------------------------------------------------------------------------
#  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, 
                        choices=["Magpie_Math_Instruct_part1", 
                                 "Magpie_Math_Instruct_part2", 
                                 "Magpie_Math_Instruct_part3", 
                                 "Magpie_Math_Instruct_part4"],
                        help="Type of dataset being processed.")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path or name of the model for vLLM.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name of the model for vLLM.")
    parser.add_argument("--output_folder_path", type=str, required=True, 
                        help="Folder name for the output.")
    parser.add_argument("--gpu", type=int, default=1, 
                        help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port for the vLLM server.")
    parser.add_argument("--input_jsonl", type=str, required=True, 
                        help="Path to the input JSONL file.")
    parser.add_argument("--threads", type=int, default=8, 
                        help="Number of threads for concurrent processing.")
    args = parser.parse_args()

    # Step 1: Start the vLLM server
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)

    # Step 2: Read the input JSONL
    data_list = list(read_jsonl(args.input_jsonl))

    # Step 3: Prepare output folder and file
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)
    output_file = os.path.join(args.output_folder_path, f'two_stage_{os.path.basename(args.input_jsonl)}')

    # Check for existing partial results
    if os.path.exists(output_file):
        logger.info(f"[INFO] Loading existing results from {output_file}")
        existing_results = list(read_jsonl(output_file))
        existing_ids = {r["idx"] for r in existing_results}
        data_list = [r for r in data_list if r.get("idx") not in existing_ids]
        logger.info(f"[INFO] {len(data_list)} new records will be processed.")
    else:
        logger.info(f"[INFO] No existing results found. Processing all records.")

    output_data = []
    api_base = f"http://localhost:{args.port}/v1"

    def save_partial_results():
        if output_data:
            write_jsonl(output_file, output_data, append=True)
            output_data.clear()

    def classify_record(question, answer):
        """
        First LLM call: classify as LOW, HIGH, or REFINE.
        """
        messages = construct_classification_message(question, answer)
        response = chat_completion(
            api_base=api_base,
            model_name=args.model_name,
            messages=messages,
            max_tokens=5,      # we only need one word
            temperature=0.0    # to ensure stable classification
        )
        # The model is instructed to respond only with 'LOW', 'HIGH', or 'REFINE'
        label = response.strip().upper()
        return label

    def refine_record(question, answer):
        """
        Second LLM call: refine Q and A if classification is REFINE.
        """
        messages = construct_refine_message(question, answer)
        response = chat_completion(
            api_base=api_base,
            model_name=args.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )
        # Parse "Refined Question:" and "Refined Answer:" from the response
        refined_question = question
        refined_answer = answer
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Refined Question:"):
                refined_question = line.replace("Refined Question:", "").strip()
            elif line.startswith("Refined Answer:"):
                refined_answer = line.replace("Refined Answer:", "").strip()
        return refined_question, refined_answer

    def process_record(record):
        idx = record.get("idx", None)
        question = record["input"]
        answer = record["output"]

        logger.info(f"[INFO] Processing idx={idx}...")

        try:
            # Stage 1: Classification
            label = classify_record(question, answer)

            if label == "LOW":
                # If it's LOW, we skip it (do not add to output).
                logger.info(f"[INFO] idx={idx} => LOW. Skipping.")
                return None

            elif label == "HIGH":
                # If it's HIGH, we keep as is, no refinement needed.
                logger.info(f"[INFO] idx={idx} => HIGH. No refinement needed.")
                return {
                    "idx": idx,
                    "input": question,
                    "output": answer,
                    "label": label
                }

            elif label == "REFINE":
                # Stage 2: Refinement
                logger.info(f"[INFO] idx={idx} => REFINE. Invoking refinement prompt...")
                refined_q, refined_a = refine_record(question, answer)
                if refined_q == question and refined_a == answer:
                    logger.info(f"[INFO] idx={idx} => No changes made during refinement.")
                return {
                    "idx": idx,
                    "input": refined_q,
                    "output": refined_a,
                    "label": label,
                    "is_refined": refined_q != question or refined_a != answer
                }
            else:
                # Unexpected fallback
                logger.warning(f"[WARNING] idx={idx} => Unexpected label: {label}. Skipping.")
                return None

        except Exception as e:
            logger.error(f"[ERROR] Failed to process idx={idx}: {str(e)}")
            return None

    # Step 4: Multithreaded processing
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_record, r) for r in data_list]
        for i, fut in enumerate(futures, start=1):
            result = fut.result()
            if result is not None:
                output_data.append(result)
            # Save partial every 2000
            if i % 2000 == 0:
                save_partial_results()
                elapsed = time.time() - start_time
                logger.warning(f"[INFO] Processed {i} records in {elapsed:.2f}s.")
    
    # Save remaining results
    save_partial_results()
    logger.info(f"[INFO] All results saved to {output_file}")

    # Step 5: Stop vLLM server
    stop_vllm_server(process)

if __name__ == "__main__":
    main()
