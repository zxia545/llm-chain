# utils.py
import os
import time
import json
import requests
from typing import Dict, Any
import subprocess
from openai import OpenAI
import json

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def chat_completion(api_base: str, model_name: str, messages: list, max_tokens=256, temperature=0.7):
    """
    Generic helper that uses the new openai client interface to get a chat completion.
    """
    
    if '/v1' not in api_base:
        api_base = api_base + '/v1'
    
    client = OpenAI(base_url=api_base, api_key="xxx")  # point to the local vLLM server
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content



def start_vllm_server(model_path: str, model_name: str, port: int, gpu: int = 1):
    """
    Launches a vLLM OpenAI API server via subprocess.
    model_path: The path or name of the model you want to host
    port: Which port to host on
    gpu: The tensor-parallel-size (number of GPUs)
    """
    # Command to activate conda environment and start the server
    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f"--served-model-name={model_name}",
        f'--tensor-parallel-size={gpu}',
        f"--gpu-memory-utilization=0.85",
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False)
    
    wait_for_server(f"http://localhost:{port}", 600)
    
    print(f"[INFO] Started vLLM server for model '{model_path}' on port {port} (GPU={gpu}).")

    return process





def wait_for_server(url: str, timeout: int = 600):
    """
    Polls the server's /models endpoint until it responds with HTTP 200 or times out.
    """
    start_time = time.time()
    while True:
        try:
            r = requests.get(url + "/v1/models", timeout=3)
            if r.status_code == 200:
                print("[INFO] vLLM server is up and running.")
                return
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise RuntimeError(f"[ERROR] Server did not start at {url} within {timeout} seconds.")
        time.sleep(2)
        
def stop_vllm_server(process):
    process.terminate()
    process.wait()
    print("[INFO] Stopped vLLM server.")



def create_output_directory(model_name: str):
    """
    Creates the output directory named after the LLM model, if it doesn't exist.
    Returns the path to that directory.
    """
    output_dir = os.path.join("outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
