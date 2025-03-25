# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
import hashlib
import json
from openai import OpenAI

# 自部署 R1  AIME2024 准确率可复现
client = OpenAI(api_key="sk-Zzu4LQYbNTNsDoyhaVKoUvtuk8oEruffE6CjhmM2fL6vOzar", base_url="http://oneapi-bcloud.bc-inner.com/v1")

api_cache = dict()
cache_dir = "/data_train/search/zengziyang/projects/deepscaler/verl/verl/trainer/api_cache.jsonl"
# print(os.getenv("CHECKPOINT_SAVE"))
# cache_dir = os.getenv("CHECKPOINT_SAVE") + "/api_cache.jsonl"

with open(cache_dir , "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        api_cache.update(line)

def get_response_from_r1(messages, idx):
    def string_to_sha256(string):
        sha256 = hashlib.sha256()
        sha256.update(string.encode('utf-8'))
        return sha256.hexdigest()

    def call_func(meassages, idx):
        key = "deepseek-r1" + "_" + json.dumps(messages, ensure_ascii=False) + "_" + str(idx)
        key = string_to_sha256(key)
        # key = str(idx) + "_" + messages[0]["content"] 
        if key in api_cache.keys():
            response = api_cache[key]
            print(f'R1 cache hit: {messages[0]["content"][:10]+ "_" + str(idx)}')
            return response
        else:
            cnt = 0
            while True:
                try:
                    response = client.chat.completions.create(
                        model="deepseek-r1",
                        messages=meassages,
                        stream=False,
                        temperature=0.6,   
                        max_tokens=32768, 
                        top_p=0.95,
                        timeout=300000,
                    )
                    response = response.to_dict()
                    if "</think>" not in response["choices"][0]["message"]["content"] :
                        raise Exception("</think> not in response, content error")
                    
                    api_cache[key] = response
                    with open(cache_dir, "a", encoding="utf-8") as f:
                        f.write(json.dumps({key: response}, ensure_ascii=False) + "\n")
                    return response
                except Exception as e:
                    cnt += 1
                    time.sleep(1)
                    print(f"R1 request failed: {e}, retrying...{cnt}")
                    if cnt == 10:
                        return {"id": "02173950662598972d13e4ddb1e80f3c77d8f65f2b0ff2741401c", "choices": [{"finish_reason": "stop", "index": 0, "logprobs": None, "message": {"content": "call error"}}], "created": 1739506651, "model": "deepseek-r1-250120", "object": "chat.completion", "usage": {"completion_tokens": 871, "prompt_tokens": 799, "total_tokens": 1670, "completion_tokens_details": {"reasoning_tokens": 711}, "prompt_tokens_details": {"cached_tokens": 0}}}
                    
    # 把所有role = system的消息删除
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "system":
            del messages[i]
        
    model_response = call_func(messages, idx)
    return model_response["choices"][0]["message"]["content"]
        

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        dataset = pd.read_parquet(config.data.output_path)
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = []  # We'll reshape at the end

        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            
            # Repeat the batch n_samples times
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
                
            index_lst = list(range(config.data.n_samples)) * len(batch_chat_lst)
            
            # ------------------------------
            with ThreadPoolExecutor(max_workers=10) as executor:
                output_text_unpad = list(tqdm(executor.map(get_response_from_r1, repeated_chat_lst, index_lst), total=len(repeated_chat_lst)))
            # ------------------------------
            output_lst.extend(output_text_unpad)

        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()

        # Add to the data frame
        dataset['responses'] = output_lst

        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
    
    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    passes_16 = 0
    total = len(dataset)
    total_scores = []
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            score = reward_fn(r, ground_truth)
            score_lst.append(score)
        max_score = np.max(score_lst)
        max_score_16 = np.max(score_lst[:16])
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1
        if max_score_16 == 1:
            passes_16 += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_16 = passes_16 / total
    pass_at_1 = np.mean(total_scores)
    dataset['correct'] = total_scores
    dataset.to_parquet(config.data.output_path)
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        'pass@16': pass_at_16,
        f'pass@{n_samples}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn

if __name__ == '__main__':
    main()
