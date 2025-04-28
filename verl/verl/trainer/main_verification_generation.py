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
import re
import numpy as np

import os
from tabulate import tabulate
from functools import partial


os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['HYDRA_FULL_ERROR'] = "1"

from verl.utils.model import compute_position_id_with_mask

import pandas as pd
import hydra

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from sklearn.metrics import confusion_matrix
from verl.utils.evaluation_function import parse_parquet, bootstrap_metric, calc_maj_val


    
@hydra.main(config_path='config', config_name='verification_generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    output_dir = os.path.join(config.data.output_folder, config.data.prompt_version, config.data.prompt_type)

    # Check if output file already exists
    if os.path.exists(output_dir):
        print(f"Output file {output_dir} already exists. Skipping generation and proceeding to evaluation.")
        dataset, n_sample, dataset_name = parse_parquet(config=config, data_path=config.data.path, model=os.path.basename(config.model.path), prompt_version=config.data.prompt_version)
        origin_dataset = pd.read_parquet(os.path.join(output_dir, f'{dataset_name}.parquet'))

    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset, n_sample, dataset_name = parse_parquet(config=config, data_path=config.data.path, model=os.path.basename(config.model.path), prompt_version=config.data.prompt_version)
        origin_dataset = pd.read_parquet(config.data.path)

        chat_lst = dataset[config.data.prompt_key].tolist()

        # chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = []  # We'll reshape at the end

        from tqdm import tqdm
        for batch_idx in tqdm(range(num_batch), desc='processing num_batch'):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            
            
            # ------------------------------
            inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=config.rollout.prompt_length,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            
            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            
            # Generate all samples at once
            print(len(data.batch['input_ids']))
            output = wg.generate_sequences(data)
            # Remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)

            # Remove padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))
            # ------------------------------
            
            output_lst.extend(output_text_unpad)


        # Reshape the output list to match the original dataset
        output_lst = np.array(output_lst).reshape(len(origin_dataset), n_sample).tolist()
        chat_lst = np.array(chat_lst).reshape(len(origin_dataset), n_sample).tolist()
        
        # Add to the data frame
        origin_dataset['verification_response'] = output_lst 
        origin_dataset['chat_lst'] = chat_lst
        origin_dataset['prompt_type'] = config.data.prompt_type
        
        # Write to a new parquet
        output_dir = os.path.join(config.data.output_folder, config.data.prompt_version, config.data.prompt_type)
        makedirs(output_dir, exist_ok=True)
        origin_dataset.to_parquet(os.path.join(output_dir, f'{dataset_name}.parquet'))
        



    dataset = origin_dataset
    output_dir = os.path.join(config.data.output_folder, config.data.prompt_version, config.data.prompt_type)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['verification_response']  # Using the generated responses
    correctnesss = dataset['correct']
    parse_answer_lst = dataset['parse_answer']

    total = len(dataset)
    total_scores = []

    for i in range(total):
        verification_response_lst = responses[i].tolist()
        prompt = prompts[i]
        correctness = correctnesss[i]
        
        def parse_verification(veri_res: str):
            try:
                PARSE_PATTERN = r"(?i)Verification[ \t]*:[ \t]*(Yes|No)"
                match = re.search(PARSE_PATTERN, veri_res)
                extracted_answer = match.group(1) if match else None
                if extracted_answer.lower() == 'yes':
                    return 1
                return 0
            # all failures return false
            except TypeError:
                print("Error in extracting verification: {}".format(veri_res))
                return -1
        score_lst = []
        for r, correct in zip(verification_response_lst, correctness):
            score = parse_verification(r)
            score_lst.append(int(correct == score))
            
        total_scores.append(score_lst)
    

    dataset['verification_correctness'] = np.array(total_scores)
    dataset.to_parquet(os.path.join(output_dir, f'{dataset_name}.parquet'))
    
    # calculatle confusion matrix
    all_groundtrue = []
    all_predict = []
    for index, row in dataset.iterrows():
        all_groundtrue.extend(row['correct'].tolist())
        all_predict.extend(row['verification_correctness'].tolist())
    conf_matrix = confusion_matrix(all_groundtrue, all_predict)
    total_samples = conf_matrix.sum()
    
    n = config.data.n
    lst_bon_mean = []
    lst_won_mean = []
    lst_maj_n = []
    for idx, item in dataset.iterrows():
        data = []
        for parse_answer, verification_correctness in zip(item['parse_answer'].tolist(), item['verification_correctness'].tolist()):
            data.append({'pred': parse_answer, 'val': verification_correctness})
            
        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
            data,
            subset_size=n,
            reduce_fns=[
                lambda arr: np.max([d["val"] for d in arr]),
                lambda arr: np.min([d["val"] for d in arr]),
            ]
        )
        
        maj_val = calc_maj_val(data[0:n], vote_key='pred', val_key='val')
        reward_fn = select_reward_fn(item[config.data.data_source_key])
        maj_val_score = reward_fn('boxed{TMP}'.replace("TMP", maj_val), item[config.data.reward_model_key]['ground_truth'])
        
        lst_bon_mean.append(bon_mean)
        lst_won_mean.append(won_mean)
        lst_maj_n.append(maj_val_score.is_correct)
    
    

    row_data = {
        'model_path': config.model.path,
        'dataset_name': config.data.path,
        "TP": conf_matrix[0][0] / total_samples,
        "TN": conf_matrix[1][1] / total_samples,
        "FP": conf_matrix[1][0] / total_samples,
        "FN": conf_matrix[0][1] / total_samples,
        "accuracy": (conf_matrix[0][0] + conf_matrix[1][1]) / total_samples,
        f"Best_of_{n}": np.mean(lst_bon_mean),
        f"Worst_of_{n}": np.mean(lst_won_mean),
        f"Maj_of_{n}": np.mean(lst_maj_n),
    }
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'evaluation.csv')
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
    # if data_source == 'gpqa':
    #     def gpqa_reward_fn(response, ground_truth):
    #         import re
    #         ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
    #         match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
    #         cur_ans = match.group(1) if match else None
    #         if cur_ans not in ['A', 'B', 'C', 'D']:
    #             print('Error in extracting answer: cur_ans={}'.format(cur_ans))
    #             # print(generated_responses[i])
    #             cur_ans = ""
    #             return 0
    #         if cur_ans == ground_truth:
    #             return 1
    #         return 0
    #     return gpqa_reward_fn
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn

if __name__ == '__main__':
    main()
