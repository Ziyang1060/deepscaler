"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any
import json
from copy import deepcopy

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset

prompt_type = {
    'with_think':{
        "MATH": "You are a math master. Given a math problem and its solution, verifying correctness step by step. The solution contains reflective and self-corrective thinking wrapped in two special tokens: <reflection> and </reflection>, which cound be used to support your judgement. At the end of the solution verification, write it in the form \"Verification\": X, where X is either Yes or No, which represent whether the answer is correct.\nQuestion: {question}\nSolution: {solution}",
        'MULTI_CHOICE': "Given a multiple choice question and its solution, verifying correctness of answer step by step. The solution contains reflective and self-corrective thinking wrapped in two special tokens: <reflection> and </reflection>, which cound be used to support your judgement. At the end of the solution verification, write it in the form \"Verification\": X, where X is either Yes or No, which represent whether the answer is correct.\nQuestion: {question}\nSolution: {solution}"
    },
    'without_think': {
        "MATH": "Given a math problem and its solution, verifying correctness step by step. At the end of the solution verification, write it in the form \"Verification\": X, where X is either Yes or No, which represent whether the answer is correct.\nQuestion: {question}\nSolution: {solution}",
        "MULTI_CHOICE": "Given a multiple choice question and its solution, verifying correctness of answer step by step. At the end of the solution verification, write it in the form \"Verification\": X, where X is either Yes or No, which represent whether the answer is correct.\nQuestion: {question}\nSolution: {solution}"
    }
}



def add_gpqa_to_custom_dataset(policy_model='DeepSeek-R1-Distill-Qwen-7B'):
    gpqa_path = "/data_train/code/search/zhaoyufan/workspace/LIMO/eval/outputs/model_load/limo-qwen-distill-1dot5b-gpqa-pass32-0325/model_load/gpqa/modified_test_qwen-instruct_t0.6_k32_s0_e198.jsonl"
    data_list = []
    with open(gpqa_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    refined_data_list = []
    for idx, data in enumerate(data_list):
        item = {
            'dataset': 'gpqa',
            'ability': 'multi_choice',
            'problem_id': idx,
            'problem': data['question'],
            'answer': data['gold_answer'],
            'policy_model': policy_model,
            'responses': data['generated_responses'],
            'correctness': [int(item) for item in data['answers_correctness']]
        }
        refined_data_list.append(item)

    
    with open("/data_train/code/search/zhaoyufan/workspace/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-1.5B_gather.json", 'r', encoding='utf-8') as f:
        gather_data = json.load(f)
    gather_data['gpqa'] = refined_data_list

    with open('/data_train/code/search/zhaoyufan/workspace/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-1.5B_gather_0326.json', 'w', encoding='utf-8') as f:
        json.dump(gather_data, f, indent=4)

def load_gpqa(file_path="/data_train/code/search/zhaoyufan/workspace/LIMO/eval/data/gpqa/test.jsonl"):
    question_format = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Question: {question}"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    refined_data_list = []
    for idx, data in enumerate(data_list):
        item = {
            'data_source': 'gpqa',
            'prompt': [{'role': 'user', 'content': question_format.format(question=data['question'])}],
            'ability': 'multi_choice',
            'reward_model': {'ground_truth': data['answer'], 'style': 'multiple_choice'},
            'extra_info': {'index': idx, 'split': 'test'},
        }
        refined_data_list.append(item)
    data = pd.DataFrame(refined_data_list)
    data.to_parquet("/data_train/code/search/zhaoyufan/deepscaler/data/gpqa.parquet")
    return
    


def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    custom_data = []
    custom_dataset_name = []
    for dataset_name, data_list in data_dict.items():
        tmp_list = []
        custom_dataset_name.append(dataset_name)
        for data_id, item in enumerate(data_list):
            for sample_id, response, correct in zip(range(len(item['responses'])), item['responses'], item['correctness']):
                if custom_dataset_name == 'gpqa':
                    tmp_list.append({
                        "dataset_name": custom_dataset_name,
                        "prompt": [{'role': 'user', 'content': prompt_type['with_think']['MULTI_CHOICE'].format(question=item['problem'], solution=response.replace("<think>", "<reflection>").replace("</think>", "</reflection>"))}],
                        'prompt_type': 'with_think',
                        "ability": "MULTI_CHOICE",
                        "prompt_id": data_id,
                        "sample_id": sample_id,
                        "correctness": correct,
                        "answer": item['answer']
                    })
                    tmp_list.append({
                        "dataset_name": custom_dataset_name,
                        "prompt": [{'role': 'user', 'content': prompt_type['without_think']['MULTI_CHOICE'].format(question=item['problem'], solution=response.split("</think>\n")[-1])}],
                        'prompt_type': 'without_think',
                        "ability": "MULTI_CHOICE",
                        "prompt_id": data_id,
                        "sample_id": sample_id,
                        "correctness": correct,
                        "answer": item['answer']
                    })
                else:
                    tmp_list.append({
                        "dataset_name": custom_dataset_name,
                        "prompt": [{'role': 'user', 'content': prompt_type['with_think']['MATH'].format(question=item['problem'], solution=response.replace("<think>", "<reflection>").replace("</think>", "</reflection>"))}],
                        'prompt_type': 'with_think',
                        "ability": "MATH",
                        "prompt_id": data_id,
                        "sample_id": sample_id,
                        "correctness": correct,
                        "answer": item['answer']
                    })
                    tmp_list.append({
                        "dataset_name": custom_dataset_name,
                        "prompt": [{'role': 'user', 'content': prompt_type['without_think']['MATH'].format(question=item['problem'], solution=response.split("</think>\n")[-1])}],
                        'prompt_type': 'without_think',
                        "ability": "MATH",
                        "prompt_id": data_id,
                        "sample_id": sample_id,
                        "correctness": correct,
                        "answer": item['answer']
                    })
        custom_data.append(deepcopy(tmp_list))
    return custom_data, custom_dataset_name
                


if __name__ == "__main__":
    # add_gpqa_to_custom_dataset()
    # add_gpqa_to_custom_dataset(policy_model="DeepSeek-R1-Distill-Qwen-1.5B")
    # load_gpqa()

    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('~/deepscaler/data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir)

    curstom_dataset_path = "/data_train/code/search/zhaoyufan/workspace/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-7B_gather_0326.json"
    custom_datasets, custom_datasets_name = load_custom_dataset(file_path=curstom_dataset_path)

    for dataset, dataset_name in zip(custom_datasets, custom_datasets_name):
        dataset_df = pd.DataFrame(dataset)
        dataset_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} data size:", len(dataset))
    