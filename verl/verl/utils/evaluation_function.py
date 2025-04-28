import json
from verl.utils.verification_prompts import prompt_dict, THINK_HINT
import os
import numpy as np
from typing import Any, Callable
from collections import defaultdict


def parse_parquet(config, data_path: str, model: str, prompt_version: str, convert_type='parquet',if_save=False, save_path=None):
    import pandas as pd
    import re
    dataset_name = os.path.basename(data_path).split('.')[0]
    data = pd.read_parquet(data_path).to_records()
    refined_data = []
    n_sample = len(data['responses'][0])
    for item in data:
        for sample_id, response, correct in enumerate(zip(item['responses'].tolist(), item['correct'].tolist())):
            assert type(response) == str
            search_pattern = r'<think>(.*?)</think>(.*)'
            match = re.search(search_pattern, response)
            think_content = match.group(1)
            solution = match.group(2)
            
            if config.data.prompt_type == 'with_think':
                item = {
                "question": item['prompt'][0]['content'],
                "data_source": item['data_source'],
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_dict[prompt_version].format(
                            think_hint=THINK_HINT,
                            question_pair="Question: {}\n".format(item['prompt'][0]['content']),
                            think_pair="Think: {}\n".format(think_content),
                            solution_pair="Solution: {}\n".format(solution)
                        )
                    }
                ],
                "prompt_type": 'with_think',
                "problem_id": item['extra_info']['index'],
                "sample_id": sample_id,
                "ability": item['ability'],
                "reward_model": item['reward_model'],
                "extra_info": item['extra_info'],
                'verification_model': model,
                'response': response,
                'correctness': int(correct)
                }
                refined_data.append(item)
                
            elif config.data.prompt_type == 'without_think':
                item = {
                "question": item['prompt'][0]['content'],
                "data_source": item['data_source'],
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_dict[prompt_version].format(
                            think_hint="",
                            question_pair="Question: {}\n".format(item['prompt'][0]['content']),
                            solution_pair="Solution: {}\n".format(solution)
                        )
                    }
                ],
                "prompt_type": 'without_think',
                "problem_id": item['extra_info']['index'],
                "sample_id": sample_id,
                "ability": item['ability'],
                "reward_model": item['reward_model'],
                "extra_info": item['extra_info'],
                'verification_model': model,
                'response': response,
                'correctness': int(correct)
                }
                refined_data.append(item)

    refined_data = pd.DataFrame(refined_data)
    return refined_data, n_sample, dataset_name

def bootstrap_metric(data: list[dict[str, Any]],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]

def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        if d[val_key] == 1:
            # only consider correct verification
            vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

