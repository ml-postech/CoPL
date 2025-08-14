import argparse
import random

import torch
from datasets import load_dataset, Dataset
import numpy as np
import os

from typing import List
from typing_extensions import Literal, TypeAlias
from datasets import concatenate_datasets

DataSubset: TypeAlias = Literal["both", "helpful", "harmless"]


def random_argmax(values):
    """ a random tie-breaking argmax """
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def random_greater_than_zero(values):
    return (np.random.randn(values.shape[0]) * (values == 0) > 0.0) | (values > 0.0)


def array_to_type(arr):
    return str(int(np.dot(arr, np.array([8, 4, 2, 1]))))


def get_user_type(chosen_ratings, rejected_ratings, augment_type):
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    chosen_rating_values = list()
    rejected_rating_values = list()
    for key in keys:
        chosen_rating_values.append(chosen_ratings[key])
        rejected_rating_values.append(rejected_ratings[key])
    chosen_values = np.asarray(chosen_rating_values)
    rejected_values = np.asarray(rejected_rating_values)
    is_equal = list(chosen_values == rejected_values)
    if augment_type == 'single' or augment_type == '84':
        data_subsets = ['8', '4', '2', '1']
        reversed_labels = {data_subsets[idx]: list(random_greater_than_zero(rejected_values - chosen_values))[idx] for
                           idx in range(len(data_subsets))}
        is_equal = {data_subsets[idx]: is_equal[idx] for idx in range(len(data_subsets))}
        return data_subsets, reversed_labels, is_equal
    else:
        raise ValueError('Invalid augment_type')

def get_hh_rlhf_dataset(
    data_subset: DataSubset,
    split: Literal["train", "test"],
    dataset_size: int = 0,
    data_path="Anthropic/hh-rlhf",
    use_subset_as_dir=True,     # new parameter
    other_subsets=None,
    per_user_examples=1000,
) -> Dataset:
    datasets: List[Dataset] = []
    if other_subsets is None:
        if data_path == "Anthropic/hh-rlhf":
            if data_subset == "harmless" or data_subset == "both":
                datasets.append(
                    load_dataset(
                        "Anthropic/hh-rlhf", data_dir="harmless-base", split=split
                    ).map(lambda data: {"data_subset": "harmless"})
                )
            if data_subset == "helpful" or data_subset == "both":
                datasets.append(
                    load_dataset(
                        "Anthropic/hh-rlhf", data_dir="helpful-base", split=split
                    ).map(lambda data: {"data_subset": "helpful"})
                )
        else:
            if not use_subset_as_dir:  # original version: combine all data subsets within the path
                datasets.append(
                    load_dataset(data_path, split=split).map(
                        lambda data: {"data_subset": data_subset}
                    )
                )
            else:  # new version: use data_subset as subdirectory
                if data_subset == "helpful" or data_subset == "both":
                    datasets.append(
                        load_dataset(
                            data_path, data_dir="helpful", split=split
                        ).map(lambda data: {"data_subset": "helpful"})
                    )
                if data_subset == "harmless" or data_subset == "both":
                    datasets.append(
                        load_dataset(
                            data_path, data_dir="harmless", split=split
                        ).map(lambda data: {"data_subset": "harmless"})
                    )
    else:   # TODO: set subsets here
        if other_subsets == 'ultra_feedback':
            subsets = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
        elif other_subsets == 'single':
            subsets = ['8', '4', '2', '1']
        elif other_subsets == '84':
            subsets = ['8', '4']
        else:  
            subsets = [other_subsets]
        last_idx = -1
        for subset in subsets:
            if data_subset == 'all' or data_subset == subset:
                
            # I want to add a user id to the dataset
            # user id is not in the dataset, so I need to add it
            # User has 1000 examples, I need to assign each user id to 1000 examples randomly
            # 1000 examples are not in order, so I need to shuffle the dataset first
                last_idx += 1
                
                temp = load_dataset(data_path, data_dir=subset, split=split)
                idx = np.arange(len(temp))//per_user_examples
                idx = idx + last_idx
                last_idx = idx[-1]
                np.random.shuffle(idx)

                datasets.append(temp)


    if dataset_size:
        datasets = [
            dataset.select(range(dataset_size // len(datasets))) for dataset in datasets
        ]

    return concatenate_datasets(datasets)

class HHRLHFPreprocessor(object):
    def __init__(self, args, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        
     
        new_examples: dict = {
            "chosen_ids": [],
            "chosen_mask": [],
            "rejected_ids": [],
            "rejected_mask": [],
            "chosen": [],
            "rejected": [],
            "index":[],
            "max_lengths": [],
            "original_idx":[],
        }
        for chosen_text, rejected_text in zip(examples["chosen"], examples["rejected"]):
            
            tokenized_text = self.tokenizer([chosen_text, rejected_text], **self.tokenizer_kwargs)
            # set_trace()
            new_examples["chosen_ids"].append(tokenized_text["input_ids"][0])
            new_examples["chosen_mask"].append(tokenized_text["attention_mask"][0])
            new_examples["rejected_ids"].append(tokenized_text["input_ids"][1])
            new_examples["rejected_mask"].append(tokenized_text["attention_mask"][1])
            new_examples["max_lengths"].append(max(len(tokenized_text["input_ids"][0]),len(tokenized_text["input_ids"][1])))
       
        new_examples["chosen"] = examples["chosen"]
        new_examples["rejected"] = examples["rejected"]
        new_examples["index"] = examples["Index"]
        new_examples["user_type"] = examples["data_subset"]
        new_examples["controversial"] = examples["controversial"]
        new_examples["original_idx"] = examples["original_idx"]

        return new_examples

class HHRLHFPreprocessor(object):
    def __init__(self, args, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        
     
        new_examples: dict = {
            "chosen_ids": [],
            "chosen_mask": [],
            "rejected_ids": [],
            "rejected_mask": [],
            "chosen": [],
            "rejected": [],
            "index":[],
            "max_lengths": [],
            "original_idx":[],
        }
        for chosen_text, rejected_text in zip(examples["chosen"], examples["rejected"]):
            
            tokenized_text = self.tokenizer([chosen_text, rejected_text], **self.tokenizer_kwargs)
            # set_trace()
            new_examples["chosen_ids"].append(tokenized_text["input_ids"][0])
            new_examples["chosen_mask"].append(tokenized_text["attention_mask"][0])
            new_examples["rejected_ids"].append(tokenized_text["input_ids"][1])
            new_examples["rejected_mask"].append(tokenized_text["attention_mask"][1])
            new_examples["max_lengths"].append(max(len(tokenized_text["input_ids"][0]),len(tokenized_text["input_ids"][1])))
       
        new_examples["chosen"] = examples["chosen"]
        new_examples["rejected"] = examples["rejected"]
        new_examples["index"] = examples["Index"]
        new_examples["user_type"] = examples["data_subset"]
        new_examples["controversial"] = examples["controversial"]
        new_examples["original_idx"] = examples["original_idx"]

        return new_examples

def inner_join(original, binarized, augment_type, users, two_two_only=False, filter_equal=False):
    agreed_counter = 0
    controversial_counter = 0
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    user_counter = {key: 0 for key in users.keys()}
    reversed_counter = {key: 0 for key in users.keys()}
    dumb_baseline = {key: 0 for key in users.keys()}
    dumb_controversial_baseline = {key: 0 for key in users.keys()}
    orig_idx = 0
    out_idx = 0
    dataset_dict = {
        'Index': list(),
        'original_idx': list(),
        'prompt': list(),
        'chosen': list(),
        'rejected': list(),
        'data_subset': list(),
        'controversial': list(),
        'reversed': list(),
        'satisfied_subset': list(),
        'survey_options': list(),
    }
    
    for bin_idx in range(len(binarized)):
        while binarized[bin_idx]['prompt'] != original[orig_idx]['instruction']:
            orig_idx += 1
        prompt = binarized[bin_idx]['prompt']
        chosen = binarized[bin_idx]['chosen'][1]['content']
        rejected = binarized[bin_idx]['rejected'][1]['content']
        if chosen == '' or rejected == '':
            continue
        chosen_ratings = dict()
        rejected_ratings = dict()
        flag = True
        for c in original[orig_idx]['completions']:
            if c['response'] == chosen:
                for key in keys:
                    r = c['annotations'][key]['Rating']
                    if r == 'N/A':
                        flag = False
                        continue
                    chosen_ratings[key] = int(r)
            elif c['response'] == rejected:
                for key in keys:
                    r = c['annotations'][key]['Rating']
                    if r == 'N/A':
                        flag = False
                        continue
                    rejected_ratings[key] = int(r)
            else:
                continue
        if not flag or len(chosen_ratings) != 4 or len(rejected_ratings) != 4:
            continue
        data_subsets, reversed_labels, is_equal = get_user_type(chosen_ratings, rejected_ratings, augment_type)
        if filter_equal:
            reversed_labels = {key: reversed_labels[key] for key in data_subsets if not is_equal[key]}
            data_subsets = [key for key in data_subsets if not is_equal[key]]
            is_equal = {key: False for key in data_subsets}
            if augment_type == '84' and len(is_equal.keys()) != 2:
                continue
        for data_subset in users.keys():
            if data_subset not in data_subsets:
                dumb_baseline[data_subset] += 0.5 * len(data_subsets)
                if True in reversed_labels.values() and False in reversed_labels.values():
                    dumb_controversial_baseline[data_subset] += 0.5 * len(data_subsets)
                continue
            user_counter[data_subset] += 1
            if True in reversed_labels.values() and False in reversed_labels.values():
                is_controversial = True
                controversial_counter += 1
            else:
                is_controversial = False
                agreed_counter += 1
            if reversed_labels[data_subset]:
                reversed_counter[data_subset] += 1
                dumb_baseline[data_subset] += list(reversed_labels.values()).count(True)
                if is_controversial:
                    dumb_controversial_baseline[data_subset] += list(reversed_labels.values()).count(True)
            else:
                dumb_baseline[data_subset] += list(reversed_labels.values()).count(False)
                if is_controversial:
                    dumb_controversial_baseline[data_subset] += list(reversed_labels.values()).count(False)
            dataset_dict['Index'].append(out_idx)
            dataset_dict['original_idx'].append(orig_idx)
            dataset_dict['prompt'].append(prompt)
            if not reversed_labels[data_subset]:
                dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
                dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
                
            else:
                dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
                dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
                
            dataset_dict['data_subset'].append(data_subset)
            dataset_dict['controversial'].append(is_controversial)
            dataset_dict['reversed'].append(reversed_labels[data_subset])
            satisfied_subset = set([key for key in users.keys() if key not in data_subsets or reversed_labels[key] == reversed_labels[data_subset]])
            dataset_dict['satisfied_subset'].append(satisfied_subset)
            dataset_dict['survey_options'].append(is_controversial and len(data_subsets) == len(users.keys()))
            out_idx += 1
    print(out_idx, agreed_counter, controversial_counter)
    print("User counter:", user_counter)
    print("Reversed counter:", reversed_counter)
    print("Dumb baseline:", dumb_baseline)
    print("Dumb controversial baseline:", dumb_controversial_baseline)
    return Dataset.from_dict(dataset_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-a', '--augment_type', type=str, default='single', help='How to augment data')
    parser.add_argument('-c', '--controversial_only', action='store_true', help='Whether to only generate controversial data')
    parser.add_argument('-n', '--name', type=str, default='P_4', help='name of dataset')
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.augment_type == 'single' :
        user_types = {
            '8': (1, 0, 0, 0),
            '4': (0, 1, 0, 0),
            '2': (0, 0, 1, 0),
            '1': (0, 0, 0, 1),
        }
    elif args.augment_type == '84':
        user_type={
            '8': (1, 0, 0, 0),
            '4': (0, 1, 0, 0),
        }
    else:
        raise ValueError('Invalid augment_type')

    ultra_feedback = load_dataset('openbmb/UltraFeedback')
    binarized_cleaned = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned')
    length = len(binarized_cleaned['train'])

    print(length)
    test_ids = list(np.random.choice(length, int(length * 0.1), replace=False))
    train_split = binarized_cleaned['train'].filter(lambda example, idx: idx not in test_ids, with_indices=True)
    test_split = binarized_cleaned['train'].filter(lambda example, idx: idx in test_ids, with_indices=True)
    print(len(train_split), len(test_split))
    print("start processing train split")
    joined_dataset_train = inner_join(ultra_feedback['train'], train_split, args.augment_type, user_types)
    print("start processing test split")
    joined_dataset_test = inner_join(ultra_feedback['train'], test_split, args.augment_type, user_types)

    output_dir = os.path.join('data', 'UltraFeedback_{}_{}'.format(args.augment_type, args.name))
    for user_type in user_types.keys():
        train_subset = joined_dataset_train.filter(lambda x: x['data_subset'] == user_type)
        test_subset = joined_dataset_test.filter(lambda x: x['data_subset'] == user_type)
        if args.controversial_only:
            train_subset = train_subset.filter(lambda x: x['controversial'] == True)
            test_subset = test_subset.filter(lambda x: x['controversial'] == True)
        print(user_types[user_type], len(train_subset), len(test_subset))
        train_subset.to_json(os.path.join(output_dir, user_type, 'train.jsonl'))
        test_subset.to_json(os.path.join(output_dir, user_type, 'test.jsonl'))


# NOTE: CODE FROM VPL : https://github.com/WEIRDLabUW/vpl_llm
# NOTE: python -m data.ultrafeedback_init -a single -n P_4 -c
