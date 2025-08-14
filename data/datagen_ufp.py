import os
import random
from typing import cast

import numpy as np
import torch
import argparse
from transformers import AutoTokenizer
from scipy.sparse import coo_matrix
from datasets import concatenate_datasets
import subprocess
import pickle


from ultrafeedback_init import get_hh_rlhf_dataset, DataSubset, HHRLHFPreprocessor


def preprocess_data(train_dataset, eval_dataset, subset, num_users=10000, ratio_evalset = 0.1, AVG=False):
    """
    Optimized preprocessing of datasets and embeddings.

    Args:
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        max_context (int): Maximum context items for each user.
        num_users (int): Number of users.

    Returns:
        adj_norm: Normalized adjacency matrix (torch sparse tensor).
        users_context_target: Context and target items for each user.
    """
    # Combine datasets
    total = concatenate_datasets([train_dataset, eval_dataset])
    eval_set = set(np.random.choice(np.unique(total['original_idx']), int(len(np.unique(total['original_idx'])) * ratio_evalset), replace=False))
    
    itemidx_to_index = {}
    index_to_itemidx = {}
    item_to_itemidx = {}
    itemidx_to_item = {}

    # Process items
    item_count = 0

    for i, entry in enumerate(total):

        for key in ["chosen", "rejected"]:
            key_sent = entry[key]
            index = i if key == "chosen" else i + len(total)
                
            if key_sent not in item_to_itemidx:
                item_to_itemidx[key_sent] = item_count
                itemidx_to_index[item_count] = index

                itemidx_to_item[item_count] = {'ids': entry[f'{key}_ids'], 'mask': entry[f'{key}_mask'], 'full_text': key_sent, 'controversial': entry['controversial']}
                index_to_itemidx[index] = item_count
                item_count += 1
            else:
                index_to_itemidx[index] = item_to_itemidx[key_sent]

    # Group by user type and context/target items
    total_user_type = np.unique(total['user_type'])
    context_item = {user_type: [] for user_type in total_user_type}
    target_item = {user_type: [] for user_type in total_user_type}
 
    for i, entry in enumerate(total):
        if entry['original_idx'] not in eval_set:
            context_item[entry['user_type']].append(i)
        else:
            target_item[entry['user_type']].append(i)

    n_context = 0
    if subset == 'UF-P-4':
        group_users = {
            '8': list(range(0, int(num_users * 0.25))),
            '4': list(range(int(num_users * 0.25), int(num_users * 0.5))),
            '2': list(range(int(num_users * 0.5), int(num_users * 0.75))),
            '1': list(range(int(num_users * 0.75), num_users)),
        }
        n_context = 16
    elif subset == 'UF-P-2':
        group_users = {
            '8': list(range(0, int(num_users * 0.5))),
            '4': list(range(int(num_users * 0.5), num_users))
        }
        n_context = 8
    else:
        raise ValueError(f"Invalid subset: {subset}")
    
    # Build user-item index
    users_context_target = []
    

    for user_type in total_user_type:
        user_list = group_users[user_type]

        np.random.shuffle(target_item[user_type])

        user_splits = {user: [] for user in user_list}
        context_items = context_item[user_type]

        for user in user_list:
            user_context = []
            unseen_indices = []
            user_target = []
            if AVG:
                max_context_num = 2*n_context
                context_num = np.random.randint(1, max_context_num)
                temp = np.random.choice(context_items, context_num, replace=False)
                if context_num >= n_context:
                    context_indices = temp[:context_num-1]
                    unseen_indices = temp[context_num-1:]
                else:
                    context_indices = temp

            else:
                context_num = n_context
                temp = np.random.choice(context_items, context_num, replace=False)
                context_indices = temp[:context_num-1]
                unseen_indices = temp[context_num-1:]
            

            user_context.extend([(index_to_itemidx[idx], index_to_itemidx[idx+len(total)]) for idx in context_indices])
            if len(unseen_indices) > 0:
                user_target.extend([(index_to_itemidx[idx], index_to_itemidx[idx+len(total)]) for idx in unseen_indices])
            else:
                user_target = []
            
            user_test = []
            temp = np.random.choice(len(target_item[user_type]), 100, replace=False)
            for idx in temp:
                user_test.append((index_to_itemidx[idx], index_to_itemidx[idx+len(total)]))

            users_context_target.append({
                'user_type': user_type,
                'user': user,
                'context': user_context,
                'context_unseen': user_target,
                'target': user_test,
            })


    return users_context_target, itemidx_to_item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate UFP dataset with configurable parameters')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--other_subsets', type=str, default='UF-P-2', help='Other subsets to use')
    parser.add_argument('--tokenizer_name', type=str, default='google/gemma-2b', help='Tokenizer name')
    parser.add_argument('--model_name', type=str, default='google/gemma-2b', help='Model name')
    parser.add_argument('--num_users', type=int, default=10000, help='Number of users to generate')
    parser.add_argument('--AVG', type=bool, default=False, help='Average context items per user')
    
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Run ultrafeedback_init.py script
    data_path = 'data/UltraFeedback_single_P_4'
    if not os.path.exists(data_path):
        subprocess.run(['python', '-m', 'data.ultrafeedback_init', '-a', 'single', '-n', 'P_4', '-c'])
    
    ###### data generation ##########

    data_subset = 'all'

    if args.other_subsets == 'UF-P-2':
        other_subsets = '84'
    elif args.other_subsets == 'UF-P-4':
        other_subsets = 'single'
    else:
        raise ValueError(f"Invalid data subset: {args.other_subsets}")

    train_dataset = get_hh_rlhf_dataset(
        data_subset,
        "train",
        0,
        data_path=data_path,
        other_subsets=other_subsets
    )
    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        0,
        data_path=data_path,
        other_subsets=other_subsets
    )
    
    print(len(train_dataset), len(eval_dataset))

    train_dataset = train_dataset.filter(lambda example: example['controversial'] == True)
    eval_dataset = eval_dataset.filter(lambda example: example['controversial'] == True)

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, add_eos_token=False)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.other_subsets == 'UF-P-2':
        train_controversial = [not set(['8', '4']).issubset(set(data['satisfied_subset'])) for data in train_dataset] 
        eval_controversial = [not set(['8', '4']).issubset(set(data['satisfied_subset'])) for data in eval_dataset]

        train_dataset = train_dataset.remove_columns(['controversial']).add_column('controversial', train_controversial)
        eval_dataset = eval_dataset.remove_columns(['controversial']).add_column('controversial', eval_controversial)

    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        HHRLHFPreprocessor(args, tokenizer),
        batched=True,
        num_proc=24,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: x["max_lengths"] <= 1024
    )
    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(args, tokenizer),
        batched=True,
        num_proc=24,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["max_lengths"] <= 1024
    )

    users_context_target, total = preprocess_data(train_dataset, eval_dataset, args.other_subsets, args.num_users, AVG=args.AVG)
    
    os.makedirs('dataset', exist_ok=True)
    name = f'dataset/{args.other_subsets}-{args.num_users}-AVG.pkl' if args.AVG else f'dataset/{args.other_subsets}-{args.num_users}-ALL.pkl'
    pickle.dump((total, users_context_target), open(name, 'wb'))
    print(f"Dataset saved to {name}")
    print(f"Parameters used: tokenizer={args.tokenizer_name}, other_subsets={args.other_subsets}, num_users={args.num_users}, AVG={args.AVG}, seed={args.seed}")

