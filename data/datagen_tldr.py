import random
import pickle
import argparse
import os

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer


def generate_dataset_wo_embed(df, tokenizer, num_item = 0):
    seen_pair = set()
    
    item_to_itemidx = {}
    itemidx_to_item = {}
    num_item = num_item
    

    for _, row in df.iterrows():
        prompt_text, left_text, right_text = (
            row["info"]["post"],
            row["summaries"][0]["text"],
            row["summaries"][1]["text"],
        )
        if None in (prompt_text, left_text, right_text):
            continue
        left_len, right_len = len(left_text), len(right_text)
        
        if left_len < right_len:
            left_str = 'Human: ' + prompt_text + ' Assistant: ' + left_text
            right_str = 'Human: ' + prompt_text + ' Assistant: ' + right_text
        else:
            left_str = 'Human: ' + prompt_text + ' Assistant: ' + right_text
            right_str = 'Human: ' + prompt_text + ' Assistant: ' + left_text

        tokenized_text = tokenizer([left_str, right_str])

        if len(tokenized_text['input_ids'][0]) > 1024 or len(tokenized_text['input_ids'][1]) > 1024:
            continue
        

        if left_str not in item_to_itemidx:
            item_to_itemidx[left_str] = num_item
            
            item = {'ids': tokenized_text['input_ids'][0],
                    'mask': tokenized_text['attention_mask'][0],
                    'full_text': left_str,
                    'prompt': 'Human: ' + prompt_text + ' Assistant: ',
                    'answer': left_text}
            itemidx_to_item[num_item] = item
            num_item += 1

        if right_str not in item_to_itemidx:
            item_to_itemidx[right_str] = num_item
            item = {'ids': tokenized_text['input_ids'][1],
                    'mask': tokenized_text['attention_mask'][1],
                    'full_text': right_str,
                    'prompt': 'Human: ' + prompt_text + ' Assistant: ',
                    'answer': right_text}
            itemidx_to_item[num_item] = item
            num_item += 1
        
        pair = (item_to_itemidx[left_str], item_to_itemidx[right_str])
        
        if pair in seen_pair:
            continue
        
        seen_pair.add(pair)
        
    return list(seen_pair), itemidx_to_item

def selection_predicate(df):
    left_policies = df.summaries.apply(lambda x: x[0]["policy"])
    right_policies = df.summaries.apply(lambda x: x[1]["policy"])
    # select rows where ppo is not in the policy and sup is in the policy
    ppo_not_in_left = left_policies.apply(lambda x: "ppo" not in x)
    ppo_not_in_right = right_policies.apply(lambda x: "ppo" not in x)
    sup_in_left = left_policies.apply(lambda x: "sup" in x)
    sup_in_right = right_policies.apply(lambda x: "sup" in x)
    predicate = ppo_not_in_left & ppo_not_in_right & sup_in_left & sup_in_right
    return predicate

def preprocess_ds(tokenizer):
    # follow the paper Li et al. 2024
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", trust_remote_code=True)
    df_train = dataset["train"].to_pandas()
    df_test = dataset["validation"].to_pandas()
    df_train_selected = df_train[selection_predicate(df_train)]
    df_test_selected = df_test[selection_predicate(df_test)]
    # select the 10 workers who answered the most questions
    top_ten_workers = df_train_selected.worker.value_counts().head(10).index

    df_train_selected_top_ten_seen = df_train_selected[df_train_selected.worker.isin(top_ten_workers)]
    df_test_selected_top_ten_seen = df_test_selected[df_test_selected.worker.isin(top_ten_workers)]

    # pal-rm new dataset generation process
    survey_train, itemidx_to_item_train = generate_dataset_wo_embed(df_train_selected_top_ten_seen, tokenizer)
    survey_test, itemidx_to_item_test = generate_dataset_wo_embed(df_test_selected_top_ten_seen, tokenizer, len(itemidx_to_item_train))
    return survey_train, itemidx_to_item_train, survey_test, itemidx_to_item_test


def assign_user(train_survey, test_survey, num_users = 10000, n_cotext=8, AVG=False):
    
    total_user_type = [8, 4]
    group_size = num_users * 0.5

    # Create user groups dynamically
    group_users = {
        8: list(range(0, int(group_size))),
        4: list(range(int(group_size), num_users))
    }
    # remaining_users = num_users % len(total_user_type)
    # if remaining_users > 0:
    #     for i in range(remaining_users):
    #         group_users[total_user_type[i]].append(len(total_user_type) * group_size + i)
    
    users_context_target = []
    users = []
    pos_items = []
    neg_items = []
    useridx_to_embedding = []

    random.seed(1111)
    for user_type in total_user_type:
        user_list = group_users[user_type]
        user_splits = {user: [] for user in user_list}
        for user in user_list:

            if AVG:
                max_context_num = n_cotext*2
                context_num = np.random.randint(1, max_context_num)
                
                temp = np.random.choice(len(train_survey), context_num, replace=False)
                if context_num >= n_cotext:
                    context_indices = temp[:context_num-1]
                    unseen_indices = temp[context_num-1:]
                else:
                    context_indices = temp
                    unseen_indices = []            
            else:    
                context_num = n_cotext
                temp = np.random.choice(len(train_survey), context_num, replace=False)
                context_indices = temp[:context_num-1]
                unseen_indices = temp[context_num-1:]

            users.extend([user] * context_num)
            user_context = []
            user_target = []

            if user_type == 8: # 8: prefer the shorter summary, 4: prefer the longer summary
                for idx in context_indices:
                    left, right = train_survey[idx]
                    pos_items.append(left)
                    neg_items.append(right)
                    user_context.append((left, right))
                if len(unseen_indices) != 0:
                    for idx in unseen_indices:
                        left, right = train_survey[idx]
                        user_target.append((left, right))
                else:
                    unseen_indices = []
            else:
                for idx in context_indices:
                    left, right = train_survey[idx]
                    pos_items.append(right)
                    neg_items.append(left)
                    user_context.append((right, left))
                if len(unseen_indices) != 0:
                    for idx in unseen_indices:
                        left, right = train_survey[idx]
                        user_target.append((right, left))
                else:
                    unseen_indices = []
            
            # randomly sample 100 items from test_survey
            temp = np.random.choice(len(test_survey), 100, replace=False)
            user_test = []
            for idx in temp:
                left, right = test_survey[idx]
                user_test.append((left, right) if user_type == 8 else (right, left))

            users_context_target.append({
                'user_type': user_type,
                'user': user,
                'context': user_context,
                'context_unseen': user_target,
                'target': user_test,
            })
    return users_context_target


def __main__():
    parser = argparse.ArgumentParser(description='Generate TLDR dataset with configurable parameters')
    parser.add_argument('--tokenizer_name', type=str, default='google/gemma-2b-it',
                        help='Name of the tokenizer to use (default: google/gemma-2b-it)')
    parser.add_argument('--AVG', action='bool', default=False,
                        help='Use average context assignment (default: True)')
    parser.add_argument('--num_users', type=int, default=10000,
                        help='Number of users to generate (default: 10000)')
    parser.add_argument('--n_context', type=int, default=8,
                        help='Number of context items per user (default: random 1-16 for AVG=True, 8 for AVG=False)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed for reproducibility (default: 1111)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_auth_token=True, add_eos_token=False)
    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    survey_train, itemidx_to_item_train, survey_test, itemidx_to_item_test = preprocess_ds(tokenizer)
    itemidx_to_item_train.update(itemidx_to_item_test)
    
    total = itemidx_to_item_train
    users_context_target = assign_user(survey_train, survey_test, num_users=args.num_users, AVG=args.AVG, n_cotext=args.n_context)
    
    # Generate output filename if not provided
    os.makedirs('dataset', exist_ok=True)
    name = f'dataset/TLDR-AVG{args.n_context}.pkl' if args.avg else f'dataset/TLDR-ALL{args.n_context}.pkl'

    pickle.dump((total, users_context_target), open(name, 'wb'))
    print(f"Dataset saved to {name}")
    print(f"Parameters used: tokenizer={args.tokenizer_name}, avg={args.avg}, num_users={args.num_users}, context_num={args.n_context}, seed={args.seed}")


if __name__ == '__main__':
    __main__()
