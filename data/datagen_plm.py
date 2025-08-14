from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import random
import argparse
import os
import pickle


def list_score(i):
    # return [f'response_{i}_gemma_2b',
    # f'response_{i}_gemma_7b',
    # f'response_{i}_mistral_raft',
    # f'response_{i}_mistral_ray',
    # f'response_{i}_mistral_weqweasdas',
    # f'response_{i}_llama3_sfairx',
    # f'response_{i}_oasst_deberta_v3',
    # f'response_{i}_beaver_7b',
    # f'response_{i}_oasst_pythia_7b',
    # f'response_{i}_oasst_pythia_1b']
    return [
    f'response_{i}_gemma_7b',
    f'response_{i}_mistral_weqweasdas',
    f'response_{i}_llama3_sfairx',
    f'response_{i}_oasst_deberta_v3']
    


def preprocess_ds(tokenizer, num_item = 0):
    # follow the paper Li et al. 2024
    ds = load_dataset("namkoong-lab/PersonalLLM")
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()


    new_examples = {}
    pair_list_train = []
    pair_list_test = []
    num_item = num_item

    for _, row in df_train.iterrows():
        prompt = row['prompt']
        
        response1 = row['response_5']
        response2 = row['response_6']
        response3 = row['response_7']
        
        score1 = np.array([row[idx] for idx in list_score(5)])
        score2 = np.array([row[idx] for idx in list_score(6)])
        score3 = np.array([row[idx] for idx in list_score(7)])
        
        tokenized_text = tokenizer([response1, response2, response3])
        if len(tokenized_text['input_ids'][0]) > 1024 or len(tokenized_text['input_ids'][1]) > 1024 or len(tokenized_text['input_ids'][2]) > 1024:
            continue

        if not (all(score1>score2) or all(score1<score2)):
            pair_list_train.append([num_item, num_item+1])
        if not (all(score1>score3) or all(score1<score3)):
            pair_list_train.append([num_item, num_item+2])
        if not (all(score2>score3) or all(score2<score3)):
            pair_list_train.append([num_item+1, num_item+2])

        new_examples[num_item] = {'ids': tokenized_text['input_ids'][0],
                                  'mask': tokenized_text['attention_mask'][0],
                                  'text': response1,
                                  'score': score1}
        num_item += 1
        new_examples[num_item] = {'ids': tokenized_text['input_ids'][1],
                                    'mask': tokenized_text['attention_mask'][1],
                                    'text': response2,
                                    'score': score2}
        num_item += 1
        new_examples[num_item] = {'ids': tokenized_text['input_ids'][2],
                                    'mask': tokenized_text['attention_mask'][2],
                                    'text': response3,
                                    'score': score3}
        num_item += 1


    for _, row in df_test.iterrows():
        prompt = row['prompt']
        
        response1 = row['response_5']
        response2 = row['response_6']
        response3 = row['response_7']
        
        score1 = np.array([row[idx] for idx in list_score(5)])
        score2 = np.array([row[idx] for idx in list_score(6)])
        score3 = np.array([row[idx] for idx in list_score(7)])
        
        tokenized_text = tokenizer([response1, response2, response3])
        if len(tokenized_text['input_ids'][0]) > 1024 or len(tokenized_text['input_ids'][1]) > 1024 or len(tokenized_text['input_ids'][2]) > 1024:
            continue

        
        if not (all(score1>score2) or all(score1<score2)):
            pair_list_test.append([num_item, num_item+1])
        if not (all(score1>score3) or all(score1<score3)):
            pair_list_test.append([num_item, num_item+2])
        if not (all(score2>score3) or all(score2<score3)):
            pair_list_test.append([num_item+1, num_item+2])

        new_examples[num_item] = {'ids': tokenized_text['input_ids'][0],
                                  'mask': tokenized_text['attention_mask'][0],
                                  'text': response1,
                                  'score': score1}
        num_item += 1
        new_examples[num_item] = {'ids': tokenized_text['input_ids'][1],
                                    'mask': tokenized_text['attention_mask'][1],
                                    'text': response2,
                                    'score': score2}
        num_item += 1
        new_examples[num_item] = {'ids': tokenized_text['input_ids'][2],
                                    'mask': tokenized_text['attention_mask'][2],
                                    'text': response3,
                                    'score': score3}
        num_item += 1


    return pair_list_train, pair_list_test, new_examples



def assign_user(train_survey, test_survey, total, num_users = 10000, n_context=16, AVG=False):
    
    users_context_target = []
    users = []
    pos_items = []
    neg_items = []
    useridx_to_embedding = []

    random.seed(1111)

    for user in range(num_users):
        
        if AVG:
            max_context_num = n_context * 2
            context_num = np.random.randint(1, max_context_num)
            temp = np.random.choice(len(train_survey), context_num, replace=False)
            if context_num >= n_context:
                context_indices = temp[:context_num-1]
                unseen_indices = temp[context_num-1:]
            else:
                context_indices = temp
                unseen_indices = []
        else:
            context_num = n_context
            temp = np.random.choice(len(train_survey), context_num, replace=False)
            context_indices = temp[:context_num-1]
            unseen_indices = temp[context_num-1:]

        user_weight = np.random.dirichlet(alpha=[0.1]*4)
        users.extend([user] * len(context_indices))
        user_context = []
        user_target = []


        for idx in context_indices:
            left, right = train_survey[idx]
            score_left = (total[left]['score'] * user_weight).sum()
            score_right = (total[right]['score'] * user_weight).sum()

            if score_left > score_right:
                pos_items.append(left)
                neg_items.append(right)
                user_context.append((left, right))
            else:
                pos_items.append(right)
                neg_items.append(left)
                user_context.append((right, left))

        if len(unseen_indices) != 0:
            for idx in unseen_indices:
                left, right = train_survey[idx]
                score_left = (total[left]['score'] * user_weight).sum()
                score_right = (total[right]['score'] * user_weight).sum()

                if score_left > score_right:
                    user_target.append((left, right))
                else:
                    user_target.append((right, left))     
        else:
            unseen_indices = []

        temp = np.random.choice(len(test_survey), 100, replace=False)
        user_test = []
        for idx in temp:
            left, right = test_survey[idx]
            score_left = (total[left]['score'] * user_weight).sum()
            score_right = (total[right]['score'] * user_weight).sum()

            if score_left > score_right:
                user_test.append((left, right))
            else:
                user_test.append((right, left))
        
        users_context_target.append({
            'user': user,
            'context': user_context,
            'context_unseen': user_target,
            'target': user_test,
        })

    return users_context_target
                
def __main__():
    parser = argparse.ArgumentParser(description='Generate PLM dataset with configurable parameters')
    parser.add_argument('--tokenizer_name', type=str, default='google/gemma-2b-it',
                        help='Name of the tokenizer to use (default: google/gemma-2b-it)')
    parser.add_argument('--AVG', action='bool', default=False,
                        help='Use average context assignment (default: False)')
    parser.add_argument('--num_users', type=int, default=10000,
                        help='Number of users to generate (default: 10000)')
    parser.add_argument('--n_context', type=int, default=16,
                        help='Number of context items per user (default: 16)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed for reproducibility (default: 1111)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_auth_token=True, add_eos_token=False)
    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    pair_list_train, pair_list_test, total = preprocess_ds(tokenizer)
    users_context_target = assign_user(pair_list_train, pair_list_test, total, num_users=args.num_users, n_context=args.n_context, AVG=args.AVG)

    os.makedirs('dataset', exist_ok=True)
    name = f'dataset/PLM-AVG{args.n_context}.pkl' if args.AVG else f'dataset/PLM-ALL{args.n_context}.pkl'

    pickle.dump((total, users_context_target), open(name, 'wb'))
    print(f"Dataset saved to {name}")
    print(f"Parameters used: tokenizer={args.tokenizer_name}, avg={args.AVG}, num_users={args.num_users}, n_context={args.n_context}, seed={args.seed}")

if __name__ == '__main__':
    __main__()
