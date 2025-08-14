import random
import numpy as np
import torch
import torch.utils.data as data
import argparse
import pickle

from models.CoPL_gcf import CoPLGCF 
from train_CoPL_gcf import CoPL_Dataset, adj_normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Run CoPLGCF Model Evaluation.")
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension for GCN layers.")
    parser.add_argument('--l', type=int, default=4, help="Number of GCN layers.")
    
    # File paths
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to the save file.")
    parser.add_argument('--unseen_data_path', type=str, required=True, help="Path to the unseen user data file.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data using pickle (consistent with train_CoPL_gcf.py)
    total, users_context_target = pickle.load(open(args.data_path, 'rb'))
    unseen_user_context_target = pickle.load(open(args.unseen_data_path, 'rb'))
    
    # Prepare training and validation data
    train_sample = []
    valid_sample = []
    user_to_num_context = {}
    
    users = []
    pos = []
    neg = []
    max_id = 0

    for user_context_target in users_context_target:
        user = user_context_target['user']
        context = user_context_target['context']
        target = user_context_target['context_unseen']
        
        user_to_num_context[user] = len(context)

        for temp in context:
            p, n = temp
            train_sample.append((user, p, n, False))
            users.append(user)
            pos.append(p)
            neg.append(n)
            max_id = max(max_id, p, n)
        
        if len(target) == 0:
            continue;

        for temp in target:
            p, n = temp
            valid_sample.append((user, p, n, False))
    
    # Create adjacency matrices (consistent with train_CoPL_gcf.py)
    pos_adj = torch.sparse_coo_tensor(torch.tensor([users, pos]), torch.ones(len(users)),
                                           size=(len(user_to_num_context), max_id+1), dtype=torch.float32).coalesce()
    neg_adj = torch.sparse_coo_tensor(torch.tensor([users, neg]), torch.ones(len(users)), 
                                           size=(len(user_to_num_context), max_id+1), dtype=torch.float32).coalesce()

    # Normalize adjacency matrices
    pos_adj = adj_normalize(pos_adj)
    neg_adj = adj_normalize(neg_adj)

    # Prepare unseen user data
    unseen_train_sample = []
    unseen_valid_sample = []
    unseen_users = []
    unseen_pos = []
    unseen_neg = []
    unseen_max_id = 0

    for user_context_target in unseen_user_context_target:
        user = user_context_target['user']
        context = user_context_target['context']
        target = user_context_target['context_unseen']
        
        for temp in context:
            p, n = temp
            unseen_train_sample.append((user, p, n, False))
            unseen_users.append(user)
            unseen_pos.append(p)
            unseen_neg.append(n)
            unseen_max_id = max(unseen_max_id, p, n)

        for temp in target:
            p, n = temp
            unseen_valid_sample.append((user, p, n, False))


    # Create unseen user adjacency matrices
    unseen_pos_adj = torch.sparse_coo_tensor(torch.tensor([unseen_users, unseen_pos]), torch.ones(len(unseen_users)),
                                           size=(len(unseen_user_context_target), max_id+1), dtype=torch.float32).coalesce()
    unseen_neg_adj = torch.sparse_coo_tensor(torch.tensor([unseen_users, unseen_neg]), torch.ones(len(unseen_users)), 
                                           size=(len(unseen_user_context_target), max_id+1), dtype=torch.float32).coalesce()

    # Normalize unseen adjacency matrices
    unseen_pos_adj = adj_normalize(unseen_pos_adj)
    unseen_neg_adj = adj_normalize(unseen_neg_adj)


    # Initialize model
    n_u, n_i = pos_adj.shape
    model = CoPLGCF(n_u, n_i, args.hidden_dim, pos_adj.cuda(), neg_adj.cuda(), 0.1, l=args.l).cuda()
    
    # Load pre-trained model
    model_dict = torch.load(args.model_path)
    model.load_state_dict(model_dict, strict=False)

    # Initialize embeddings
    model(0, 0, 0, test=True)  # Initialize E_u and E_i
    model.cuda()
    
    E_u = model.E_u.detach()
    E_i = model.E_i.detach()
    
    # Calculate two-hop adjacency for unseen users
    two_hop_adj = unseen_pos_adj.matmul(pos_adj.t()).cuda()
    two_hop_adj = (two_hop_adj.to_dense() > 0)
    
    two_hop_adj_mean = two_hop_adj.float() / two_hop_adj.float().sum(-1).unsqueeze(-1)
    naive_agg = two_hop_adj_mean @ E_u

    # Calculate scores for unseen users
    score = E_u @ E_i.T
    unseen_user_embeds = []
    
    for unseen_two_adj, user_context_target in zip(two_hop_adj, unseen_user_context_target):
        score_unseen = score[unseen_two_adj]
        context = user_context_target['context'] + user_context_target['context_unseen']
        loss = []
        
        for idx in context:
            p, n = idx
            loss.append((score_unseen[:, p] - score_unseen[:, n]).sigmoid().log())
        
        loss = torch.stack(loss).mean(0)
        softmax = torch.nn.functional.softmax(loss / 0.07)
        unseen_user_embeds.append((E_u[unseen_two_adj].T @ softmax))

    unseen_user_embeds = torch.stack(unseen_user_embeds).detach().cpu()
    
    # Save results
    torch.save(unseen_user_embeds, args.save_path)
    print(f"Unseen user embeddings saved to {args.save_path}")


if __name__ == "__main__":
    main()
