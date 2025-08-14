import os
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, cast
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb

from models.CoPL_gcf import CoPLGCF

def lr_lambda(current_step: int):
    """Warm-up and cosine decay learning rate schedule"""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = torch.tensor((current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    return 0.5 * (1.0 + torch.cos(progress * torch.pi))

class CoPL_Dataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Run GCNRM Model.")

    # General arguments
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension for GCN layers.")
    parser.add_argument('--l', type=int, default=4, help="Number of GCN layers.")
    parser.add_argument('--wandbon', action='store_true', help="Whether to use wandb.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--lambda_', type=float, default=1e-6, help="Lambda for regularization.")
    parser.add_argument('--save_prefix', type=str, default='', help="Path to save the model.")
    parser.add_argument('--num_epoch', type=int, default=100, help="Number of epochs.")
    return parser.parse_args()

def plot_latent(latent, user_type):
    """Plot latent space using TSNE"""
    z_embedding = TSNE(n_components=2, init='random', perplexity=10, learning_rate="auto").fit_transform(latent.cpu().numpy())
    colors = [f"C{int(i)}" for i in user_type]
    plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=colors)
    im = wandb.Image(plt)
    plt.close()
    return im

def adj_normalize(adj):
    """Normalize adjacency matrix"""
    rowD = adj.sum(1).to_dense().squeeze()
    colD = adj.sum(0).to_dense().squeeze()
    
    indices = adj.indices()
    for i in range(indices.size(1)):
        row_idx = indices[0, i]
        col_idx = indices[1, i]
        norm_factor = pow(rowD[row_idx] * colD[col_idx], 0.5)
        adj.values()[i] = adj.values()[i] / norm_factor if adj.values()[i] != 0 else 0

    print('normalization done')
    return adj

if __name__ == "__main__":
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    total, users_context_target = pickle.load(open(args.data_path, 'rb'))
    uids_to_usertype = {i['user']: int(i['user_type']) for i in users_context_target}
    
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
    
    # Create adjacency matrices
    pos_adj = torch.sparse_coo_tensor(torch.tensor([users, pos]), torch.ones(len(users)),
                                           size=(len(user_to_num_context), max_id+1), dtype=torch.float32).coalesce()
    neg_adj = torch.sparse_coo_tensor(torch.tensor([users, neg]), torch.ones(len(users)), 
                                           size=(len(user_to_num_context), max_id+1), dtype=torch.float32).coalesce()

    # Normalize adjacency matrices
    pos_adj = adj_normalize(pos_adj)
    neg_adj = adj_normalize(neg_adj)
    
    # Create datasets and dataloaders
    train_dataset = CoPL_Dataset(train_sample)
    val_dataset = CoPL_Dataset(valid_sample)
    
    n_u, n_i = pos_adj.shape

    train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=24)
    valid_loader = data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=24)
    
    # Initialize model
    model = CoPLGCF(n_u, n_i, args.hidden_dim, pos_adj.cuda(), neg_adj.cuda(), 0.1, l=args.l).cuda()
    
    # Initialize optimizer and tracking variables
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_val_acc = 0
    best_val_con_acc = 0
    best_val_noncon_acc = 0
    
    # Initialize wandb
    if args.wandbon:
        wandb.init(project='CoPL-GCF')
        wandb.run.name = f"CoPL-GCF_layer_{args.l}_" + args.save_prefix

    # Training parameters
    num_epoch = args.num_epoch
    total_steps = num_epoch * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    
    if 'UF-P-2' in args.data_path:
        save_name = 'UF-P-2'
    elif 'UF-P-4' in args.data_path:
        save_name = 'UF-P-4'
    elif 'TLDR' in args.data_path:
        save_name = 'TLDR'
    elif 'PLM' in args.data_path:
        save_name = 'PLM'
    else:
        raise ValueError(f"Invalid dataset: {args.data_path}")
    
    if 'AVG' in args.data_path:
        save_name += '-AVG'
    else:
        save_name += '-ALL'
    
    # Learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop
    for epoch in range(num_epoch):
        epoch_loss_reg = 0
        epoch_loss_seen = 0
        epoch_acc = []
        epoch_controversial_acc = []
        epoch_non_controversial_acc = []       
       
        # Training step
        for batch in tqdm(train_loader):
            c_uids, c_pos, c_neg, c_controversial = [x.cuda() if isinstance(x, torch.Tensor) else x for x in batch]
            c_uids = c_uids.long()
            c_pos = c_pos.long()
            c_neg = c_neg.long()

            optimizer.zero_grad()
            model.train()

            (loss_seen, loss_reg), (pos_scores, neg_scores) = model(c_uids, c_pos, c_neg)
            
            acc = (pos_scores > neg_scores).float()
            
            epoch_acc.extend(acc)
            epoch_controversial_acc.extend(acc[~c_controversial])
            epoch_non_controversial_acc.extend(acc[c_controversial])
            
            loss = loss_seen + args.lambda_ * loss_reg
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_seen += loss_seen.cpu().item()
            epoch_loss_reg += loss_reg.cpu().item()

            torch.cuda.empty_cache()

        # Calculate epoch metrics
        batch_no = len(train_loader)
        epoch_loss_seen /= batch_no
        epoch_loss_reg /= batch_no
        
        epoch_acc = torch.stack(epoch_acc).mean().item()
        epoch_controversial_acc = torch.stack(epoch_controversial_acc).mean().item() if epoch_controversial_acc else 0
        epoch_non_controversial_acc = torch.stack(epoch_non_controversial_acc).mean().item() if epoch_non_controversial_acc else 0
        
        print(f"Epoch: {epoch}, Loss: {epoch_loss_seen + args.lambda_*epoch_loss_reg:.4f}, Accuracy: {epoch_acc:.4f}, "
              f"Controversial_Accuracy: {epoch_controversial_acc:.4f}, Non_Controversial_Accuracy: {epoch_non_controversial_acc:.4f}")
        
        if args.wandbon:
            wandb.log({
                'train/Epoch': epoch,
                'train/Loss': epoch_loss_seen +args.lambda_*epoch_loss_reg,
                'train/Loss_reg': epoch_loss_reg,
                'train/Loss_seen': epoch_loss_seen,
                'train/Accuracy': epoch_acc,
                'train/Controversial_Accuracy': epoch_controversial_acc,
                'train/Non_Controversial_Accuracy': epoch_non_controversial_acc,
            })    

        # Validation step
        val_acc = []
        val_controversial_acc = []
        val_non_controversial_acc = []
        val_loss_seen = 0
        val_loss_reg = 0
        all_uids = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                c_uids, c_pos, c_neg, c_controversial = [x.cuda() if isinstance(x, torch.Tensor) else x for x in batch]
                c_uids = c_uids.long()
                c_pos = c_pos.long()
                c_neg = c_neg.long()
                
                all_uids.append(c_uids)

                (loss_seen, loss_reg), (pos_scores, neg_scores) = model(c_uids, c_pos, c_neg, test=True)
                        
                acc = (pos_scores > neg_scores).float()
                
                val_acc.extend(acc)
                val_controversial_acc.extend(acc[~c_controversial])
                val_non_controversial_acc.extend(acc[c_controversial])
                
                val_loss_seen += loss_seen.cpu().item()
                val_loss_reg += loss_reg.cpu().item()

        # Plot latent space
        unique_uids, unique_indices = torch.unique(torch.cat(all_uids), return_inverse=True, return_counts=False, dim=0)
        unique_user_types = [uids_to_usertype[i.item()] for i in unique_uids]
        user_emb = model.E_u[unique_uids]
        

        # Calculate validation metrics
        val_acc = torch.stack(val_acc).mean().item()
        val_controversial_acc = torch.stack(val_controversial_acc).mean().item() if val_controversial_acc else 0
        val_non_controversial_acc = torch.stack(val_non_controversial_acc).mean().item() if val_non_controversial_acc else 0
        val_loss_seen /= len(valid_loader)
        val_loss_reg /= len(valid_loader)

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_con_acc = val_controversial_acc
            best_val_noncon_acc = val_non_controversial_acc
            os.makedirs("gcf_models", exist_ok=True)
            os.makedirs(f"gcf_user_embeds", exist_ok=True)
            torch.save(model.state_dict(), f'gcf_models/{save_name}-split.pt')
            torch.save(user_emb, f'gcf_user_embeds/{save_name}-user_emb.pt')

            if args.wandbon:
                im = plot_latent(user_emb, unique_user_types)
                wandb.log({'valid/user_emb': im})
            
        print(f"Validation - Accuracy: {val_acc:.4f}, Controversial_Accuracy: {val_controversial_acc:.4f}, "
              f"Non_Controversial_Accuracy: {val_non_controversial_acc:.4f}")
        
        torch.cuda.empty_cache()
        
        if args.wandbon:
            wandb.log({
                'valid/Epoch': epoch,
                'valid/Accuracy': val_acc,
                'valid/Controversial_Accuracy': val_controversial_acc,
                'valid/Non_Controversial_Accuracy': val_non_controversial_acc,
                'valid/Loss': val_loss_seen + args.lambda_*val_loss_reg,
                'valid/Loss_seen': val_loss_seen,
                'valid/Loss_reg': val_loss_reg,
                'valid/best_acc': best_val_acc,
                'valid/best_con_acc': best_val_con_acc,
                'valid/best_noncon_acc': best_val_noncon_acc
            })
