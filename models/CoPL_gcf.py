import torch
import torch.nn.functional as F
import torch.nn as nn

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

class CoPLGCF(nn.Module):
    def __init__(self, n_u, n_i, d, pos_adj_norm, neg_adj_norm, dropout, l=3):
        super(CoPLGCF, self).__init__()
        
        torch.set_default_dtype(torch.float32)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.n_u, self.n_i = n_u, n_i
        self.pos_adj_norm = pos_adj_norm.float()  # Positive adjacency matrix (sparse)
        self.neg_adj_norm = neg_adj_norm.float()  # Negative adjacency matrix (sparse)

        self.l = l
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)).cuda()).float()
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)).cuda()).float()

        self.dropout = dropout
        self.E_u = None
        self.E_i = None


        # Transformation layers for positive and negative edges
        self.W_u_pos = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_pos = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_u_neg = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_neg = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])

        # Self-connection weights
        self.W_u_self = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_self = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])


    def forward(self, uids, pos, neg, test=False):
        E_u_prev = self.E_u_0
        E_i_prev = self.E_i_0

        for layer in range(self.l):
            # Message passing for positive edges
            if test is False:
                Z_u_pos = torch.spmm(sparse_dropout(self.pos_adj_norm, self.dropout), E_i_prev)
                Z_i_pos = torch.spmm(sparse_dropout(self.pos_adj_norm, self.dropout).transpose(0, 1), E_u_prev)
            else:
                Z_u_pos = torch.spmm(self.pos_adj_norm, E_i_prev)
                Z_i_pos = torch.spmm(self.pos_adj_norm.transpose(0, 1), E_u_prev)

            # Message passing for negative edges
            if test is False:
                Z_u_neg = torch.spmm(sparse_dropout(self.neg_adj_norm, self.dropout), E_i_prev)
                Z_i_neg = torch.spmm(sparse_dropout(self.neg_adj_norm, self.dropout).transpose(0, 1), E_u_prev)
            else:
                Z_u_neg = torch.spmm(self.neg_adj_norm, E_i_prev)
                Z_i_neg = torch.spmm(self.neg_adj_norm.transpose(0, 1), E_u_prev)

            # Feature transformation
            Z_u_transformed_pos = Z_u_pos
            Z_i_transformed_pos = Z_i_pos
            Z_u_transformed_neg = Z_u_neg
            Z_i_transformed_neg = Z_i_neg

            # Self-connection transformation
            E_u_self_transformed = self.W_u_self[layer](E_u_prev)
            E_i_self_transformed = self.W_u_self[layer](E_i_prev)

            # **NEW: Element-wise multiplication (Hadamard product) 추가**
            Z_u_hadamard_pos = self.W_i_pos[layer](Z_u_pos * E_u_prev)  
            Z_i_hadamard_pos = self.W_i_pos[layer](Z_i_pos * E_i_prev)  
            Z_u_hadamard_neg = self.W_i_neg[layer](Z_u_neg * E_u_prev)  
            Z_i_hadamard_neg = self.W_i_neg[layer](Z_i_neg * E_i_prev)  

            # Aggregate positive and negative contributions with Hadamard product
            E_u_curr = self.activation(
                Z_u_transformed_pos + Z_u_hadamard_pos - (Z_u_transformed_neg + Z_u_hadamard_neg) + E_u_self_transformed
            )
            E_i_curr = self.activation(
                Z_i_transformed_pos + Z_i_hadamard_pos - (Z_i_transformed_neg + Z_i_hadamard_neg) + E_i_self_transformed
            )

            # Update embeddings for the next layer
            E_u_prev = E_u_curr
            E_i_prev = E_i_curr

        # Final user and item embeddings
        if test is False:
            self.E_u = F.normalize(E_u_prev, dim=-1)
            self.E_i = E_i_prev
        else:
            self.E_u = F.normalize(E_u_prev.clone(), dim=-1)
            self.E_i = E_i_prev.clone()

        u_emb = E_u_prev[uids]
        pos_emb = E_i_prev[pos]
        neg_emb = E_i_prev[neg]

        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)

        loss_seen = -((pos_scores - neg_scores).sigmoid().log().clamp(-2000, 2000)).mean()
        loss_reg = (
            u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
        )

        return (loss_seen, loss_reg), (pos_scores, neg_scores)