import torch
import torch.nn as nn
from torch.nn import functional as F

from glob import glob
import os
import numpy as np

class CalibrateMLP(nn.Module):

    def __init__(self, answer_tokens):
        super().__init__()
        dim = len(answer_tokens)
        self.c_fc    = nn.Linear(dim * 2, 8 * dim)
        self.gelu    = nn.GELU()
        self.c_out   = nn.Linear(8 * dim, dim)

    def forward(self, p1, p2):
        x = torch.cat((p1, p2), dim=-1)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_out(x)
        return x

def ECE_multidim_new(probs, labels, K=5, K_cross=5, smooth=False, sigma=0.1, collaborator_probs=None, confidence=True):
    """
    Computes ECE loss on probabilities and labels. 
    If smooth is False, uses ECE loss (K buckets).
    If smooth is True, uses smooth ECE loss (Gaussian weights centered at K points with std dev sigma).
    if collaborator_probs is not None, use it to compute cross ECE.
    """

    if confidence:
        # sample predictions from probs
        sampled_predictions = torch.distributions.Categorical(probs=probs).sample()
        conf_labels = torch.gather(labels, dim=-1, index=sampled_predictions.unsqueeze(-1))
        conf_labels = conf_labels.squeeze(-1)
        labels = conf_labels.unsqueeze(-1) # shape B x T x 1 (D = 1)
        # use max probabilities as predictions
        predictions = torch.max(probs, dim=-1).values.unsqueeze(-1) # shape B x T x 1 (D = 1)
        # TODO: specify collaborator predictions
    else:
        predictions = probs
        if collaborator_probs is not None:
            collaborator_predictions = collaborator_probs
    
    B, D = predictions.shape # prediction dimension
    N = B # number of examples
    num_buckets = K ** D # number of buckets

    flat_predictions = predictions.reshape(N, D)
    flat_labels = labels.to(dtype=probs.dtype).reshape(N, D)
            
    if smooth: # memory intensive
        # soft bucket assignment: define D-dimensional soft buckets using Gaussian kernels
        centers_1d = torch.linspace(0, 1, K, device=probs.device, dtype=probs.dtype)
        grids = torch.meshgrid([centers_1d] * D, indexing="ij")
        centers = torch.stack(grids, dim=-1).reshape(-1, D)  # shape K^D x D
        diff = predictions.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)  # shape B x T x K^D x D
        diff_norm = (torch.pow(diff, 2)).sum(dim=-1)  # shape B x T x K^D; L2 norm squared across D
        unnorm_weights = torch.exp(- (diff_norm) / (2 * sigma ** 2))
        weights = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if collaborator_probs is not None: # cross ECE buckets are joint buckets of the model and the collaborator
            # define D-dimensional soft buckets using Gaussian kernels
            centers_1d_collab = torch.linspace(0, 1, K_cross, device=probs.device, dtype=probs.dtype)
            grids_collab = torch.meshgrid([centers_1d_collab] * D, indexing="ij")
            centers_collab = torch.stack(grids_collab, dim=-1).reshape(-1, D)  # shape K^D x D
            diff_collab = collaborator_predictions.unsqueeze(-2) - centers_collab.unsqueeze(0).unsqueeze(0)  # shape B x T x K_cross^D x D
            diff_norm_collab = (torch.pow(diff_collab, 2)).sum(dim=-1)  # shape B x T x K_cross^D; L2 norm across D
            unnorm_weights_collab = torch.exp(- (diff_norm_collab) / (2 * sigma ** 2))
            weights_collaborator = unnorm_weights_collab / unnorm_weights_collab.sum(dim=-1, keepdim=True).clamp_min(1e-12) 

            joint_weights = weights_collaborator.unsqueeze(-1) * weights.unsqueeze(-2) # shape B x T x K_cross^D x K^D
            joint_weights = joint_weights.reshape(probs.size(0), probs.size(1), -1) # flatten to shape B x T x (K_cross^D*K^D)
            weights = joint_weights

        S = weights.sum(dim=(0, 1)) # sum of weights for each bucket

        p_bar = (weights.unsqueeze(-1) * predictions.unsqueeze(-2)).sum(dim=(0, 1)) / S.clamp_min(1e-12).unsqueeze(-1)
        y_bar = (weights.unsqueeze(-1) * labels.unsqueeze(-2)).sum(dim=(0, 1)) / S.clamp_min(1e-12).unsqueeze(-1)
    else: # memory efficient version
        # hard bucket assignment: each example gets one D-dimensional bucket
        edges_1d = torch.linspace(0, 1, K + 1, device=probs.device, dtype=probs.dtype)
        idx_per_coord = torch.bucketize(flat_predictions, edges_1d, right=True) - 1  # N x D: index of 1-dim bucket that prediction[n,d] is in
        idx_per_coord.clamp_(min=0, max=K - 1)
        powers = (K ** torch.arange(D, device=probs.device, dtype=torch.int64))  # D
        own_idx = (idx_per_coord.to(torch.int64) * powers).sum(dim=-1)  # N: index of D-dim bucket that prediction[n] is in
        if collaborator_probs is None:
            # use active buckets only
            active_bucket_ids, inverse = torch.unique(own_idx, sorted=False, return_inverse=True)
            M = active_bucket_ids.numel()  # M <= N

            S = torch.zeros(M, device=probs.device, dtype=probs.dtype)
            wp = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)
            wl = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)

            ones = torch.ones(N, device=probs.device, dtype=probs.dtype)
            S.scatter_add_(0, inverse, ones) # number of examples in each bucket
            inverse_exp = inverse.unsqueeze(-1).expand(-1, D) 
            wp.scatter_add_(0, inverse_exp, flat_predictions) # weighted sum of predictions in each bucket
            wl.scatter_add_(0, inverse_exp, flat_labels) # weighted sum of labels in each bucket
        else:
            flat_collab_predictions = collaborator_predictions.reshape(N, D)
            edges_1d_collab = torch.linspace(0, 1, K_cross + 1, device=probs.device, dtype=probs.dtype)
            collab_coord_idx = torch.bucketize(flat_collab_predictions, edges_1d_collab, right=True) - 1
            collab_coord_idx.clamp_(0, K_cross - 1)
            powers_collab = (K_cross ** torch.arange(D, device=probs.device, dtype=torch.int64))
            collab_idx = (collab_coord_idx.to(torch.int64) * powers_collab).sum(dim=-1)

            joint_idx_raw = own_idx + (K ** D) * collab_idx
            _, joint_inverse = torch.unique(joint_idx_raw, sorted=False, return_inverse=True) # index of joint bucket that prediction[n] is in

            M = int(joint_inverse.max().item()) + 1 # number of joint buckets

            S = torch.zeros(M, device=probs.device, dtype=probs.dtype)
            wp = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)
            wl = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)

            ones = torch.ones(N, device=probs.device, dtype=probs.dtype)
            S.scatter_add_(0, joint_inverse, ones)
            joint_exp = joint_inverse.unsqueeze(-1).expand(-1, D)
            wp.scatter_add_(0, joint_exp, flat_predictions)
            wl.scatter_add_(0, joint_exp, flat_labels)

        p_bar = wp / S.clamp_min(1e-12).unsqueeze(-1)
        y_bar = wl / S.clamp_min(1e-12).unsqueeze(-1)
        
    ece_loss = (S * torch.linalg.norm(p_bar - y_bar, ord=2, dim=-1)).sum() / N

    return ece_loss
    
def generate_calibration_data(maze_conversation_log_path, round):
    """
    Generate calibration data for a given round of the maze conversation logs.
    Args:
        maze_conversation_log_path: str, path to the maze conversation log
        round > 0: int, the round to generate calibration data for
    Returns:
        probs, collaborator_probs: list of probability vectors
        labels: list of correct answer tokens
    """
    probs = []
    collaborator_probs = []
    labels = []

    answer_to_indices = {'d': 0, 'r': 1, 'u': 2, 'l': 3}

    maze_paths = glob(os.path.join(maze_conversation_log_path, 'maze_*.npy'))
    for maze_path in maze_paths:
        maze_conversation_logs = np.load(maze_path, allow_pickle=True).item()
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                prob = conversation_log['prob_vectors'][round]
                collaborator_prob = conversation_log['prob_vectors'][round - 1]
                label_token = conversation_log['label']

                probs.append(prob)
                collaborator_probs.append(collaborator_prob)
                labels.append(answer_to_indices[label_token])
                
    # create train and val splits
    N = len(probs)
    split_idx = int(N * 0.9)
    probs_train = torch.tensor(probs[:split_idx], dtype=torch.float32)
    collaborator_probs_train = torch.tensor(collaborator_probs[:split_idx], dtype=torch.float32)
    labels_train = torch.tensor(labels[:split_idx], dtype=torch.int64)
    probs_val = torch.tensor(probs[split_idx:], dtype=torch.float32)
    collaborator_probs_val = torch.tensor(collaborator_probs[split_idx:], dtype=torch.float32)
    labels_val = torch.tensor(labels[split_idx:], dtype=torch.int64)

    print(f"Generated {probs_train.shape} training and {probs_val.shape} validation calibration data points for round {round}")
    return probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val

def get_batch(probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val, split, batch_size=256):
    """
    Get a batch of calibration data. Returns torch tensors of shape (batch_size, num_answer_tokens) for probs and collaborator_probs,
     and torch tensors of shape (batch_size, num_answer_tokens) for labels which are one-hot vectors of the answer.

    Args:
        split: str, 'train' or 'val'
        batch_size: int, the number of examples to get
    Returns:
        probs, collaborator_probs, labels: torch tensors of shape (batch_size, num_answer_tokens)
    """
    if split == 'train':
        probs = probs_train
        collaborator_probs = collaborator_probs_train
        labels = labels_train
    elif split == 'val':
        probs = probs_val
        collaborator_probs = collaborator_probs_val
        labels = labels_val
    
    N = len(probs) # number of data points
    example_idx = torch.randint(0, N, (batch_size,))
    print(example_idx.shape)
    print(probs.shape)
    print(collaborator_probs.shape)
    print(labels.shape)

    probs = probs[example_idx]
    collaborator_probs = collaborator_probs[example_idx]
    labels = labels[example_idx]
    
    return probs, collaborator_probs, labels


def train(model, maze_conversation_logs, round, num_iters=100, batch_size=256, learning_rate=1e-4):
    """
    Train the model on the calibration data.
    Args:
        model: nn.Module, the model to train
        maze_conversation_logs: dict of maze conversation logs
        round: int, the round to train on
        batch_size: int, the number of examples to train on
    Returns:
        model: nn.Module, the trained model
    """
    probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val = generate_calibration_data(maze_conversation_logs, round)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for i in range(num_iters):
        probs, collaborator_probs, labels = get_batch(probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val, 'train', batch_size)
        logits = model(probs, collaborator_probs)
        probs = F.softmax(logits, dim=-1)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if i % 10 == 0:
                print(f"Iteration {i}, train loss: {loss.item()}")
                # TODO: make labels a one hot vector for ece loss calculation
                ece_loss = ECE_multidim_new(probs, labels, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
                cross_ece_loss = ECE_multidim_new(probs, labels, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=False)
                print(f"Iteration {i}, train ECE loss: {ece_loss.item()}")
                print(f"Iteration {i}, train cross ECE loss: {cross_ece_loss.item()}")

                # evaluate on validation set
                probs_val, collaborator_probs_val, labels_val = get_batch(probs_val, collaborator_probs_val, labels_val, 'val', batch_size)
                logits_val = model(probs_val, collaborator_probs_val)
                probs_val = F.softmax(logits_val, dim=-1)
                val_loss = F.cross_entropy(logits_val, labels_val)
                val_ece_loss = ECE_multidim_new(probs_val, labels_val, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
                val_cross_ece_loss = ECE_multidim_new(probs_val, labels_val, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs_val, confidence=False)
                print(f"Iteration {i}, val loss: {val_loss.item()}")
                print(f"Iteration {i}, val ECE loss: {val_ece_loss.item()}")
                print(f"Iteration {i}, val cross ECE loss: {val_cross_ece_loss.item()}")
    return model


if __name__ == "__main__":
    maze_conversation_log_path = "out-api_exp1/test2/conversations"
    round = 1
    answer_tokens = ['d', 'r', 'u', 'l']
    # probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val = generate_calibration_data(maze_conversation_log_path, round)

    model = CalibrateMLP(answer_tokens)
    model = train(model, maze_conversation_log_path, round)