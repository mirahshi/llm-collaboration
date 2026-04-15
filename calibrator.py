import torch
import torch.nn as nn
from torch.nn import functional as F

from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from api_converse_eval import get_maze_conversation_logs_no_format_failures

class CalibrateMLP(nn.Module):

    def __init__(self, answer_tokens):
        self.answer_tokens = answer_tokens
        super().__init__()
        dim = len(self.answer_tokens)
        self.c_fc    = nn.Linear(dim * 2, 16 * dim)
        self.gelu    = nn.GELU()
        self.c_out   = nn.Linear(16 * dim, dim)

    def forward(self, p1, p2):
        x = torch.cat((p1, p2), dim=-1)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_out(x)
        return x


def load_calibrator(model_path):
    """
    Load a trained calibrator model from disk.
    
    Args:
        model_path: str, path to the saved model checkpoint
    Returns:
        model: CalibrateMLP, the loaded model in eval mode
        answer_tokens: list of answer tokens the model was trained on
        round: int, the round the model was trained for
    """
    checkpoint = torch.load(model_path, weights_only=False)
    answer_tokens = checkpoint['answer_tokens']
    model = CalibrateMLP(answer_tokens)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, answer_tokens, checkpoint['round']


def calibrate_probabilities(model, current_probs, prev_probs):
    """
    Apply the calibrator to transform probabilities.
    
    Args:
        model: CalibrateMLP, the trained calibrator model
        current_probs: list or tensor of shape (num_classes,), probabilities from current round
        prev_probs: list or tensor of shape (num_classes,), probabilities from previous round
    Returns:
        calibrated_probs: list of calibrated probabilities (sums to 1)
    """
    with torch.no_grad():
        if isinstance(current_probs, list):
            current_probs = torch.tensor(current_probs, dtype=torch.float32)
        if isinstance(prev_probs, list):
            prev_probs = torch.tensor(prev_probs, dtype=torch.float32)
        
        # Add batch dimension if needed
        if current_probs.dim() == 1:
            current_probs = current_probs.unsqueeze(0)
        if prev_probs.dim() == 1:
            prev_probs = prev_probs.unsqueeze(0)
        
        logits = model(current_probs, prev_probs)
        calibrated_probs = F.softmax(logits, dim=-1)
        
        return calibrated_probs.squeeze(0).tolist()


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
        diff = predictions.unsqueeze(-2) - centers.unsqueeze(0)  # shape B x K^D x D
        diff_norm = (torch.pow(diff, 2)).sum(dim=-1)  # shape B x K^D; L2 norm squared across D
        unnorm_weights = torch.exp(- (diff_norm) / (2 * sigma ** 2))
        weights = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # shape B x K^D

        if collaborator_probs is not None: # cross ECE buckets are joint buckets of the model and the collaborator
            # define D-dimensional soft buckets using Gaussian kernels
            centers_1d_collab = torch.linspace(0, 1, K_cross, device=probs.device, dtype=probs.dtype)
            grids_collab = torch.meshgrid([centers_1d_collab] * D, indexing="ij")
            centers_collab = torch.stack(grids_collab, dim=-1).reshape(-1, D)  # shape K_cross^D x D
            diff_collab = collaborator_predictions.unsqueeze(-2) - centers_collab.unsqueeze(0)  # shape B x K_cross^D x D
            diff_norm_collab = (torch.pow(diff_collab, 2)).sum(dim=-1)  # shape B x K_cross^D; L2 norm across D
            unnorm_weights_collab = torch.exp(- (diff_norm_collab) / (2 * sigma ** 2))
            weights_collaborator = unnorm_weights_collab / unnorm_weights_collab.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # shape B x K_cross^D

            joint_weights = weights_collaborator.unsqueeze(-1) * weights.unsqueeze(-2)  # shape B x K_cross^D x K^D
            joint_weights = joint_weights.reshape(N, -1)  # flatten to shape B x (K_cross^D * K^D)
            weights = joint_weights

        S = weights.sum(dim=0)  # sum of weights for each bucket

        p_bar = (weights.unsqueeze(-1) * predictions.unsqueeze(-2)).sum(dim=0) / S.clamp_min(1e-12).unsqueeze(-1)
        y_bar = (weights.unsqueeze(-1) * labels.unsqueeze(-2)).sum(dim=0) / S.clamp_min(1e-12).unsqueeze(-1)
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

def label_entries(maze_conversation_log_path, input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    label_sequences = [line.split('=')[1].strip() for line in lines]
    
    maze_paths = glob(os.path.join(maze_conversation_log_path, 'maze_*.npy'))
    for maze_path in maze_paths:
        maze_conversation_logs = np.load(maze_path, allow_pickle=True).item()
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                label = label_sequences[i][j]
                conversation_log['label'] = label
        np.save(maze_path, maze_conversation_logs)

    
def generate_calibration_data(maze_conversation_log_path, round):
    """
    Generate calibration data for a given round of the maze conversation logs.
    Args:
        maze_conversation_log_path: str, path to the maze conversation log
        round > 0: int, the round to generate calibration data for
    Returns:
        probs, collaborator_probs: list of probability vectors
        labels: token indices of the correct answer tokens
    """
    probs = []
    collaborator_probs = []
    labels = []

    answer_to_indices = {'d': 0, 'r': 1, 'u': 2, 'l': 3}

    # maze_paths = glob(os.path.join(maze_conversation_log_path, 'maze_*.npy'))
    maze_paths = [os.path.join(maze_conversation_log_path, f"maze_{0}.npy")]
    for maze_path in maze_paths:
        maze_conversation_logs = np.load(maze_path, allow_pickle=True).item()
        maze_conversation_logs = get_maze_conversation_logs_no_format_failures(maze_conversation_logs)
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                prob = conversation_log['prob_vectors'][round]
                if round > 1: # for round 2 and beyond, use calibrated probabilities from previous round
                    collaborator_prob = conversation_log['calibrated_prob_vectors'][round - 1]
                else: # for round 1, use raw probabilities from round 0
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

    print(f"Generated {probs_train.shape[0]} training and {probs_val.shape[0]} validation calibration data points for round {round}")
    return probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val

def base_calibration_losses(answer_tokens, probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val):
    """
    Compute the base calibration losses for the training and validation sets.
    Args:
        answer_tokens: list of answer tokens
        probs_train: torch tensor of shape (N, num_answer_tokens)
        collaborator_probs_train: torch tensor of shape (N, num_answer_tokens)
        labels_train: torch tensor of shape (N, num_answer_tokens)
    """
    labels_train_one_hot = F.one_hot(labels_train, num_classes=len(answer_tokens))
    train_ece_loss = ECE_multidim_new(probs_train, labels_train_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
    train_cross_ece_loss = ECE_multidim_new(probs_train, labels_train_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs_train, confidence=False)
    labels_val_one_hot = F.one_hot(labels_val, num_classes=len(answer_tokens))
    val_ece_loss = ECE_multidim_new(probs_val, labels_val_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
    val_cross_ece_loss = ECE_multidim_new(probs_val, labels_val_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs_val, confidence=False)
    return train_ece_loss, train_cross_ece_loss, val_ece_loss, val_cross_ece_loss
    

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

    probs = probs[example_idx]
    collaborator_probs = collaborator_probs[example_idx]
    labels = labels[example_idx]
    
    return probs, collaborator_probs, labels


def get_learning_rate(step, base_learning_rate, total_steps, lr_scheduler="cosine", warmup_iters=0, min_learning_rate=0.0):
    if lr_scheduler == "constant":
        return base_learning_rate

    warmup_iters = max(0, min(warmup_iters, total_steps - 1))
    if warmup_iters > 0 and step < warmup_iters:
        return base_learning_rate * (step + 1) / warmup_iters

    decay_steps = max(1, total_steps - warmup_iters)
    decay_progress = (step - warmup_iters) / decay_steps
    decay_progress = min(max(decay_progress, 0.0), 1.0)

    if lr_scheduler == "linear":
        return base_learning_rate - (base_learning_rate - min_learning_rate) * decay_progress
    if lr_scheduler == "cosine":
        cosine_coeff = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
        return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine_coeff

    raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}. Use one of: constant, linear, cosine.")


def train(
    model,
    maze_conversation_logs,
    round,
    use_smECE=False,
    num_iters=1000,
    batch_size=256,
    learning_rate=1e-4,
    lr_scheduler="cosine",
    warmup_iters=0,
    min_learning_rate=1e-5,
):
    """
    Train the model on the calibration data.
    Args:
        model: nn.Module, the model to train
        maze_conversation_logs: dict of maze conversation logs
        round: int, the round to train on
        use_smECE: bool, whether to use smooth ECE loss to train calibrator
        batch_size: int, the number of examples to train on
    Returns:
        model: nn.Module, the trained model
    """
    answer_tokens = model.answer_tokens
    probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val = generate_calibration_data(maze_conversation_logs, round)
    
    # get base calibration losses
    train_ece_loss_base, train_cross_ece_loss_base, val_ece_loss_base, val_cross_ece_loss_base = base_calibration_losses(answer_tokens, probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val)
    print(f"Base Train ECE loss: {train_ece_loss_base.item()}, Base Train cross ECE loss: {train_cross_ece_loss_base.item()}")
    print(f"Base Val ECE loss: {val_ece_loss_base.item()}, Base Val cross ECE loss: {val_cross_ece_loss_base.item()}")
    print(
        f"Learning rate schedule: {lr_scheduler} "
        f"(base_lr={learning_rate}, min_lr={min_learning_rate}, warmup_iters={warmup_iters})"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    ece_losses = []
    cross_ece_losses = []
    train_losses = []
    val_losses = []
    val_ece_losses = []
    val_cross_ece_losses = []

    for i in tqdm(range(num_iters)):
        current_lr = get_learning_rate(
            step=i,
            base_learning_rate=learning_rate,
            total_steps=num_iters,
            lr_scheduler=lr_scheduler,
            warmup_iters=warmup_iters,
            min_learning_rate=min_learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        probs, collaborator_probs, labels = get_batch(probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val, 'train', batch_size)
        logits = model(probs, collaborator_probs)
        probs = F.softmax(logits, dim=-1)

        if use_smECE:
            labels_one_hot = F.one_hot(labels, num_classes=len(answer_tokens))
            loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=False)
        else:
            loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if i % 10 == 0:
                labels_one_hot = F.one_hot(labels, num_classes=len(answer_tokens))
                ece_loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
                cross_ece_loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=False)

                # evaluate on validation set
                probs_eval, collaborator_probs_eval, labels_eval = get_batch(probs_train, collaborator_probs_train, labels_train, probs_val, collaborator_probs_val, labels_val, 'val', batch_size)
                logits_eval = model(probs_eval, collaborator_probs_eval)
                probs_eval = F.softmax(logits_eval, dim=-1)
                val_loss = F.cross_entropy(logits_eval, labels_eval)
                labels_eval_one_hot = F.one_hot(labels_eval, num_classes=len(answer_tokens))
                val_ece_loss = ECE_multidim_new(probs_eval, labels_eval_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
                val_cross_ece_loss = ECE_multidim_new(probs_eval, labels_eval_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs_eval, confidence=False)

                # log losses
                train_losses.append(loss.item())
                ece_losses.append(ece_loss.item())
                cross_ece_losses.append(cross_ece_loss.item())
                val_losses.append(val_loss.item())
                val_ece_losses.append(val_ece_loss.item())
                val_cross_ece_losses.append(val_cross_ece_loss.item())

                # # print losses
                # if i % 100 == 0:
                #     print(f"Train loss: {loss.item()}, Train ECE loss: {ece_loss.item()}, Train cross ECE loss: {cross_ece_loss.item()}")
                #     print(f"Val loss: {val_loss.item()}, Val ECE loss: {val_ece_loss.item()}, Val cross ECE loss: {val_cross_ece_loss.item()}")

    return model, ece_losses, cross_ece_losses, train_losses, val_losses, val_ece_losses, val_cross_ece_losses, train_ece_loss_base, train_cross_ece_loss_base, val_ece_loss_base, val_cross_ece_loss_base


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train calibrator for maze conversations.")
    parser.add_argument("--maze_conversation_log_path", type=str, default="out-api_exp3/probs-rollouts/conversations", help="Path to maze conversation logs")
    parser.add_argument("--out_dir", type=str, default="out-api_exp3/calibrator", help="Output directory for calibration results")
    parser.add_argument("--round", type=int, default=1, help="Round to train calibrator for")
    args = parser.parse_args()
    config = {
        "maze_conversation_log_path": args.maze_conversation_log_path,
        "out_dir": args.out_dir,
        "answer_tokens": ['d', 'r', 'u', 'l'],
        "round": args.round,
        "num_iters": 5000,
        "use_smECE": False, # use smooth ECE loss to train calibrator
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine", # options: constant, linear, cosine
        "warmup_iters": 0,
        "min_learning_rate": 1e-5,
    }

    # # if not already labeled, label the entries
    # input_file = "out-api_exp1/input0.txt"
    # label_entries(config["maze_conversation_log_path"], input_file)

    r = config["round"]
    print(f"Training calibrator for round {r}")

    model = CalibrateMLP(config["answer_tokens"])
    calibrated_model, ece_losses, cross_ece_losses, train_losses, val_losses, val_ece_losses, val_cross_ece_losses, train_ece_loss_base, train_cross_ece_loss_base, val_ece_loss_base, val_cross_ece_loss_base = train(
        model,
        maze_conversation_logs=config["maze_conversation_log_path"],
        round=r,
        use_smECE=config["use_smECE"],
        num_iters=config["num_iters"],
        learning_rate=config["learning_rate"],
        lr_scheduler=config["lr_scheduler"],
        warmup_iters=config["warmup_iters"],
        min_learning_rate=config["min_learning_rate"],
    )

    # plot losses
    os.makedirs(os.path.join(config["out_dir"], "plots"), exist_ok=True)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    if config["use_smECE"]:
        plt.savefig(f'{config["out_dir"]}/plots/calibrator_losses_smECE_round{r}.png')
    else:   
        plt.savefig(f'{config["out_dir"]}/plots/calibrator_losses_round{r}.png')
    plt.close()

    line1, = plt.plot(ece_losses, label='train ECE loss')
    line2, = plt.plot(val_ece_losses, label='val ECE loss')
    line3, = plt.plot(cross_ece_losses, label='train cross ECE loss')
    line4, = plt.plot(val_cross_ece_losses, label='val cross ECE loss')
    plt.plot([train_ece_loss_base] * len(ece_losses), label='train base ECE loss', linestyle='--', color=line1.get_color())
    plt.plot([val_ece_loss_base] * len(val_ece_losses), label='val base ECE loss', linestyle='--', color=line2.get_color())
    plt.plot([train_cross_ece_loss_base] * len(cross_ece_losses), label='train base cross ECE loss', linestyle='--', color=line3.get_color())
    plt.plot([val_cross_ece_loss_base] * len(val_cross_ece_losses), label='val base cross ECE loss', linestyle='--', color=line4.get_color())
    plt.legend()
    if config["use_smECE"]:
        plt.savefig(f'{config["out_dir"]}/plots/calibrator_ece_losses_smECE_round{r}.png')
    else:
        plt.savefig(f'{config["out_dir"]}/plots/calibrator_ece_losses_round{r}.png')
    plt.close()

    # Save the calibrated model
    model_path = os.path.join(config["out_dir"], f"calibrator_round{r}.pt")
    torch.save({
        'model_state_dict': calibrated_model.state_dict(),
        'answer_tokens': config["answer_tokens"],
        'round': r,
    }, model_path)
    print(f"Saved calibrated model to {model_path}")