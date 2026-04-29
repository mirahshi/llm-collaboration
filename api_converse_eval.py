import numpy as np
import os
from termcolor import colored
from glob import glob
import torch
from torch.nn import functional as F

def get_maze_conversation_logs(maze_conversation_log_path):
    maze_conversation_logs = {}
    maze_paths = glob(os.path.join(maze_conversation_log_path, 'maze_*.npy'))
    for maze_path in maze_paths:
        maze_conversation_logs_loaded = np.load(maze_path, allow_pickle=True).item()
        # append contents of maze_conversation_logs_loaded to maze_conversation_logs
        maze_conversation_logs.update(maze_conversation_logs_loaded)
    print("maze_conversation_logs: ", len(maze_conversation_logs))
    return maze_conversation_logs

def get_maze_conversation_logs_no_format_failures(maze_conversation_logs):
    """
    Filter maze_conversation_logs to only include mazes with no format failures. And remove mazes with no conversation logs.
    
    Args:
        maze_conversation_logs: dict (or numpy array containing dict) mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict with same structure, but only mazes where no move had any format failure
    """
    # Handle numpy array from np.load
    if isinstance(maze_conversation_logs, np.ndarray):
        maze_conversation_logs = maze_conversation_logs.item()
    
    maze_conversation_logs_no_format_failures = {}
    for i, conversation_logs in maze_conversation_logs.items():
        has_format_failure = False
        for conversation_log in conversation_logs:
            if any(conversation_log['format_failures']):
                has_format_failure = True
                break
        if not has_format_failure and len(conversation_logs) > 0:
            maze_conversation_logs_no_format_failures[i] = conversation_logs
    return maze_conversation_logs_no_format_failures

def measure_zero_one_losses(maze_conversation_logs):
    """
    Compute average 0/1 loss per round.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict mapping round number to average 0/1 loss
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])

    losses = {}
    for r in range(num_rounds):
        round_losses = []
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                final_answer = conversation_log['final_answers'][r]
                label = conversation_log['label']
                loss = 0 if final_answer == label else 1
                round_losses.append(loss)
        losses[r] = np.mean(round_losses)
    return losses

def measure_disagreement(maze_conversation_logs):
    """
    Compute average L2 distance between prob vectors of consecutive rounds.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict mapping round number r to average L2 distance between prob vectors at r-1 and r
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])

    distances = {}
    for r in range(1, num_rounds):
        round_distances = []
        for i, conversation_logs in maze_conversation_logs.items():
            for j, conversation_log in enumerate(conversation_logs):
                prob_prev = conversation_log['prob_vectors'][r - 1]
                prob_curr = conversation_log['prob_vectors'][r]
                if prob_prev is not None and prob_curr is not None:
                    l2_dist = np.linalg.norm(np.array(prob_prev) - np.array(prob_curr))
                    round_distances.append(l2_dist)
        if round_distances:
            distances[r] = np.mean(round_distances)
        else:
            distances[r] = None
    return distances


def measure_maze_success_rates(maze_conversation_logs):
    """
    Compute maze success rate per round.
    
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
    
    Returns:
        dict mapping round number to maze success rate (fraction of mazes with all moves correct)
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])

    success_rates = {}
    for r in range(num_rounds):
        successful_mazes = 0
        total_mazes = len(maze_conversation_logs)
        for i, conversation_logs in maze_conversation_logs.items():
            maze_correct = True
            for j, conversation_log in enumerate(conversation_logs):
                final_answer = conversation_log['final_answers'][r]
                label = conversation_log['label']
                if final_answer != label:
                    maze_correct = False
                    break
            if maze_correct:
                successful_mazes += 1
        success_rates[r] = successful_mazes / total_mazes
    return success_rates

def generate_calibration_data(maze_conversation_logs, round):
    """
    Generate calibration data for a given round of the maze conversation logs.
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
        round > 0: int, the round to generate calibration data for
    Returns:
        probs, collaborator_probs: list of probability vectors
        labels: token indices of the correct answer tokens
    """
    probs = []
    collaborator_probs = []
    labels = []

    answer_to_indices = {'d': 0, 'r': 1, 'u': 2, 'l': 3}

    for i, conversation_logs in maze_conversation_logs.items():
        for j, conversation_log in enumerate(conversation_logs):
            prob = conversation_log['prob_vectors'][round]
            # if round > 1: # for round 2 and beyond, use calibrated probabilities from previous round
            #     collaborator_prob = conversation_log['calibrated_prob_vectors'][round - 1]
            # else: # for round 1, use raw probabilities from round 0
            if round > 0:
                collaborator_prob = conversation_log['prob_vectors'][round - 1]
            else:
                collaborator_prob = None
            label_token = conversation_log['label']
            # print(f"collaborator_prob: {collaborator_prob}, prob: {prob}, label: {label_token}")

            probs.append(prob)
            collaborator_probs.append(collaborator_prob)
            labels.append(answer_to_indices[label_token])
    
    if round == 0:
        collaborator_probs = None
    else:
        collaborator_probs = torch.tensor(collaborator_probs, dtype=torch.float32)
                
    return torch.tensor(probs, dtype=torch.float32), collaborator_probs, torch.tensor(labels, dtype=torch.int64)

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

def measure_calibration_losses(maze_conversation_logs):
    """
    Compute the calibration losses for each round.
    Args:
        maze_conversation_logs: dict mapping maze index to list of conversation_logs (one per move)
    Returns:
        ece_losses: dict mapping round number to calibration loss
        cross_ece_losses: dict mapping round number to cross-calibration loss
    """
    first_maze_idx = next(iter(maze_conversation_logs))
    num_rounds = len(maze_conversation_logs[first_maze_idx][0]['full_responses'])
    answer_tokens = ['d', 'r', 'u', 'l']
    
    ece_losses = {}
    cross_ece_losses = {}
    for r in range(num_rounds):
        # print(f"round {r}")
        probs, collaborator_probs, labels = generate_calibration_data(maze_conversation_logs, r)
        labels_one_hot = F.one_hot(labels, num_classes=len(answer_tokens))
        ece_loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=None, confidence=False)
        ece_losses[r] = ece_loss.item()
        if r > 0:
            cross_ece_loss = ECE_multidim_new(probs, labels_one_hot, K=10, K_cross=10, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=False)
            cross_ece_losses[r] = cross_ece_loss.item()
    return ece_losses, cross_ece_losses

if __name__ == "__main__":
    group_name = "api_exp8"
    run_names = ["multiprompt-hard-input"]

    # Load and filter all conditions first
    all_logs = {}
    all_filtered_logs = {}
    for run_name in run_names:
        out_dir = f'/vast/projects/surbhig/multi-agent-collab/out-{group_name}/{run_name}'
        maze_conversation_logs_path = os.path.join(out_dir, 'conversations')
        maze_conversation_logs = get_maze_conversation_logs(maze_conversation_logs_path)
        # maze_conversation_logs = np.load(os.path.join(out_dir, 'maze_conversation_logs.npy'), allow_pickle=True).item()
        all_logs[run_name] = maze_conversation_logs
        all_filtered_logs[run_name] = get_maze_conversation_logs_no_format_failures(maze_conversation_logs)
        print(f"{run_name}: {len(all_filtered_logs[run_name])} mazes with no format failures")

    # Find intersection of maze indices with no format failures in all conditions
    common_maze_indices = set(all_filtered_logs[run_names[0]].keys())
    for run_name in run_names[1:]:
        common_maze_indices &= set(all_filtered_logs[run_name].keys())
    print(colored(f"Common mazes with no format failures in all conditions: {len(common_maze_indices)}", 'light_green'))
    print(colored(f"Total number of examples in common mazes with no format failures: {sum([len(all_filtered_logs[run_name][i]) for i in common_maze_indices])}", 'light_green'))

    # Evaluate each condition on the common set of mazes
    for run_name in run_names:
        print(colored(f"\nEvaluating {run_name} (on {len(common_maze_indices)} common mazes)", 'light_yellow'))
        
        # Filter to only common mazes
        filtered_logs = {i: all_filtered_logs[run_name][i] for i in common_maze_indices}
        
        losses = measure_zero_one_losses(filtered_logs)
        success_rates = measure_maze_success_rates(filtered_logs)
        disagreements = measure_disagreement(filtered_logs)
        ece_losses, cross_ece_losses = measure_calibration_losses(filtered_logs)

        print(colored("zero-one losses:", 'light_blue'))
        for r, loss in losses.items():
            print(f"Round {r}: {loss}")
        print(colored("maze success rates:", 'light_blue'))
        for r, success_rate in success_rates.items():
            print(f"Round {r}: {success_rate}")
        print(colored("disagreement:", 'light_blue'))
        for r, disagreement in disagreements.items():
            print(f"Round {r}: {disagreement}")
        print(colored("ece losses:", 'light_blue'))
        for r, ece_loss in ece_losses.items():
            print(f"Round {r}: {ece_loss}")
        print(colored("cross ece losses:", 'light_blue'))
        for r, cross_ece_loss in cross_ece_losses.items():
            print(f"Round {r}: {cross_ece_loss}")